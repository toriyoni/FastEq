"""One fresh-process, exclusive-GPU scaling point, with auditable raw samples.

Paired points measure normal eager latency; individual capacity boundaries must
come from a single-backend worker. Operator force_loss is a feature-gradient
microchain, whereas model force_loss differentiates physical coordinates.
"""

import argparse
import gc
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path


class SharedGPUError(RuntimeError):
    pass


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def verify_manifest(path):
    """Reject drift in both the validated v6 implementation and new drivers."""
    manifest = json.loads(Path(path).read_text())
    if manifest.get("status") != "frozen_ready":
        raise RuntimeError("Source manifest is not frozen_ready")
    root = Path(manifest["root"])
    reference_path = Path(manifest["source_manifest_v6_path"])
    if hashlib.sha256(reference_path.read_bytes()).hexdigest() != manifest["source_manifest_v6_sha256"]:
        raise RuntimeError("Validated v6 manifest changed")
    reference = json.loads(reference_path.read_text())
    expected = {**reference["files_sha256"], **reference["driver_and_test_sha256"],
                **manifest["new_files_sha256"]}
    for relative, digest in expected.items():
        file = root / relative
        if not file.is_file() or hashlib.sha256(file.read_bytes()).hexdigest() != digest:
            raise RuntimeError("Source drift: " + str(file))
    return {"manifest_sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            "verified_files": len(expected)}


class ExclusiveGuard:
    def __init__(self, uuid):
        import pynvml

        self.nvml = pynvml
        self.nvml.nvmlInit()
        self.handle = self.nvml.nvmlDeviceGetHandleByUUID(uuid)
        self.uuid = uuid
        self.foreign_seen = []
        self.checks = 0

    def check(self):
        self.checks += 1
        processes = self.nvml.nvmlDeviceGetComputeRunningProcesses(self.handle)
        others = [int(value.pid) for value in processes if value.pid != os.getpid()]
        if others:
            self.foreign_seen.extend(others)
            raise SharedGPUError("Foreign GPU compute processes: " + str(others))
        memory = self.nvml.nvmlDeviceGetMemoryInfo(self.handle)
        return {"uuid": self.uuid, "total_bytes": int(memory.total),
                "used_bytes": int(memory.used), "free_bytes": int(memory.free),
                "foreign_pids": others, "guard_checks": self.checks}


def flatten_tensors(value, prefix="output"):
    """No concatenation or atom-scaled temporary is needed for tuple outputs."""
    if isinstance(value, dict):
        result = {}
        for key, item in value.items():
            result.update(flatten_tensors(item, prefix + "/" + str(key)))
        return result
    if isinstance(value, (tuple, list)):
        result = {}
        for index, item in enumerate(value):
            result.update(flatten_tensors(item, prefix + "/" + str(index)))
        return result
    return {prefix: value}


def execute_operator(case, label, mode, torch):
    mapping = {"original": "baseline", "v6": "candidate"}
    selected = mapping[label]
    inputs = case["inputs"]
    parameters = case[selected + "_parameters"]
    input_leaves = tuple(value for value in inputs if value.requires_grad)
    leaves = input_leaves + tuple(parameters)
    with torch.set_grad_enabled(mode != "forward"):
        output = case[selected](*inputs)
        if mode == "forward":
            return {"values": output}
        tensors = [value for value in flatten_tensors(output).values() if value is not None]
        # Squared norms of normalized SH / orthogonal Wigner blocks are
        # constant. The nonlinear term prevents a degenerate gradient chain.
        objective = sum(value.sin().mean() + 0.1 * value.square().mean() for value in tensors)
        if mode == "backward":
            gradients = torch.autograd.grad(objective, leaves, allow_unused=True)
            return {"values": output, "gradients": gradients[:len(input_leaves)],
                    "parameter_gradients": dict(enumerate(gradients[len(input_leaves):]))}
        first = torch.autograd.grad(objective, input_leaves, create_graph=True, allow_unused=True)
        terms = [value.square().mean() for value in first
                 if value is not None and value.requires_grad]
        if not terms:
            raise RuntimeError("Higher-order microchain has no differentiable input gradient")
        second = torch.autograd.grad(sum(terms), leaves, allow_unused=True)
        return {"values": output, "first": first, "gradients": second[:len(input_leaves)],
                "parameter_gradients": dict(enumerate(second[len(input_leaves):]))}


def is_parameter_gradient(name):
    return name.startswith(("parameter_grad/", "output/parameter_gradients/"))


def logical_sample(tensor, torch, sample_count):
    """Gather logical flat indices without flattening a noncontiguous view."""
    count = min(sample_count, tensor.numel())
    if count == tensor.numel():
        return tensor.detach().cpu().reshape(-1).to(torch.float64)
    indices = torch.arange(count, device=tensor.device, dtype=torch.long)
    indices = indices.mul(tensor.numel() - 1).div(count - 1, rounding_mode="floor")
    coordinates = []
    for dimension in reversed(tensor.shape):
        coordinates.append(indices.remainder(dimension))
        indices = indices.div(dimension, rounding_mode="floor")
    return tensor.detach()[tuple(reversed(coordinates))].cpu().to(torch.float64)


def verify_finite(output, parameters, torch):
    """Check full values in CPU slabs, without an atom-scaled GPU temporary."""
    tensors = flatten_tensors(output)
    tensors.update({"parameter_grad/" + name: parameter.grad for name, parameter in parameters})
    checked = elements = 0
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        detached = tensor.detach()
        if detached.ndim == 0:
            detached = detached.unsqueeze(0)
        row_elements = detached[0].numel() if detached.shape[0] else 1
        rows = max(1, 1048576 // max(1, row_elements))
        for slab in detached.split(rows, dim=0):
            host = slab.cpu()
            if not torch.isfinite(host).all():
                raise AssertionError("Nonfinite full tensor: " + name)
            elements += host.numel()
            del host
        checked += 1
    return {"status": "passed", "all_present_tensors_finite": True,
            "tensors_checked": checked, "elements_checked": elements,
            "scope": "one untimed execution; CPU slabs; no paired precision claim"}


def compact_snapshot(output, parameters, torch, sample_count=64):
    """Sample atom-scaled values; compare every named parameter gradient on CPU.

    Avoid retaining original GPU outputs while executing the candidate, which
    would incorrectly reduce its capacity. The 64-point sample is explicitly
    not a full large-array precision test; factories have full small-case QA.
    """
    result = {}
    for name, tensor in flatten_tensors(output).items():
        if tensor is None:
            result[name] = None
            continue
        sample = (tensor.detach().cpu().to(torch.float64) if is_parameter_gradient(name)
                  else logical_sample(tensor, torch, sample_count))
        result[name] = {"shape": list(tensor.shape), "values": sample}
    for name, parameter in parameters:
        result["parameter_grad/" + name] = (None if parameter.grad is None else
                                             {"shape": list(parameter.shape),
                                              "values": parameter.grad.detach().cpu().to(torch.float64)})
    return result


def compare_snapshots(reference, actual, dtype, torch):
    if tuple(reference) != tuple(actual):
        raise AssertionError("Snapshot keys differ")
    atol, rtol = (2e-4, 2e-3) if dtype == "float32" else (2e-8, 2e-7)
    max_abs = 0.0
    grad_diff = grad_norm = 0.0
    checked = 0
    for name, left in reference.items():
        right = actual[name]
        if left is None or right is None:
            if left is None and right is None:
                continue
            present = right if left is None else left
            if not is_parameter_gradient(name) or present["values"].count_nonzero():
                raise AssertionError("Gradient presence differs: " + name)
            continue
        if left["shape"] != right["shape"]:
            raise AssertionError("Tensor shape differs: " + name)
        a, b = left["values"], right["values"]
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError("Nonfinite sampled values: " + name)
        if a.numel():
            max_abs = max(max_abs, (a - b).abs().max().item())
        if is_parameter_gradient(name):
            grad_diff += (a - b).square().sum().item()
            grad_norm += a.square().sum().item()
        else:
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol, msg=name)
        checked += 1
    relative = (grad_diff / max(grad_norm, 1e-300)) ** 0.5
    if relative >= (2e-4 if dtype == "float32" else 2e-7):
        raise AssertionError("Parameter gradient relative L2: " + str(relative))
    return {"status": "passed", "sampled_values_per_atom_scaled_tensor": 64,
            "all_present_parameter_gradients_compared": any(is_parameter_gradient(name) for name in reference),
            "tensors_checked": checked, "max_sampled_absolute_error": max_abs,
            "parameter_gradient_relative_l2": relative,
            "scope": "Large tensors sampled; full small-case validation recorded separately"}


def run_point(args, result):
    # Set device/experimental flags before importing Torch, factories or plans.
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    os.environ["FASTEQ_EXPERIMENTAL_LINEAR_SIMT"] = "1"
    os.environ["FASTEQ_EXPERIMENTAL_NATIVE_MODULATION"] = "1"
    os.environ["FASTEQ_UNIFORM1D_FWD_TUNE"] = "0"
    os.environ["FASTEQ_UNIFORM1D_BWD_TUNE"] = "0"
    for name in ("EQX", "OEQ", "CUE", "EQT"):
        os.environ["TACE_USE_" + name] = "0"
    guard = ExclusiveGuard(args.gpu_uuid)
    result["gpu"] = guard.check()
    manifest = args.manifest or (Path(args.output).parent.parent / "manifest.json")
    result["source_guard"] = verify_manifest(manifest)

    import torch

    torch.set_num_threads(1)
    torch.manual_seed(199)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    result["runtime"] = {"torch": torch.__version__, "cuda": torch.version.cuda,
                         "device": torch.cuda.get_device_name(0), "tf32": False,
                         "allocator_memory_fraction": 1.0, "pid": os.getpid()}
    selected_uuid = str(torch.cuda.get_device_properties(0).uuid)
    if not selected_uuid.startswith("GPU-"):
        selected_uuid = "GPU-" + selected_uuid
    if selected_uuid != args.gpu_uuid:
        raise RuntimeError("Selected device UUID mismatch")
    labels = ["original", "v6"] if args.protocol == "paired" else [args.variant]

    def stage(value):
        result["stage"] = value
        guard.check()

    stage("setup")
    variant = None if args.protocol == "paired" else args.variant
    if args.kind == "operator":
        from scaling_operator_factories import build_case

        case = build_case(args.name, args.atoms, args.dtype, "cuda", variant=variant,
                          stage_callback=stage)
        result["metadata"] = case["metadata"]
        result["metadata"]["mode_scope"] = ("equivariant forward" if args.mode == "forward" else
                                            "operator feature-gradient microchain; not physical coordinate force loss")
        result["metadata"]["gradient_objective"] = "sum(mean(sin(output)) + 0.1*mean(output**2))"
        result["metadata"]["common_objective_and_autograd_included_in_gradient_timing"] = True
        callables = {label: lambda label=label: execute_operator(case, label, args.mode, torch)
                     for label in labels}
        parameter_lists = {label: [] for label in labels}
        clear = lambda: None
    else:
        from scaling_model_case import build_case

        case = build_case(args.name, args.atoms, args.dtype, "cuda", args.mode,
                          variant=variant, stage_callback=stage)
        result["metadata"] = case.metadata
        callables = case.callables
        parameter_lists = {label: list(case.models[label].named_parameters())
                           if args.mode == "force_loss" else [] for label in labels}
        clear = case.clear_gradients
    result["failure_variant"] = None
    torch.cuda.synchronize()
    setup_allocated = torch.cuda.memory_allocated()
    setup_reserved = torch.cuda.memory_reserved()
    result["setup_memory"] = {"allocated_bytes": setup_allocated,
                              "reserved_bytes": setup_reserved,
                              "scope": "two backends for paired timing; one backend for single capacity"}

    def invoke(label):
        result["failure_variant"] = label
        return callables[label]()

    # Warmup/compilation are excluded from timing. No profiler or dispatch hooks.
    pilot_ms = []
    for label in labels:
        stage("warmup")
        for index in range(args.warmup):
            guard.check()
            torch.cuda.synchronize()
            start = time.perf_counter()
            output = invoke(label)
            del output
            torch.cuda.synchronize()
            if index == args.warmup - 1:
                pilot_ms.append((time.perf_counter() - start) * 1000)
            guard.check()
        clear()

    if args.protocol == "paired":
        snapshots = {}
        for label in labels:
            stage("validation")
            output = invoke(label)
            snapshots[label] = compact_snapshot(output, parameter_lists[label], torch)
            del output
            clear()
            torch.cuda.synchronize()
            guard.check()
        result["validation"] = compare_snapshots(snapshots["original"], snapshots["v6"], args.dtype, torch)
        del snapshots
    else:
        stage("validation")
        output = invoke(labels[0])
        result["validation"] = verify_finite(output, parameter_lists[labels[0]], torch)
        del output
    clear()
    gc.collect()
    torch.cuda.synchronize()
    iterations = 3 if max(pilot_ms) < 50 else 1
    measurements = {label: {"wall_ms": [], "event_ms": [], "iterations": iterations,
                            "warmup": args.warmup, "peak_allocated_bytes": 0,
                            "peak_reserved_bytes": 0,
                            "incremental_peak_allocated_bytes": 0} for label in labels}
    result["measurements"] = measurements
    result["round_orders"] = []
    for index in range(args.rounds):
        order = labels if index % 2 == 0 else list(reversed(labels))
        result["round_orders"].append(order)
        for label in order:
            stage("timing_round_" + str(index))
            clear()
            torch.cuda.synchronize()
            base = torch.cuda.memory_allocated()
            torch.cuda.reset_peak_memory_stats()
            begin, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            begin.record()
            start = time.perf_counter()
            for _ in range(iterations):
                output = invoke(label)
                del output
            end.record()
            end.synchronize()
            wall = (time.perf_counter() - start) * 1000 / iterations
            event = begin.elapsed_time(end) / iterations
            measurement = measurements[label]
            measurement["wall_ms"].append(wall)
            measurement["event_ms"].append(event)
            measurement["peak_allocated_bytes"] = max(measurement["peak_allocated_bytes"],
                                                       torch.cuda.max_memory_allocated())
            measurement["peak_reserved_bytes"] = max(measurement["peak_reserved_bytes"],
                                                      torch.cuda.max_memory_reserved())
            measurement["incremental_peak_allocated_bytes"] = max(
                measurement["incremental_peak_allocated_bytes"], torch.cuda.max_memory_allocated() - base)
            guard.check()
    clear()
    result["gpu_after"] = guard.check()
    result["failure_variant"] = None
    if args.protocol == "paired":
        result["ratios"] = {}
        for metric in ("wall", "event"):
            left = measurements["original"][metric + "_ms"]
            right = measurements["v6"][metric + "_ms"]
            result["ratios"][metric] = (None if any(value <= 0 for value in right) else
                                         [a / b for a, b in zip(left, right, strict=True)])
    result["source_guard_end"] = verify_manifest(manifest)
    if result["source_guard"] != result["source_guard_end"]:
        raise RuntimeError("Source manifest drift during point")
    result["stage"] = "complete"
    result["status"] = "ok"


def classify_failure(error, stage):
    message = str(error)
    lowered = message.lower()
    # CPU OOM, illegal accesses, indexing limits and shared reservations are not
    # a model's genuine individual CUDA memory boundary.
    is_cuda_oom = any(fragment in lowered for fragment in (
        "cuda out of memory", "cuda_error_out_of_memory", "cudaerrormemoryallocation"))
    if isinstance(error, SharedGPUError):
        return "shared_gpu", "foreign_compute_process", False
    if is_cuda_oom and stage != "validation":
        return "oom", "cuda_oom", True
    if stage == "validation":
        return "error", "validation", False
    if isinstance(error, MemoryError) or "defaultcpuallocator" in lowered:
        return "error", "cpu_oom", False
    if "source drift" in lowered or "manifest" in lowered:
        return "error", "source_drift", False
    if "32-bit" in lowered or "index" in lowered and "overflow" in lowered:
        return "error", "index_limit", False
    return "error", "other", False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=("operator", "model"), required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--mode", required=True)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--protocol", choices=("paired", "single"), required=True)
    parser.add_argument("--variant", choices=("original", "v6"))
    parser.add_argument("--gpu", type=int, required=True)
    parser.add_argument("--gpu-uuid", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--warmup", type=int, default=5)
    args = parser.parse_args()
    valid_modes = {"operator": ("forward", "backward", "force_loss"),
                   "model": ("energy", "forces", "force_loss")}
    if args.mode not in valid_modes[args.kind] or args.atoms < 1 or args.rounds < 1 or args.warmup < 1:
        parser.error("Invalid mode, atoms, rounds or warmup")
    if args.protocol == "single" and args.variant is None:
        parser.error("Single-backend capacity requires --variant")
    if args.protocol == "paired" and args.variant is not None:
        parser.error("Paired timing cannot select one variant")
    result = {"id": args.output.stem, "kind": args.kind, "name": args.name,
              "dtype": args.dtype, "mode": args.mode, "atoms": args.atoms,
              "protocol": args.protocol, "variant": args.variant, "status": "running",
              "stage": "startup", "failure_variant": args.variant or "unknown",
              "measurements": {}, "ratios": None, "validation": None,
              "physical_gpu_index": args.gpu, "requested_rounds": args.rounds,
              "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    started = time.perf_counter()
    exit_code = 0
    try:
        run_point(args, result)
    except Exception as error:  # noqa: BLE001 -- all worker failures must be persisted.
        status, kind, is_oom = classify_failure(error, result["stage"])
        # A foreign process appearing during an allocation takes precedence
        # over the resulting CUDA OOM. Never attribute its reservation to us.
        try:
            guard = ExclusiveGuard(args.gpu_uuid)
            result["gpu_failure"] = guard.check()
        except SharedGPUError as shared:
            status, kind, is_oom = "shared_gpu", "foreign_compute_process", False
            result["foreign_gpu_error"] = str(shared)
        except Exception as guard_error:  # noqa: BLE001 -- never accept an unguarded OOM.
            status, kind, is_oom = "error", "exclusivity_unverified", False
            result["guard_error"] = str(guard_error)
        result["status"], result["error_kind"] = status, kind
        torch_module = sys.modules.get("torch")
        if torch_module is not None and torch_module.cuda.is_initialized():
            result["memory_failure"] = {
                "allocated_bytes": torch_module.cuda.memory_allocated(),
                "reserved_bytes": torch_module.cuda.memory_reserved(),
                "peak_allocated_bytes": torch_module.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch_module.cuda.max_memory_reserved(),
            }
        result["error"] = {"exception_type": type(error).__name__, "message": str(error),
                           "traceback": traceback.format_exc(limit=8)}
        if is_oom:
            result["oom"] = {"is_cuda_oom": True, "stage": result["stage"],
                             "exception_type": type(error).__name__, "message": str(error)}
        if status == "error":
            exit_code = 2
    result["elapsed_s"] = time.perf_counter() - started
    atomic_json(args.output, result)
    print(json.dumps({key: result.get(key) for key in (
        "id", "status", "stage", "failure_variant", "elapsed_s")}), flush=True)
    raise SystemExit(exit_code)


if __name__ == "__main__":
    main()
