"""New factory precision smoke; deliberately contains no performance timer."""

import argparse
import gc
import json
import os
import subprocess
import sys
import traceback
from pathlib import Path

os.environ["FASTEQ_EXPERIMENTAL_LINEAR_SIMT"] = "1"
os.environ["FASTEQ_EXPERIMENTAL_NATIVE_MODULATION"] = "1"

import torch
from scaling_operator_factories import SPECS, build_case, catalog


def tensors(value):
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors(item)]
    return [value]


def execute(fn, inputs, parameters):
    output = fn(*inputs)
    flat = tensors(output)
    objective = sum(value.sin().mean() + 0.1 * value.square().mean() for value in flat)
    leaves = (*inputs, *parameters)
    first = torch.autograd.grad(objective, leaves, create_graph=True, allow_unused=True)
    second_objective = sum(value.square().mean() for value in first[:len(inputs)]
                           if value is not None and value.requires_grad)
    if torch.is_tensor(second_objective):
        second = torch.autograd.grad(second_objective, leaves, allow_unused=True)
    else:
        second = (None,) * len(leaves)
    return output, first, second


def compare(left, right, dtype):
    absolute, relative = 0.0, 0.0
    atol, rtol = (3e-4, 4e-4) if dtype == torch.float32 else (3e-8, 3e-7)
    values_left, values_right = tensors(left), tensors(right)
    if len(values_left) != len(values_right):
        raise AssertionError("Output/gradient count mismatch")
    for actual, expected in zip(values_left, values_right, strict=True):
        if actual is None or expected is None:
            other = expected if actual is None else actual
            if other is not None:
                torch.testing.assert_close(other, torch.zeros_like(other), atol=atol, rtol=0)
            continue
        if not torch.isfinite(actual).all() or not torch.isfinite(expected).all():
            raise AssertionError("Nonfinite output/gradient")
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        if actual.numel():
            absolute = max(absolute, (actual - expected).abs().max().item())
            relative = max(relative, ((actual - expected).double().norm()
                                     / expected.double().norm().clamp_min(1e-30)).item())
    return {"max_abs": absolute, "max_relative_l2": relative}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operators", nargs="+", default=[spec.name for spec in SPECS])
    parser.add_argument("--atoms", type=int, default=2)
    parser.add_argument("--cpu-core", type=int, default=168)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    torch.cuda.set_per_process_memory_fraction(0.025)
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "tests" / "adapters"))
    from eqx_cuda_spy import forbid_eqx_cuda

    artifact = {
        "status": "shared_correctness_only", "performance_completed": False,
        "timing_fields_empty": True, "timings": [], "oom_probing": False,
        "atoms": args.atoms, "allocator_fraction": 0.025,
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu_processes": subprocess.check_output([
            "nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_gpu_memory", "--format=csv"], text=True),
        "torch": torch.__version__, "catalog": catalog(), "results": [], "errors": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for name in args.operators:
        for dtype in (torch.float32, torch.float64):
            torch.manual_seed(907)
            try:
                case = build_case(name, args.atoms, dtype)
                reference_inputs = tuple(value.detach().clone().requires_grad_() for value in case["inputs"])
                expected = execute(case["baseline"], reference_inputs, case["baseline_parameters"])
                with forbid_eqx_cuda():
                    actual = execute(case["candidate"], case["inputs"], case["candidate_parameters"])
                stats = compare(actual, expected, dtype)
                artifact["results"].append({"operator": name, "dtype": str(dtype),
                                            "metadata": case["metadata"], **stats})
                print("PASS", name, dtype, stats, flush=True)
                del case, reference_inputs, expected, actual
            except Exception as exc:  # noqa: BLE001 -- retain every failed factory.
                traceback.print_exc()
                artifact["errors"].append({"operator": name, "dtype": str(dtype), "error": repr(exc)})
            args.output.write_text(json.dumps(artifact, indent=2))
            gc.collect()
            torch.cuda.empty_cache()
    print("FACTORY_SMOKE", len(artifact["results"]), "passed", len(artifact["errors"]), "failed", flush=True)
    if artifact["errors"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
