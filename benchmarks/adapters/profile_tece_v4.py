"""Read-only TECE v4: normal four-way timing, separate exclusive-family profiles."""

import argparse
import gc
import hashlib
import inspect
import json
import os
import sys
import threading
from collections import defaultdict
from pathlib import Path

import torch
from benchmark_models import (
    assert_exclusive_gpu,
    build_data,
    build_model,
    configure,
    execute,
    one_batch,
    stats,
    validate_pair,
)
from fasteq.adapters import adapt_model
from mixed_snapshot import load_mixed
from profile_models import hardware_snapshot, summarize
from profile_tece64 import event_rows, shape
from v3_snapshot import load_v3


def snapshot(root):
    return {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for relative in ("src/FastEq/fasteq", "src/tace/tace/models", "src/tace/eqx", "src/prophet/prophet")
            for path in (root / relative).rglob("*")
            if path.is_file() and path.suffix in (".py", ".cu", ".cpp", ".h", ".hpp", ".cuh")
            and "__pycache__" not in path.parts and "generated" not in str(path)}


def tensors(value):
    if isinstance(value, torch.Tensor):
        return [value]
    if isinstance(value, (tuple, list)):
        return [tensor for item in value for tensor in tensors(item)]
    if isinstance(value, dict):
        return [tensor for item in value.values() for tensor in tensors(item)]
    return []


def family(module):
    cls, origin = type(module).__name__, type(module).__module__
    if isinstance(module, torch.nn.Linear) or cls in ("MLP", "mlpLinear", "ScalarReadOut"):
        return "scalar_mlp"
    if cls in ("Linear", "e3nnLinear", "e3nnElementLinear", "e3nnMoEElementLinear"):
        return "eq_linear"
    if "TensorProduct" in cls:
        return "tensor_product"
    if "LayerNorm" in cls or "RMSNorm" in cls:
        return "standalone_norm"
    if cls == "Gate":
        return "standalone_gate"
    if cls == "LayoutTransform":
        return "layout_misc"
    if cls in ("RadialBasis", "C2PolynomialCutoff", "j0SphericalBesselBasis", "OneHotToAtomicEnergy"):
        return "graph_geometry"
    if cls in ("SphericalHarmonics", "WignerD") or origin.endswith("tece_oam_rra") and cls == "Convolution":
        return "sh_rotation"
    return None


def annotate(model, backend, mode):
    """Hooks are profile-only and never change gradients or source files."""
    handles, restore, node_handles = [], [], []
    descriptors, tagged, node_records = {}, {}, []
    active = threading.local()

    def tag_nodes(output, inputs, owner, fam, initial_role=None):
        boundaries = {tensor.grad_fn for tensor in tensors(inputs) if tensor.grad_fn is not None}
        queue = [tensor.grad_fn for tensor in tensors(output) if tensor.grad_fn is not None]
        seen = set()
        while queue:
            node = queue.pop()
            if node is None or node in boundaries or node in seen or node in tagged:
                continue
            seen.add(node)
            if node.name().endswith("AccumulateGrad"):
                continue
            role = initial_role if "IndexedLinear" in node.name() else None
            tagged[node] = (owner, fam, role)
            node_records.append({"name": node.name(), "sequence": node._sequence_nr(),
                                 "module": owner, "family": fam, "linear_output_role": role})
            local_stack = defaultdict(list)

            def before(grads, _node=node, _owner=owner, _fam=fam, _role=role, _stack=local_stack):
                phase = "first" if mode != "force_loss" or torch.is_grad_enabled() else "second"
                suffix = f"_from_role{_role}" if _role is not None else ""
                key = f"family::{_fam}::{_owner}::{phase}{suffix}::{_node.name()}"
                context = torch.profiler.record_function(key)
                context.__enter__()
                _stack[threading.get_ident()].append(context)

            def after(grads_in, grads_out, _owner=owner, _fam=fam, _role=role, _stack=local_stack):
                if torch.is_grad_enabled():
                    roles = [value for value in range(3) if value != _role] if _role is not None else []
                    for index, gradient in enumerate(grads_in):
                        if gradient is not None and gradient.grad_fn is not None:
                            tag_nodes(gradient, (), _owner, _fam, roles[index] if index < len(roles) else None)
                _stack[threading.get_ident()].pop().__exit__(None, None, None)

            node_handles.extend((node.register_prehook(before), node.register_hook(after)))
            queue.extend(next_node for next_node, _ in node.next_functions)

    def wrapped_call(fn, owner, fam):
        def wrapped(*args, **kwargs):
            key = f"family::{fam}::{owner}::forward"
            row = descriptors.setdefault(key, {"count": 0, "first_args": None})
            row["count"] += 1
            if row["first_args"] is None:
                row["first_args"] = shape(args)
            old = getattr(active, "owner", None)
            active.owner = owner
            with torch.profiler.record_function(key):
                result = fn(*args, **kwargs)
                tag_nodes(result, args, owner, fam, 2 if fam == "eq_linear" else None)
            active.owner = old
            return result
        return wrapped

    for path, module in model.named_modules():
        fam = family(module)
        if fam is None:
            continue
        owner = f"{backend}/{path}"
        original = module.forward
        local = "forward" in module.__dict__
        old_local = module.__dict__.get("forward")
        module.forward = wrapped_call(original, owner, fam)
        restore.append((module, "forward", local, old_local))
        if fam == "eq_linear":
            plan = getattr(module, "_fasteq_plan", None)
            indexed = getattr(plan, "element", None) or getattr(module, "_fasteq_linear", None)
            descriptors[f"linear_descriptor::{owner}"] = {
                "class": type(module).__name__, "weight_numel": getattr(module, "weight_numel", None),
                "weight_shape": shape(getattr(module, "weight", None)),
                "input_dim": getattr(indexed, "input_dim", getattr(getattr(module, "irreps_in", None), "dim", None)),
                "output_dim": getattr(indexed, "output_dim", getattr(getattr(module, "irreps_out", None), "dim", None)),
                "paths": getattr(indexed, "paths", None),
            }
    # Loaded namespaces are distinct; only this profiling process is affected.
    for module in list(sys.modules.values()):
        name = getattr(module, "__name__", "")
        if not name.startswith(("fasteq.", "fasteq_v3.")):
            continue
        if name.endswith(".ops.rotation"):
            selected = [(key, "sh_rotation") for key in ("rotate", "rotary_product", "wigner_packed")]
        elif name.endswith(".ops.local_product"):
            selected = [("local_product", "tensor_product/local_product")]
        elif name.endswith(".ops.segmented_product"):
            selected = [("evaluate", "tensor_product/segmented")]
        else:
            continue
        for key, fam in selected:
            original = getattr(module, key, None)
            if original is None or not inspect.isfunction(original):
                continue

            def helper(*args, _fn=original, _fam=fam, _name=key, **kwargs):
                owner = getattr(active, "owner", None) or backend + "/helper"
                return wrapped_call(_fn, owner + "/" + _name, _fam)(*args, **kwargs)

            setattr(module, key, helper)
            restore.append((module, key, True, original))

    def undo():
        for handle in handles + node_handles:
            handle.remove()
        for obj, attribute, local, original in reversed(restore):
            if local:
                setattr(obj, attribute, original)
            else:
                delattr(obj, attribute)
    original = model.zero_grad
    model.zero_grad = wrapped_call(original, backend + "/zero_grad", "loss_zero_grad")
    restore.append((model, "zero_grad", True, original))
    return descriptors, node_records, undo


def partition(trace_path):
    events = json.loads(trace_path.read_text())["traceEvents"]
    ranges = defaultdict(list)
    cpu_ops = defaultdict(list)
    launches = {}

    def canonical(name):
        """Disambiguate identical _Product node names using their scoped owner."""
        parts = name.split("::")
        if len(parts) < 5:
            return name
        node = parts[-1]
        if node == "LocalSplitBackward" or node == "_ProductBackward" and parts[1] == "standalone_gate":
            parts[1] = "tensor_product/local_product"
        elif node == "_ProductBackward" and parts[1] == "tensor_product":
            parts[1] = "tensor_product/segmented"
        elif "FastUniform1d" in node:
            parts[1] = "tensor_product/uniform1d"
        return "::".join(parts)

    for event in events:
        if event.get("ph") != "X":
            continue
        if event.get("name", "").startswith("family::") and event.get("cat") == "user_annotation":
            event["name"] = canonical(event["name"])
            ranges[event["tid"]].append(event)
        if event.get("cat") == "cpu_op":
            cpu_ops[event["tid"]].append(event)
        if "LaunchKernel" in event.get("name", "") or "Memcpy" in event.get("name", "") or "Memset" in event.get("name", ""):
            correlation = event.get("args", {}).get("correlation")
            if correlation is not None:
                launches[correlation] = event

    def owner(event):
        launch = launches.get(event.get("args", {}).get("correlation"))
        if launch is None:
            return "unclassified", None
        matches = [row for row in ranges[launch["tid"]]
                   if row["ts"] <= launch["ts"] and row["ts"] + row["dur"] >= launch["ts"] + launch["dur"]]
        if not matches:
            operations = [row for row in cpu_ops[launch["tid"]]
                          if row["ts"] <= launch["ts"] and row["ts"] + row["dur"] >= launch["ts"] + launch["dur"]]
            if operations:
                operator = min(operations, key=lambda row: row["dur"])["name"]
                if any(token in operator for token in ("copy", "clone", "contiguous", "aten::cat", "empty", "zeros", "fill", "aten::to")):
                    return f"family::layout_misc::aten_fallback::{operator}", launch
                # An unscoped add/index/mul could belong to geometry, a loss,
                # layout, or gradient accumulation. Preserve the operator and
                # do not invent a model-level family from its kernel name.
                return f"family::unclassified::aten_fallback::{operator}", launch
            return "unclassified", launch
        deepest = min(matches, key=lambda row: row["dur"])
        return deepest["name"], launch

    groups = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    kernel_rows, copies = [], []
    for event in events:
        if event.get("ph") != "X" or event.get("cat") not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        key, launch = owner(event)
        if "uniform1d" in event["name"].lower() and key.startswith("family::tensor_product::"):
            key = key.replace("family::tensor_product::", "family::tensor_product/uniform1d::", 1)
        row = {"name": event["name"], "category": event["cat"], "ts": event["ts"],
               "duration_us": event["dur"], "owner": key, "args": event.get("args", {}),
               "runtime_launch": launch["name"] if launch else None}
        (kernel_rows if event["cat"] == "kernel" else copies).append(row)
        fam = key.split("::")[1] if key != "unclassified" else key
        group = groups[(event["cat"], fam)]
        group["count"] += 1
        group["duration_us"] += event["dur"]

    intervals = sorted((row["ts"], row["ts"] + row["duration_us"]) for row in kernel_rows)
    busy, end = 0.0, None
    for start, stop in intervals:
        busy += stop - max(start, end or start) if end is None or stop > end else 0
        end = max(end or stop, stop)
    span = max(stop for _, stop in intervals) - intervals[0][0] if intervals else 0
    cpu_partition = defaultdict(float)
    cpu_annotations = defaultdict(lambda: {"count": 0, "inclusive_cpu_us": 0.0})
    for tid, rows in ranges.items():
        for row in rows:
            cpu_annotations[row["name"]]["count"] += 1
            cpu_annotations[row["name"]]["inclusive_cpu_us"] += row["dur"]
        bounds = sorted({t for row in rows for t in (row["ts"], row["ts"] + row["dur"])})
        for first, last in zip(bounds, bounds[1:]):
            covered = [row for row in rows if row["ts"] <= first and row["ts"] + row["dur"] >= last]
            if covered:
                deepest = min(covered, key=lambda row: row["dur"])
                cpu_partition[deepest["name"].split("::")[1]] += last - first
    memory_rows = []
    for event in events:
        if event.get("name") != "[memory]" or event.get("args", {}).get("Device Type") != 1:
            continue
        candidates = [row for row in ranges[event["tid"]] if row["ts"] <= event["ts"] <= row["ts"] + row["dur"]]
        key = min(candidates, key=lambda row: row["dur"])["name"] if candidates else "unclassified"
        memory_rows.append({"bytes": event["args"]["Bytes"], "owner": key, "ts": event["ts"],
                            "total_allocated": event["args"]["Total Allocated"]})
    return {"exclusive_gpu_families": [{"category": key[0], "family": key[1], **value}
                                         for key, value in groups.items()],
            "kernel_rows": kernel_rows, "copy_rows": copies, "allocator_events": memory_rows,
            "kernel_union_busy_us": busy, "first_to_last_kernel_timeline_us": span,
            "timeline_gap_us": span - busy, "exclusive_annotated_cpu_intervals_us": dict(cpu_partition),
            "cpu_annotation_calls": [{"name": name, **value} for name, value in cpu_annotations.items()],
            "matched_kernel_count": sum(row["owner"] != "unclassified" and not row["owner"].startswith("family::unclassified::") for row in kernel_rows),
            "unclassified_kernel_count": sum(row["owner"] == "unclassified" or row["owner"].startswith("family::unclassified::") for row in kernel_rows),
            "direct_annotation_kernel_count": sum(row["owner"] != "unclassified" and "::aten_fallback::" not in row["owner"] for row in kernel_rows),
            "direct_annotation_kernel_duration_us": sum(row["duration_us"] for row in kernel_rows if row["owner"] != "unclassified" and "::aten_fallback::" not in row["owner"]),
            "postprocess_version": 3}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=160)
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=3)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument("--modes", nargs="+", default=["energy", "forces", "force_loss"])
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    assert_exclusive_gpu()
    args.output.mkdir(parents=True, exist_ok=True)
    frozen = snapshot(args.root)
    mixed, mixed_provenance = load_mixed(args.root)
    v3, v3_provenance = load_v3(args.root)
    labels = ("original", "mixed", "v3", "v4")
    report = {"metadata": {"model": "tece", "channels": 64, "nodes": 256, "edges": 3072,
             "torch": torch.__version__, "cuda": torch.version.cuda, "cpu_affinity": [args.cpu],
             "gpu_uuid": os.environ.get("CUDA_VISIBLE_DEVICES"), "initial_hardware": hardware_snapshot(),
             "source_sha256": frozen, "mixed": mixed_provenance, "v3": v3_provenance,
             "normal_protocol": {"warmup": args.warmup, "rounds": args.rounds, "iterations": args.iterations,
                                 "order": "four-way rotating/reversed order"},
             "scope": "benchmark_models.execute; full force-loss parameter backward; no optimizer/stress; prebuilt neighbors",
             "profile_note": "One instrumented step per backend; memory/shape/hooks enabled only in profiler. Inclusive nested CPU/GPU times must not be added; families use deepest launch annotation."}, "results": []}
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        original = build_model("tece", 64, dtype, 2)
        models = [original, mixed(original), v3(original), adapt_model(original)]
        data = build_data(256, 12, dtype)
        for mode in args.modes:
            print("CASE", dtype_name, mode, flush=True)
            assert snapshot(args.root) == frozen
            for model in models:
                configure(model, "tece", mode)
            assert_exclusive_gpu()
            validation = {name: validate_pair([models[0], model], "tece", data, mode, dtype)
                          for name, model in zip(labels[1:], models[1:], strict=True)}
            for model in models:
                for _ in range(args.warmup):
                    execute(model, "tece", data, mode)
            torch.cuda.synchronize()
            samples = [[], [], [], []]
            for block in range(args.rounds):
                assert_exclusive_gpu()
                order = [(block + i) % 4 for i in range(4)]
                if block % 2:
                    order.reverse()
                for i in order:
                    samples[i].append(one_batch(lambda m=models[i], mode=mode, data=data: execute(m, "tece", data, mode), args.iterations))
                assert_exclusive_gpu()
            case = {"dtype": dtype_name, "mode": mode, "validation": validation,
                    "normal_timing": {label: {"wall_ms": stats([s["wall_ms"] for s in values]),
                                      "cuda_event_ms": stats([s["cuda_event_ms"] for s in values]), "samples": values}
                                      for label, values in zip(labels, samples, strict=True)},
                    "paired_speedup": {f"{a}_over_{b}": stats([x["wall_ms"] / y["wall_ms"] for x, y in zip(samples[i], samples[j], strict=True)])
                                       for a, b, i, j in (("original", "v4", 0, 3), ("mixed", "v4", 1, 3), ("v3", "v4", 2, 3))},
                    "profiles": []}
            print("TIMING", dtype_name, mode, {k: v["wall_ms"]["median"] for k, v in case["normal_timing"].items()}, flush=True)
            for label, model in zip(labels, models, strict=True):
                assert_exclusive_gpu()
                model.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                unprofiled_resident = torch.cuda.memory_allocated()
                execute(model, "tece", data, mode)
                torch.cuda.synchronize()
                unprofiled_peak = torch.cuda.max_memory_allocated() - unprofiled_resident
                descriptors, nodes, undo = annotate(model, label, mode)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                resident = torch.cuda.memory_allocated()
                step = f"step::{dtype_name}::{mode}::{label}"
                print("PROFILE", step, flush=True)
                with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                                            record_shapes=True, profile_memory=True, with_stack=False) as profile:
                    with torch.profiler.record_function(step):
                        execute(model, "tece", data, mode)
                    torch.cuda.synchronize()
                undo()
                trace_path = args.output / f"{dtype_name}_{mode}_{label}.trace.json"
                profile.export_chrome_trace(str(trace_path))
                row = {"backend": label, **summarize(profile, trace_path, step), **partition(trace_path),
                       "shape_groups": event_rows(profile), "module_descriptors": descriptors, "node_records": nodes,
                       "allocated_resident_bytes": resident, "profile_peak_increment_bytes": torch.cuda.max_memory_allocated() - resident,
                       "unprofiled_peak_increment_bytes": unprofiled_peak}
                case["profiles"].append(row)
                assert_exclusive_gpu()
                assert snapshot(args.root) == frozen
            report["results"].append(case)
            (args.output / "profile.json").write_text(json.dumps(report, indent=2))
            gc.collect()
    report["metadata"]["final_hardware"] = hardware_snapshot()
    (args.output / "profile.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
