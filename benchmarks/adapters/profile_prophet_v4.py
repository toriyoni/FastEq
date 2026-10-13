"""Four-way Prophet v4 diagnosis: normal timing and instrumented work are separate."""

import argparse
import gc
import hashlib
import inspect
import json
import os
import threading
from collections import defaultdict
from contextlib import contextmanager
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
from fasteq.adapters import adapt_model, adaptation_report
from mixed_snapshot import load_mixed
from profile_models import hardware_snapshot
from v3_snapshot import load_v3

LABELS = ("original", "mixed", "v3", "v4")


def source_hashes(root):
    return {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for name in (
            "src/FastEq/fasteq",
            "src/tace/tace/models",
            "src/tace/eqx",
            "src/prophet/prophet",
        )
        for p in (root / name).rglob("*")
        if p.is_file()
        and p.suffix in (".py", ".cpp", ".cu", ".h", ".cuh")
        and "__pycache__" not in p.parts
        and "generated" not in str(p)
    }


def shape(value):
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "stride": list(value.stride()),
            "requires_grad": value.requires_grad,
        }
    if isinstance(value, (tuple, list)):
        return [shape(x) for x in value]
    return type(value).__name__


@contextmanager
def scopes(model, label):
    """Add reversible markers; current production files and math stay unchanged."""
    restore, classes, object_tags, descriptors = [], set(), {}, {}
    metadata_tags = {}
    active = threading.local()

    def marker(fn, family, path, phase):
        key = f"family::{family}::{path}::{phase}"
        row = descriptors.setdefault(
            key, {"calls": 0, "source": inspect.getsourcefile(fn), "first_args": None}
        )

        def wrapped(*args, **kwargs):
            row["calls"] += 1
            if row["first_args"] is None:
                row["first_args"] = shape(args)
            stack = getattr(active, "stack", [])
            active.stack = [*stack, (family, path, phase)]
            try:
                with torch.profiler.record_function(key):
                    return fn(*args, **kwargs)
            finally:
                active.stack = stack

        return wrapped

    def tag(obj, family, path):
        if obj is None or isinstance(obj, (str, tuple)):
            return
        object_tags[id(obj)] = (family, path)
        classes.add(type(obj))

    gate_paths = [
        path for path, m in model.named_modules() if type(m).__name__ == "Gate"
    ]
    for path, module in model.named_modules():
        if isinstance(module, torch.jit.ScriptModule):
            continue
        plan = getattr(module, "_fasteq_plan", None)
        cls = type(module).__name__
        gate_child = any(
            path == prefix or path.startswith(prefix + ".") for prefix in gate_paths
        )
        family = (
            "standalone_gate"
            if gate_child
            else "tensor_product"
            if type(plan).__name__ == "TensorProductPlan" or "TensorProduct" in cls
            else "eq_linear"
            if type(plan).__name__ == "LinearPlan"
            or (
                cls == "Linear"
                and type(module).__module__.startswith(("e3nn.", "eqx."))
            )
            else "standalone_norm"
            if cls == "RMSLayerNorm"
            else "scalar_mlp"
            if isinstance(module, (torch.nn.Linear, torch.nn.LayerNorm))
            or cls in ("MLP", "ScalarMLP")
            else None
        )
        if family is not None:
            original, local = module.forward, "forward" in module.__dict__
            old = module.__dict__.get("forward")
            module.forward = marker(original, family, label + "." + path, "fwd")
            restore.append((module, "forward", local, old))
        elif cls in ("EQFFN", "ProphetBlock", "TPConvolution"):
            original, local = module.forward, "forward" in module.__dict__
            old = module.__dict__.get("forward")
            key = f"block::{label}::{path}::{cls}"

            def block(*args, _fn=original, _key=key, **kwargs):
                with torch.profiler.record_function(_key):
                    return _fn(*args, **kwargs)

            module.forward = block
            restore.append((module, "forward", local, old))
        if plan is not None:
            if cls == "EQFFN":
                tag(plan, "ffn_container", label + "." + path)
                tag(
                    getattr(plan, "norm", None),
                    "ffn_invariants",
                    label + "." + path + ".invariants",
                )
                tag(
                    getattr(plan, "modulate", None),
                    "ffn_modulation",
                    label + "." + path + ".modulation",
                )
                for attr, family in (
                    ("norm", "ffn_invariants"),
                    ("modulate", "ffn_modulation"),
                ):
                    metadata = getattr(plan, attr, None)
                    if isinstance(metadata, str):
                        metadata_tags[metadata] = (
                            family,
                            label + "." + path + "." + attr,
                        )
            else:
                tag(plan, family or "unclassified", label + "." + path)
            tag(
                getattr(plan, "element", None),
                "eq_linear",
                label + "." + path + ".element",
            )
        post = getattr(module, "_fasteq_postprocess", None)
        if post is not None:
            tag(post, "fused_gate_norm", label + "." + path + ".postprocess")
            tag(
                getattr(post, "fused", None),
                "fused_gate_norm",
                label + "." + path + ".postprocess.fused",
            )
            tag(
                getattr(post, "pre", None),
                "standalone_norm",
                label + "." + path + ".postprocess.pre",
            )
            tag(
                getattr(post, "post", None),
                "standalone_norm",
                label + "." + path + ".postprocess.post",
            )
            tag(
                getattr(post, "gate", None),
                "standalone_gate",
                label + "." + path + ".postprocess.gate",
            )

    modules = set()
    for cls in classes:
        modules.add(inspect.getmodule(cls))
        if "__call__" not in cls.__dict__ or issubclass(cls, torch.nn.Module):
            continue
        original = cls.__dict__["__call__"]

        def call(self, *args, _fn=original, **kwargs):
            family, path = object_tags.get(
                id(self), ("unclassified", label + "." + type(self).__name__)
            )
            return marker(_fn, family, path, "fwd")(self, *args, **kwargs)

        cls.__call__ = call
        restore.append((cls, "__call__", True, original))
    # Label custom Function phases on both original v3 and optimized modules.
    family_classes = {
        "_Norm": ("standalone_norm", "fwd", "first"),
        "_NormBackward": ("standalone_norm", "first", "second"),
        "_GateNorm": ("fused_gate_norm", "fwd", "first"),
        "_GateNormBackward": ("fused_gate_norm", "first", "second"),
        "_Modulation": ("ffn_modulation", "fwd", "first"),
        "_Gradient": ("ffn_modulation", "first", "second"),
        "_Norms": ("ffn_invariants", "fwd", "first"),
        "_First": ("ffn_invariants", "first", "second"),
    }
    for name, module in list(__import__("sys").modules.items()):
        if (
            name.startswith(("fasteq.", "fasteq_v3."))
            and module is not None
            and any(
                token in name
                for token in (
                    "equivariant_norm",
                    "gate_norm",
                    "modulation",
                    "irreps_invariants",
                )
            )
        ):
            modules.add(module)
    for module in modules:
        if module is None:
            continue
        original_run = getattr(module, "run", None)
        if original_run is not None and metadata_tags:

            def run(metadata, *args, _fn=original_run, **kwargs):
                family, path = metadata_tags.get(
                    metadata, ("framework_other", label + ".program_run")
                )
                return marker(_fn, family, path, "fwd")(metadata, *args, **kwargs)

            module.run = run
            restore.append((module, "run", True, original_run))
        for name, (family, forward_phase, backward_phase) in family_classes.items():
            cls = getattr(module, name, None)
            if not isinstance(cls, type):
                continue
            for method, phase in (
                ("forward", forward_phase),
                ("backward", backward_phase),
            ):
                original = cls.__dict__.get(method)
                if original is None:
                    continue
                fn = (
                    original.__func__
                    if isinstance(original, staticmethod)
                    else original
                )
                setattr(
                    cls,
                    method,
                    staticmethod(
                        marker(
                            fn,
                            family,
                            label + "." + module.__name__ + "." + name,
                            phase,
                        )
                    ),
                )
                restore.append((cls, method, True, original))
        for name in ("_reduce_rows", "_parameter_gradients"):
            original = getattr(module, name, None)
            if original is None:
                continue

            def reduce(*args, _fn=original, _module=module.__name__, **kwargs):
                stack = getattr(active, "stack", [])
                family, path, _ = (
                    stack[-1]
                    if stack
                    else ("standalone_norm", label + "." + _module, "")
                )
                return marker(
                    _fn, family, path + ".parameter_reduce", "parameter_reduce"
                )(*args, **kwargs)

            setattr(module, name, reduce)
            restore.append((module, name, True, original))
    try:
        yield descriptors
    finally:
        for obj, key, had, original in reversed(restore):
            if had:
                setattr(obj, key, original)
            else:
                delattr(obj, key)


def union_duration(intervals):
    intervals = sorted(intervals)
    if not intervals:
        return 0.0
    total = 0.0
    first, last = intervals[0]
    for begin, end in intervals[1:]:
        if begin > last:
            total += last - first
            first, last = begin, end
        else:
            last = max(last, end)
    return total + last - first


def infer_family(kernel, owner):
    name = kernel.lower()
    op = owner.lower()
    if "uniform1d" in name or "_product" in op:
        return "tensor_product"
    if (
        "indexing_" in name
        or "index_" in name
        or "scatter" in name
        or "sort" in name
        or "nonzero" in name
    ):
        return "layout_misc"
    if "direct_copy" in name or "copy_" in op or "clone" in op or "contiguous" in op:
        return "memory_copy"
    if any(
        x in op
        for x in (
            "index_select",
            "index_add",
            "aten::index",
            "aten::nonzero",
            "aten::_to_copy",
        )
    ):
        return "layout_misc"
    if any(
        x in op
        for x in (
            "addmm",
            "native_layer_norm",
            "silu",
            "tanh",
            "sigmoid",
            "mmbackward",
            "layernorm",
        )
    ):
        return "scalar_mlp"
    if any(
        x in op
        for x in ("indexlinear", "indexedlinear", "indexed_linear", "elementlinear")
    ):
        return "eq_linear"
    if "spherical_harmonic" in name or "scalar_expression" in op:
        return "sh_rotation"
    if "gate_double" in name or "gate_backward" in name or "gate_forward" in name:
        return "standalone_gate"
    if "gat en orm" in op.replace("_", " "):
        return "fused_gate_norm"
    if name in ("forward", "backward", "double_backward"):
        return "fused_gate_norm"
    if name in (
        "_forward_kernel",
        "_backward_kernel",
        "_double_kernel",
        "_tree_reduce_rows",
    ):
        return "standalone_norm"
    if "aten::" in op:
        return "framework_other"
    return "unclassified"


def analyze(profile, trace, step, descriptors):
    events = json.loads(trace.read_text())["traceEvents"]
    cpu = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("cpu_op", "user_annotation")
    ]
    gpu = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")
    ]
    runtime = [
        e
        for e in events
        if e.get("ph") == "X" and e.get("cat") in ("cuda_runtime", "cuda_driver")
    ]
    by_external = {
        e.get("args", {}).get("External id"): e
        for e in cpu
        if e.get("args", {}).get("External id") is not None
    }
    by_correlation = {
        e.get("args", {}).get("correlation"): e
        for e in runtime
        if e.get("args", {}).get("correlation") is not None
    }
    family_by_tid = defaultdict(list)
    for e in cpu:
        if e["name"].startswith("family::"):
            family_by_tid[e["tid"]].append(e)

    def enclosing(event):
        hits = [
            e
            for e in family_by_tid[event["tid"]]
            if e["ts"] <= event["ts"]
            and e["ts"] + e["dur"] >= event["ts"] + event.get("dur", 0)
        ]
        return min(hits, key=lambda e: e["dur"]) if hits else None

    owner_rows, groups = (
        defaultdict(lambda: {"count": 0, "duration_us": 0.0}),
        defaultdict(lambda: {"count": 0, "duration_us": 0.0}),
    )
    annotation_count = 0
    correlation_count = 0
    assignments = []
    for e in gpu:
        launch = by_correlation.get(e.get("args", {}).get("correlation"))
        owner = by_external.get(e.get("args", {}).get("External id"))
        annotation = enclosing(launch) if launch else None
        if annotation is None and owner:
            annotation = enclosing(owner)
        if annotation:
            _, family, path, phase = annotation["name"].split("::", 3)
            annotation_count += 1
        else:
            family, path, phase = (
                infer_family(e["name"], owner["name"] if owner else ""),
                "",
                "unknown",
            )
        if launch:
            correlation_count += 1
        key = (family, phase, e["cat"])
        groups[key]["count"] += 1
        groups[key]["duration_us"] += e["dur"]
        k = (family, phase, e["name"], owner["name"] if owner else "", path)
        owner_rows[k]["count"] += 1
        owner_rows[k]["duration_us"] += e["dur"]
        assignments.append(
            {
                "family": family,
                "phase": phase,
                "category": e["cat"],
                "start_us": e["ts"],
                "end_us": e["ts"] + e["dur"],
                "stream": e.get("args", {}).get("stream"),
                "duration_us": e["dur"],
            }
        )
    kernels = [e for e in gpu if e["cat"] == "kernel"]
    intervals = [(e["ts"], e["ts"] + e["dur"]) for e in gpu]
    streams = defaultdict(list)
    for e in gpu:
        streams[e.get("args", {}).get("stream")].append(e)
    timeline = []
    for stream, items in streams.items():
        start = min(e["ts"] for e in items)
        end = max(e["ts"] + e["dur"] for e in items)
        ordered = sorted(items, key=lambda e: e["ts"])
        gaps = []
        last = ordered[0]["ts"] + ordered[0]["dur"]
        for e in ordered[1:]:
            if e["ts"] > last:
                gaps.append(e["ts"] - last)
            last = max(last, e["ts"] + e["dur"])
        busy = union_duration([(e["ts"], e["ts"] + e["dur"]) for e in items])
        timeline.append(
            {
                "stream": stream,
                "gpu_activity_span_us": end - start,
                "busy_union_us": busy,
                "gap_us": end - start - busy,
                "gap_count": len(gaps),
                "largest_gap_us": max(gaps, default=0),
                "gap_quantiles_us": {
                    "median": float(__import__("numpy").median(gaps)),
                    "p95": float(__import__("numpy").percentile(gaps, 95)),
                }
                if gaps
                else {},
            }
        )
    cpu_rows = []
    for e in profile.key_averages(group_by_input_shape=True):
        if e.device_type != torch.autograd.DeviceType.CPU:
            continue
        cpu_rows.append(
            {
                "name": e.key,
                "count": e.count,
                "input_shapes": e.input_shapes,
                "self_cpu_us": e.self_cpu_time_total,
                "inclusive_cpu_us": e.cpu_time_total,
                "owned_kernel_duration_us": e.device_time_total,
                "self_cpu_memory_bytes": e.self_cpu_memory_usage,
                "self_cuda_memory_bytes": e.self_device_memory_usage,
                "inclusive_cuda_memory_bytes": e.device_memory_usage,
            }
        )
    cpu_rows.sort(key=lambda e: e["self_cpu_us"], reverse=True)
    memory = [
        e
        for e in events
        if e.get("name") == "[memory]" and e.get("args", {}).get("Device Type") == 1
    ]
    alloc = sum(
        e["args"].get("Bytes", 0) for e in memory if e["args"].get("Bytes", 0) > 0
    )
    freed = -sum(
        e["args"].get("Bytes", 0) for e in memory if e["args"].get("Bytes", 0) < 0
    )
    runtime_rows = defaultdict(lambda: {"count": 0, "self_api_duration_us": 0.0})
    for e in runtime:
        runtime_rows[e["name"]]["count"] += 1
        runtime_rows[e["name"]]["self_api_duration_us"] += e["dur"]
    transfers = defaultdict(lambda: {"count": 0, "duration_us": 0.0, "bytes": 0})
    for e in gpu:
        if e["cat"] != "kernel":
            transfers[e["name"]]["count"] += 1
            transfers[e["name"]]["duration_us"] += e["dur"]
            transfers[e["name"]]["bytes"] += e.get("args", {}).get("bytes", 0)
    return {
        "trace": str(trace),
        "kernel_count": len(kernels),
        "sum_kernel_duration_us": sum(e["dur"] for e in kernels),
        "activity_busy_union_us": union_duration(intervals),
        "stream_timeline": timeline,
        "family_groups": [
            {"family": key[0], "phase": key[1], "category": key[2], **row}
            for key, row in groups.items()
        ],
        "kernel_owners": [
            {
                "family": key[0],
                "phase": key[1],
                "kernel": key[2],
                "owner": key[3],
                "module_path": key[4],
                **row,
            }
            for key, row in sorted(
                owner_rows.items(), key=lambda x: x[1]["duration_us"], reverse=True
            )
        ],
        "attribution": {
            "gpu_event_count": len(gpu),
            "correlation_to_launch_count": correlation_count,
            "deepest_family_annotation_count": annotation_count,
            "unclassified_count": sum(
                row["count"] for key, row in groups.items() if key[0] == "unclassified"
            ),
            "unclassified_duration_us": sum(
                row["duration_us"]
                for key, row in groups.items()
                if key[0] == "unclassified"
            ),
        },
        "cpu_operations": cpu_rows,
        "cpu_total_self_us": sum(x["self_cpu_us"] for x in cpu_rows),
        "runtime_api": dict(runtime_rows),
        "transfers": dict(transfers),
        "cuda_memory": {
            "positive_allocation_event_bytes": alloc,
            "freed_event_bytes": freed,
            "allocation_event_count": sum(
                e["args"].get("Bytes", 0) > 0 for e in memory
            ),
            "peak_total_allocated_bytes": max(
                (e["args"].get("Total Allocated", 0) for e in memory), default=0
            ),
        },
        "markers": descriptors,
        "notes": [
            "Profiled GPU spans/gaps include instrumentation and are not normal idle measurements.",
            "Each raw GPU event is assigned to exactly one family/phase; nested inclusive CPU ranges must not be summed.",
            "CUDA event normal timing is a stream timeline, not a sum of kernels.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"])
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=3)
    args = parser.parse_args()
    assert args.rounds >= 3 and args.warmup >= 3 and args.iterations > 0
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert list(os.sched_getaffinity(0)) == [162]
    args.output.mkdir(exist_ok=True, parents=True)
    frozen = source_hashes(args.root)
    digest = hashlib.sha256(
        json.dumps(frozen, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    mixed, mixed_provenance = load_mixed(args.root)
    v3, v3_provenance = load_v3(args.root)
    report = {
        "metadata": {
            "model": "prophet",
            "channels": 64,
            "nodes": 256,
            "neighbors": 12,
            "layers": 2,
            "lmax": 2,
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "source_sha256": frozen,
            "source_digest": digest,
            "hardware_before": hardware_snapshot(),
            "mixed": mixed_provenance,
            "v3": v3_provenance,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "normal_protocol": {
                "rounds": args.rounds,
                "iterations": args.iterations,
                "warmup": args.warmup,
                "order": "four-way rotated/reversed within each round",
            },
            "profile_protocol": {
                "steps": 1,
                "activities": ["CPU", "CUDA"],
                "record_shapes": True,
                "profile_memory": True,
                "with_stack": False,
                "normal_timing_separate": True,
            },
            "scope": "cloned positions+geometry+model; forces computes actual energy-position derivatives; force_loss adds zero_grad/loss/all parameter backward; no optimizer/stress; neighbors prebuilt",
        },
        "results": [],
    }
    for dtype_name in ("float32", "float64"):
        dtype = getattr(torch, dtype_name)
        original = build_model("prophet", 64, dtype, 2)
        models = [original, mixed(original), v3(original), adapt_model(original)]
        data = build_data(256, 12, dtype)
        report.setdefault("adaptation", {})[dtype_name] = adaptation_report(models[3])
        for mode in ("energy", "forces", "force_loss"):
            assert source_hashes(args.root) == frozen
            print("START", dtype_name, mode, flush=True)
            for model in models:
                configure(model, "prophet", mode)
            assert_exclusive_gpu()
            validation = {
                label: validate_pair([models[0], model], "prophet", data, mode, dtype)
                for label, model in zip(LABELS[1:], models[1:], strict=True)
            }
            for model in models:
                for _ in range(args.warmup):
                    out = execute(model, "prophet", data, mode)
                    del out
                torch.cuda.synchronize()
            samples = [[] for _ in LABELS]
            for block in range(args.rounds):
                assert_exclusive_gpu()
                order = [(block + i) % 4 for i in range(4)]
                if block % 2:
                    order.reverse()
                for i in order:
                    samples[i].append(
                        one_batch(
                            lambda m=models[i], data=data, mode=mode: execute(
                                m, "prophet", data, mode
                            ),
                            args.iterations,
                        )
                    )
                assert_exclusive_gpu()
            normal = {
                label: {
                    "wall_ms": stats([s["wall_ms"] for s in ss]),
                    "cuda_event_ms": stats([s["cuda_event_ms"] for s in ss]),
                    "samples": ss,
                }
                for label, ss in zip(LABELS, samples, strict=True)
            }
            paired = {
                label + "_over_v4": stats(
                    [
                        x["wall_ms"] / y["wall_ms"]
                        for x, y in zip(ss, samples[3], strict=True)
                    ]
                )
                for label, ss in zip(LABELS[:3], samples[:3], strict=True)
            }
            case = {
                "dtype": dtype_name,
                "mode": mode,
                "source_digest_before": digest,
                "validation": validation,
                "normal_timing": normal,
                "paired_speedup": paired,
                "profiles": [],
            }
            print(
                "NORMAL",
                dtype_name,
                mode,
                json.dumps(
                    {label: normal[label]["wall_ms"]["median"] for label in LABELS}
                ),
                flush=True,
            )
            for label, model in zip(LABELS, models, strict=True):
                assert_exclusive_gpu()
                assert source_hashes(args.root) == frozen
                step = f"profile_prophet_v4::{dtype_name}::{mode}::{label}"
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                baseline_memory = torch.cuda.memory_allocated()
                with (
                    scopes(model, label) as descriptors,
                    torch.profiler.profile(
                        activities=[
                            torch.profiler.ProfilerActivity.CPU,
                            torch.profiler.ProfilerActivity.CUDA,
                        ],
                        record_shapes=True,
                        profile_memory=True,
                        with_stack=False,
                    ) as profile,
                ):
                    with torch.profiler.record_function(step):
                        out = execute(model, "prophet", data, mode)
                        del out
                    torch.cuda.synchronize()
                peak_memory = torch.cuda.max_memory_allocated()
                trace = args.output / (
                    dtype_name + "_" + mode + "_" + label + ".trace.json"
                )
                profile.export_chrome_trace(str(trace))
                summary = analyze(profile, trace, step, descriptors)
                summary.update(
                    {
                        "backend": label,
                        "allocator_baseline_bytes": baseline_memory,
                        "allocator_peak_bytes": peak_memory,
                        "allocator_peak_growth_bytes": peak_memory - baseline_memory,
                    }
                )
                case["profiles"].append(summary)
                print(
                    "PROFILE",
                    dtype_name,
                    mode,
                    label,
                    summary["kernel_count"],
                    round(summary["sum_kernel_duration_us"], 2),
                    summary["attribution"],
                    flush=True,
                )
                assert_exclusive_gpu()
                assert source_hashes(args.root) == frozen
            case["source_digest_after"] = digest
            case["hardware_after"] = hardware_snapshot()
            report["results"].append(case)
            (args.output / "profile.json").write_text(
                json.dumps(report, indent=2) + "\n"
            )
        del original, models, data
        gc.collect()
        torch.cuda.empty_cache()
    assert_exclusive_gpu()
    assert source_hashes(args.root) == frozen
    report["metadata"]["hardware_after"] = hardware_snapshot()
    (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
    print("COMPLETE", args.output, flush=True)


if __name__ == "__main__":
    main()
