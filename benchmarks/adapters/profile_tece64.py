"""Read-only TECE FP64 force-loss diagnosis beside verified mixed snapshot.

Normal paired timing is separate from the instrumented CPU/CUDA profile.
The benchmark_models construction and execute protocol remain unchanged.
"""

import argparse
import hashlib
import json
import os
import time
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
from fasteq.adapters import adapt_model, adaptation_report
from mixed_snapshot import load_mixed
from profile_models import hardware_snapshot, summarize


def snapshot(root):
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for relative in ("src/FastEq/fasteq", "src/tace/tace/models", "src/tace/eqx")
        for path in (root / relative).rglob("*")
        if path.is_file()
        and path.suffix in (".py", ".cpp", ".cu", ".h", ".cuh")
        and "generated" not in str(path)
    }


def shape(value):
    if isinstance(value, torch.Tensor):
        return {"shape": list(value.shape), "dtype": str(value.dtype),
                "stride": list(value.stride()), "requires_grad": value.requires_grad}
    if isinstance(value, (tuple, list)):
        return [shape(item) for item in value]
    if isinstance(value, dict):
        return {key: shape(item) for key, item in value.items()}
    return str(value)


def scope_modules(model, label):
    """Instance-only wrappers; retain source files and the original callable."""
    descriptors, restore = {}, []
    for path, module in model.named_modules():
        plan = getattr(module, "_fasteq_plan", None)
        if plan is None and type(module).__name__ not in (
            "Convolution", "TensorProduct", "Linear", "e3nnElementLinear",
            "e3nnMoEElementLinear", "EquivariantMergeLayerNorm",
            "EquivariantRMSNorm", "ProductBasisBlock", "EquivariantProductBasisBlock",
        ):
            continue
        original = module.forward
        key = "module::" + label + "::" + path
        descriptors[key] = {"class": type(module).__name__,
                            "backend": getattr(plan, "backend", None), "calls": []}

        def wrapped(*args, _fn=original, _key=key, **kwargs):
            row = descriptors[_key]
            if not row["calls"]:
                row["calls"].append({"args": shape(args), "kwargs": shape(kwargs)})
            with torch.profiler.record_function(_key):
                result = _fn(*args, **kwargs)
            return result

        had_local = "forward" in module.__dict__
        old_local = module.__dict__.get("forward")
        module.forward = wrapped
        restore.append((module, had_local, old_local))

    def undo():
        for module, local, value in restore:
            if local:
                module.forward = value
            else:
                del module.forward

    return descriptors, undo


def event_rows(profile):
    rows = []
    for event in profile.key_averages(group_by_input_shape=True):
        rows.append({"name": event.key, "input_shapes": event.input_shapes,
                     "count": event.count, "self_cpu_us": event.self_cpu_time_total,
                     "cpu_total_us": event.cpu_time_total,
                     "self_device_us": event.self_device_time_total,
                     "device_total_us": event.device_time_total})
    rows.sort(key=lambda row: row["self_device_us"], reverse=True)
    return rows


def kernel_owners(trace_path):
    events = json.loads(trace_path.read_text())["traceEvents"]
    cpu = {event["args"].get("External id"): event for event in events
           if event.get("ph") == "X" and event.get("cat") == "cpu_op"
           and event.get("args", {}).get("External id") is not None}
    grouped = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        owner = cpu.get(event.get("args", {}).get("External id"), {})
        key = (event["name"], owner.get("name", "unknown"),
               str(owner.get("args", {}).get("Input Dims", [])))
        row = grouped[key]
        row["count"] += 1
        row["duration_us"] += event["dur"]
    rows = [{"kernel": key[0], "owner": key[1], "input_dims": key[2], **value}
            for key, value in grouped.items()]
    return sorted(rows, key=lambda row: row["duration_us"], reverse=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=2)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert_exclusive_gpu()
    sources = snapshot(args.root)
    mixed, provenance = load_mixed(args.root)
    original = build_model("tece", 64, torch.float64, 2)
    models = [mixed(original), adapt_model(original)]
    labels = ("mixed", "fasteq")
    data = build_data(256, 12, torch.float64)
    for model in models:
        configure(model, "tece", "force_loss")
    report = {"metadata": {
        "model": "tece", "dtype": "float64", "channels": 64, "nodes": 256,
        "neighbors": 12, "edges": int(data["edge_index"].shape[1]),
        "mode": "force_loss", "scope": "benchmark_models.execute; no optimizer or stress",
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(), "uuid": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "hardware": hardware_snapshot(), "mixed": provenance,
        "frozen_sources": sources,
        "profile_note": "Instrumented durations diagnose work only; paired samples give normal latency.",
    }, "adaptation": adaptation_report(models[1]), "results": []}
    print("VALIDATE", flush=True)
    report["validation"] = validate_pair(models, "tece", data, "force_loss", torch.float64)
    warmup = []
    for label, model in zip(labels, models):
        print("WARMUP", label, flush=True)
        start = time.perf_counter()
        for _ in range(args.warmup):
            execute(model, "tece", data, "force_loss")
        torch.cuda.synchronize()
        warmup.append(time.perf_counter() - start)
    samples = [[], []]
    for index in range(args.rounds):
        assert_exclusive_gpu()
        for i in (0, 1) if index % 2 == 0 else (1, 0):
            samples[i].append(one_batch(lambda m=models[i]: execute(m, "tece", data, "force_loss"), args.iterations))
        assert_exclusive_gpu()
    report["normal_timing"] = {
        label: {"wall_ms": stats([s["wall_ms"] for s in values]),
                "cuda_event_ms": stats([s["cuda_event_ms"] for s in values]),
                "samples": values, "warmup_seconds": cold}
        for label, values, cold in zip(labels, samples, warmup)}
    print("TIMING", json.dumps(report["normal_timing"]), flush=True)
    for label, model in zip(labels, models):
        assert_exclusive_gpu()
        descriptors, undo = scope_modules(model, label)
        step = "profile_tece64_force_loss_" + label
        print("PROFILE", label, flush=True)
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
            record_shapes=True, profile_memory=False, with_stack=False,
        ) as profile:
            with torch.profiler.record_function(step):
                execute(model, "tece", data, "force_loss")
            torch.cuda.synchronize()
        undo()
        trace_path = args.output / (label + ".trace.json")
        profile.export_chrome_trace(str(trace_path))
        row = {"backend": label, **summarize(profile, trace_path, step),
               "shape_groups": event_rows(profile), "kernel_owners": kernel_owners(trace_path),
               "module_shapes": descriptors}
        report["results"].append(row)
        assert snapshot(args.root) == sources, "Sources changed during read-only diagnosis"
        (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")
        print("PROFILE_DONE", label, row["sum_cuda_kernel_duration_us"], flush=True)
    assert_exclusive_gpu()
    report["metadata"]["final_hardware"] = hardware_snapshot()
    (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
