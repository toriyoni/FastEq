"""Separate profiler diagnostics for Prophet adapters; not latency measurements.

Run after activating the isolated environment. Compilation and complete warmup
steps occur outside the CPU/CUDA profiler. Parameters are never updated.
"""

import argparse
import json
import os
import subprocess
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
    validate_pair,
)

from fasteq.adapters import adapt_model, adaptation_report


def hardware_snapshot():
    fields = "index,uuid,name,driver_version,memory.used,utilization.gpu,clocks.current.sm,clocks.current.memory,power.draw"
    return (
        subprocess.check_output(
            ["nvidia-smi", "--query-gpu=" + fields, "--format=csv,noheader"],
            text=True,
        )
        .strip()
        .splitlines()
    )


def summarize(profile, trace_path, label):
    # Chrome trace categories distinguish kernels from CUDA memcpy/memset.
    trace = json.loads(trace_path.read_text())
    events = trace["traceEvents"]
    kernels = [e for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]
    kernel_groups = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    for event in kernels:
        group = kernel_groups[event["name"]]
        group["count"] += 1
        group["duration_us"] += event["dur"]
    runtime_launch = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
    for event in events:
        if event.get("ph") == "X" and "LaunchKernel" in event.get("name", ""):
            group = runtime_launch[event["name"]]
            group["count"] += 1
            group["duration_us"] += event.get("dur", 0.0)
    cpu = []
    custom = []
    for event in profile.key_averages():
        row = {
            "name": event.key,
            "count": event.count,
            "self_cpu_us": event.self_cpu_time_total,
            "cpu_total_us": event.cpu_time_total,
            "self_device_us": event.self_device_time_total,
            "device_total_us": event.device_time_total,
        }
        if event.device_type == torch.autograd.DeviceType.CPU:
            cpu.append(row)
        if any(
            token in event.key.lower()
            for token in ("eqx::", "fasteq::", "uniform1d", "generatedbackward")
        ):
            custom.append(row)
    markers = [r for r in cpu if r["name"] == label]
    cpu.sort(key=lambda row: row["self_cpu_us"], reverse=True)
    custom.sort(key=lambda row: row["self_cpu_us"], reverse=True)
    grouped = sorted(
        [{"name": name, **values} for name, values in kernel_groups.items()],
        key=lambda row: row["duration_us"],
        reverse=True,
    )
    return {
        "trace": str(trace_path),
        "profiled_steps": 1,
        "cuda_kernel_launch_count": len(kernels),
        "sum_cuda_kernel_duration_us": sum(e["dur"] for e in kernels),
        "cuda_kernel_groups": grouped,
        "cuda_runtime_launch_groups": dict(runtime_launch),
        "top_self_cpu_ops": cpu[:50],
        "all_cpu_ops": cpu,
        "adapter_related_custom_ops": custom,
        "whole_step_marker": markers,
        "kernel_duration_note": "Sum of profiled kernel durations; not normal end-to-end latency, and may include overlaps.",
        "cpu_note": "Self CPU times include profiling instrumentation. Whole-step self time includes Python and work without nested profiler ranges; it is not a precise Python dispatch measurement.",
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--neighbors", type=int, default=12)
    parser.add_argument("--warmup", type=int, default=3)
    args = parser.parse_args()
    if args.warmup < 3:
        raise ValueError("At least three complete warmup steps are required")
    if os.environ.get("CUDA_LAUNCH_BLOCKING") == "1":
        raise RuntimeError("Unset CUDA_LAUNCH_BLOCKING")
    assert_exclusive_gpu()
    args.output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    baseline = build_model("prophet", args.channels, torch.float32, 2)
    adapted = adapt_model(baseline)
    pair = [baseline, adapted]
    data = build_data(args.nodes, args.neighbors, torch.float32)
    edge = data["edge_index"]
    lengths = (data["positions"][edge[0]] - data["positions"][edge[1]]).norm(dim=-1)
    report = {
        "metadata": {
            "purpose": "Profiler diagnosis only; use separate unprofiled benchmark for latency/speedup.",
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "gpu_capability": torch.cuda.get_device_capability(),
            "cpu_threads": torch.get_num_threads(),
            "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
            "initial_hardware_snapshot": hardware_snapshot(),
            "model": "prophet",
            "dtype": "float32",
            "nodes": args.nodes,
            "channels": args.channels,
            "layers": 2,
            "lmax": 2,
            "cutoffs": [2.0, 4.0],
            "edges": int(edge.shape[1]),
            "active_edges_by_cutoff": {
                str(c): int((lengths < c).sum()) for c in (2.0, 4.0)
            },
            "neighbor_construction": "Prebuilt before profiling",
            "profile_scope": "Position clone, geometry, forward; force_loss also zero_grad, force creation, loss, backward. No stress or optimizer step.",
            "profiler_options": {
                "activities": ["CPU", "CUDA"],
                "record_shapes": False,
                "profile_memory": False,
                "with_stack": False,
            },
            "warmup_steps_per_backend_per_mode": args.warmup,
            "fwd_tune": os.environ.get("FASTEQ_UNIFORM1D_FWD_TUNE"),
            "bwd_tune": os.environ.get("FASTEQ_UNIFORM1D_BWD_TUNE"),
        },
        "adaptation": adaptation_report(adapted),
        "results": [],
    }
    output_path = args.output / "profile.json"
    for mode in ("energy", "force_loss"):
        for model in pair:
            configure(model, "prophet", mode)
        print("VALIDATE", mode, flush=True)
        before = time.perf_counter()
        errors = validate_pair(pair, "prophet", data, mode, torch.float32)
        compile_seconds = time.perf_counter() - before
        for name, model in zip(("baseline", "adapted"), pair):
            print("WARMUP", mode, name, flush=True)
            before = time.perf_counter()
            for _ in range(args.warmup):
                output = execute(model, "prophet", data, mode)
                del output
            torch.cuda.synchronize()
            warmup_seconds = time.perf_counter() - before
            label = "profile_prophet_" + mode + "_" + name
            print("PROFILE", mode, name, flush=True)
            assert_exclusive_gpu()
            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=False,
                profile_memory=False,
                with_stack=False,
            ) as profile:
                with torch.profiler.record_function(label):
                    output = execute(model, "prophet", data, mode)
                    del output
                torch.cuda.synchronize()
            trace_path = args.output / (mode + "_" + name + ".trace.json")
            profile.export_chrome_trace(str(trace_path))
            assert_exclusive_gpu()
            row = {
                "mode": mode,
                "backend": name,
                "validation": errors,
                "excluded_validation_and_lazy_compile_seconds": compile_seconds,
                "excluded_warmup_seconds": warmup_seconds,
                **summarize(profile, trace_path, label),
            }
            report["results"].append(row)
            output_path.write_text(json.dumps(report, indent=2) + "\n")
            print(
                json.dumps(
                    {
                        k: row[k]
                        for k in (
                            "mode",
                            "backend",
                            "cuda_kernel_launch_count",
                            "sum_cuda_kernel_duration_us",
                        )
                    }
                ),
                flush=True,
            )
    for model in pair:
        model.zero_grad(set_to_none=True)
    report["metadata"]["final_hardware_snapshot"] = hardware_snapshot()
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print("PROFILE_DONE", output_path, flush=True)


if __name__ == "__main__":
    main()
