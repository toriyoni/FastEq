"""Separate diagnostics for regressed primitives; never a scaling timing run."""

import argparse
import gc
import json
import os
import time
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default="GPU-af44a7bc-24c0-45be-df6e-03f4c445c78d")
    parser.add_argument("--operator", required=True)
    parser.add_argument("--atoms", type=int, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument("--mode", choices=("forward", "backward", "force_loss"), required=True)
    parser.add_argument("--cpu", type=int, default=172)
    parser.add_argument("--detach-weights", action="store_true",
                        help="Input-only diagnostic ablation, not equivalent full parameter training")
    parser.add_argument("--ncu-variant", choices=("original", "v6"),
                        help="NVTX-only counter entry, without simultaneous Torch profiler")
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_uuid
    for flag in ("FASTEQ_EXPERIMENTAL_LINEAR_SIMT", "FASTEQ_EXPERIMENTAL_NATIVE_MODULATION"):
        os.environ[flag] = "1"
    for flag in ("FASTEQ_UNIFORM1D_FWD_TUNE", "FASTEQ_UNIFORM1D_BWD_TUNE"):
        os.environ[flag] = "0"
    os.sched_setaffinity(0, {args.cpu})

    import torch
    from benchmark_scaling_case import (
        ExclusiveGuard,
        compact_snapshot,
        compare_snapshots,
        execute_operator,
        verify_manifest,
    )
    from profile_models import summarize
    from scaling_operator_factories import build_case

    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    torch.manual_seed(199)
    root = Path(os.environ["TACE_PROPHET_ROOT"])
    campaign = root / "performance/fasteq_optimization/v6/scaling_20261013"
    manifest = campaign / "manifest.json"
    guard = ExclusiveGuard(args.gpu_uuid)
    guard.check()
    source = verify_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    case = build_case(args.operator, args.atoms, args.dtype, "cuda",
                      variant=args.ncu_variant)
    if args.detach_weights:
        for label in ("baseline", "candidate"):
            for parameter in case[label + "_parameters"]:
                parameter.requires_grad_(False)
            case[label + "_parameters"] = ()
    labels = [args.ncu_variant] if args.ncu_variant else ["original", "v6"]
    report = {"purpose": "profiler diagnostics only; not normal latency/speedup",
              "operator": args.operator, "atoms": args.atoms, "dtype": args.dtype,
              "mode": args.mode, "detach_weights_ablation": args.detach_weights,
              "ablation_scope": "input-only gradients; parameter derivatives omitted" if args.detach_weights else None,
              "gpu": guard.check(), "cpu_affinity": [args.cpu], "tf32": False,
              "source_guard": source, "metadata": case["metadata"],
              "gradient_objective": "sum(mean(sin(output)) + 0.1*mean(output**2))",
              "higher_order_scope": "feature gradients, not physical atomic forces",
              "existing_normal_timing_points": [], "profiles": []}
    for path in (campaign / "points").glob("*.json"):
        if path.name.endswith(".partial.json"):
            continue
        point = json.loads(path.read_text())
        if (point.get("status") == "ok" and point.get("protocol") == "paired"
                and point.get("name") == args.operator and point.get("atoms") == args.atoms
                and point.get("dtype") == args.dtype and point.get("mode") == args.mode):
            report["existing_normal_timing_points"].append(str(path))
    snapshots = {}
    for label in labels:
        guard.check()
        output = execute_operator(case, label, args.mode, torch)
        if not args.ncu_variant:
            snapshots[label] = compact_snapshot(output, [], torch)
        del output
        torch.cuda.synchronize()
        for _ in range(3):
            output = execute_operator(case, label, args.mode, torch)
            del output
        torch.cuda.synchronize()
        guard.check()
    if not args.ncu_variant:
        report["validation"] = compare_snapshots(snapshots["original"], snapshots["v6"], args.dtype, torch)
    del snapshots

    import fasteq.ops.uniform1d_jit as uniform

    original_forward = uniform.fast_uniform1d_jit
    layouts = []

    @contextmanager
    def annotate_uniform():
        def wrapped(w, x, y, *rest, **keywords):
            layouts.append({"w_shape": list(w.shape), "w_stride": list(w.stride()),
                            "w_storage_bytes": w.untyped_storage().nbytes(),
                            "w_requires_grad": w.requires_grad,
                            "x_shape": list(x.shape), "y_shape": list(y.shape)})
            with torch.profiler.record_function("diagnostic::uniform1d"):
                return original_forward(w, x, y, *rest, **keywords)
        uniform.fast_uniform1d_jit = wrapped
        try:
            yield
        finally:
            uniform.fast_uniform1d_jit = original_forward

    if args.ncu_variant:
        guard.check()
        torch.cuda.nvtx.range_push("REGRESSION_COUNTER")
        output = execute_operator(case, args.ncu_variant, args.mode, torch)
        del output
        torch.cuda.nvtx.range_pop()
        torch.cuda.synchronize()
        report["ncu_scope"] = "one warmed execution, separate replay counters; not a timing measurement"
    else:
        primer = torch.empty(1, device="cuda")
        for label in labels:
            guard.check()
            torch.cuda.reset_peak_memory_stats()
            label_name = "diagnostic::" + args.operator + "::" + label + "::" + args.mode
            layouts.clear()
            with annotate_uniform(), torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=False, profile_memory=False, with_stack=False,
            ) as profiler:
                # Kineto can miss a first direct Driver API launch. Activate
                # CUDA collection with one common Torch kernel, then exclude
                # it from all mathematical-kernel statistics below.
                with torch.profiler.record_function("diagnostic::primer"):
                    primer.fill_(1)
                torch.cuda.synchronize()
                with torch.profiler.record_function(label_name):
                    output = execute_operator(case, label, args.mode, torch)
                    del output
                torch.cuda.synchronize()
            guard.check()
            trace = args.output / (label + ".trace.json")
            profiler.export_chrome_trace(str(trace))
            summary = summarize(profiler, trace, label_name)
            events = json.loads(trace.read_text())["traceEvents"]
            marker = next(event for event in events if event.get("name") == label_name
                          and event.get("cat") == "user_annotation" and event.get("ph") == "X")
            kernels = [event for event in events if event.get("cat") == "kernel"
                       and event.get("ph") == "X" and event["ts"] >= marker["ts"]]
            all_kernels = [event for event in events if event.get("cat") == "kernel"
                           and event.get("ph") == "X"]
            groups = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
            for event in kernels:
                groups[event["name"]]["count"] += 1
                groups[event["name"]]["duration_us"] += event["dur"]
            summary["cuda_kernel_launch_count"] = len(kernels)
            summary["sum_cuda_kernel_duration_us"] = sum(event["dur"] for event in kernels)
            summary["cuda_kernel_groups"] = sorted(
                [{"name": name, **values} for name, values in groups.items()],
                key=lambda row: row["duration_us"], reverse=True)
            summary["excluded_primer_kernels"] = len(all_kernels) - len(kernels)
            summary["cpu_note"] += " CPU aggregates include a tiny common primer; mathematical GPU statistics exclude it."
            report["profiles"].append({"label": label, **summary,
                                        "uniform_operand_layouts": list(layouts),
                                        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                                        "peak_reserved_bytes": torch.cuda.max_memory_reserved()})
            print(json.dumps({"operator": args.operator, "mode": args.mode, "label": label,
                              "kernels": summary["cuda_kernel_launch_count"],
                              "kernel_sum_us": summary["sum_cuda_kernel_duration_us"]}), flush=True)
            del profiler
            gc.collect()
    report["gpu_after"] = guard.check()
    report["source_guard_end"] = verify_manifest(manifest)
    if report["source_guard_end"] != source:
        raise RuntimeError("Source changed during diagnostics")
    report["status"] = "completed"
    report["completed_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    (args.output / "profile.json").write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
