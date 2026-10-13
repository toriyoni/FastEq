"""Targeted, instrumented profiles of frozen generic TP regressions.

This is not the normal timing benchmark. CUDA kernel intervals and CPU profiler
intervals are reported separately; diagnostic parameter freezing is an ablation
and does not represent a valid replacement for trainable weight gradients.
"""

import argparse
import gc
import hashlib
import json
import os
import statistics
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path

os.environ["FASTEQ_EXPERIMENTAL_LINEAR_SIMT"] = "1"
os.environ["FASTEQ_EXPERIMENTAL_NATIVE_MODULATION"] = "1"
os.environ["FASTEQ_UNIFORM1D_FWD_TUNE"] = "0"
os.environ["FASTEQ_UNIFORM1D_BWD_TUNE"] = "0"

import torch
from benchmark_scaling_case import ExclusiveGuard, execute_operator, verify_manifest
from scaling_operator_factories import build_case

ROOT = Path("/public-data/zhouxibo/zxb/tace_prophet_fasteq")
DEFAULT_CASES = (
    ("tp_uvw", "float32", 256), ("tp_uvw", "float32", 16384),
    ("tp_uvw", "float64", 4096), ("tp_uvw", "float64", 32768),
    ("tp_nonuniform", "float32", 256), ("tp_nonuniform", "float32", 65536),
    ("tp_nonuniform", "float64", 4096), ("tp_nonuniform", "float64", 32768),
)


@contextmanager
def annotations(mode):
    """Process-local profiling labels, restored after each instrumented sample."""
    from fasteq.ops import segmented_product as implementation

    original_execute, original_grad = implementation._execute, torch.autograd.grad
    records = []

    def traced_execute(metadata, role, required, rows, values):
        dimensions, terms, _ = implementation._parse(metadata)
        count = rows if role == 3 else values[role].shape[0]
        records.append({"role": role, "required_mask": required, "rows": rows,
                        "operand_shapes": [list(value.shape) for value in values],
                        "grid_x": (count * dimensions[role] + 127) // 128,
                        "block_x": 128, "output_rows": count,
                        "descriptor_term_count": len(terms),
                        "shared_serial_row_reduction": role != 3 and count == 1,
                        "cuda_source_sha256": hashlib.sha256(
                            implementation._source(values[1].dtype, role, required).encode()).hexdigest()})
        with torch.profiler.record_function(f"segmented::role{role}::required{required}"):
            return original_execute(metadata, role, required, rows, values)

    def traced_grad(*args, **kwargs):
        phase = "first" if kwargs.get("create_graph") or mode == "backward" else "higher"
        with torch.profiler.record_function("derivative_stage::" + phase):
            return original_grad(*args, **kwargs)

    implementation._execute, torch.autograd.grad = traced_execute, traced_grad
    try:
        yield records
    finally:
        implementation._execute, torch.autograd.grad = original_execute, original_grad


def summarize(profiler):
    kernels = defaultdict(lambda: {"count": 0, "device_us": 0.0})
    averages = []
    for event in profiler.key_averages():
        averages.append({"name": event.key, "count": event.count,
                         "device_type": str(event.device_type),
                         "is_user_annotation": event.is_user_annotation,
                         "self_cpu_us": event.self_cpu_time_total,
                         "inclusive_cpu_us": event.cpu_time_total,
                         "self_device_us": event.self_device_time_total,
                         "inclusive_device_us": event.device_time_total})
    excluded_primer_count = 0
    for row in averages:
        if row["device_type"].endswith("CUDA") and not row["is_user_annotation"]:
            if "spin_kernel" in row["name"] or "sleep" in row["name"].lower():
                excluded_primer_count += row["count"]
                continue
            kernels[row["name"]]["count"] += row["count"]
            kernels[row["name"]]["device_us"] += row["self_device_us"]
    device_total = sum(row["device_us"] for row in kernels.values())
    return {
        "kernel_count": sum(row["count"] for row in kernels.values()),
        "excluded_primer_kernel_count": excluded_primer_count,
        "summed_kernel_device_us": device_total,
        "kernels": sorted(({"name": name, **row} for name, row in kernels.items()),
                          key=lambda row: row["device_us"], reverse=True)[:25],
        "top_self_cpu": sorted(averages, key=lambda row: row["self_cpu_us"], reverse=True)[:20],
        "top_device_ops": sorted(averages, key=lambda row: row["inclusive_device_us"], reverse=True)[:20],
        "annotated_roles": [row for row in averages if row["name"].startswith("segmented::")],
        "derivative_stages": [row for row in averages if row["name"].startswith("derivative_stage::")],
    }


def existing_normal_point(name, dtype, atoms, mode):
    candidates = list((ROOT / "performance/fasteq_optimization/v6/scaling_20261013/points").glob(
        f"operator__{name}__{dtype}__{mode}__N{atoms}__pair__*.json"))
    records = []
    for path in candidates:
        result = json.loads(path.read_text())
        if result.get("status") == "ok":
            measurements = result["measurements"]
            records.append({"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                            "wall_ms_medians": {label: statistics.median(value["wall_ms"])
                                                for label, value in measurements.items()},
                            "event_ms_medians": {label: statistics.median(value["event_ms"])
                                                 for label, value in measurements.items()}})
    return records[-1] if records else None


def save(path, report):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--gpu-uuid", default="GPU-13222c8f-3c05-b93a-e9d6-1192c18a00ce")
    parser.add_argument("--cpu-core", type=int, default=168)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--probe-primer", action="store_true")
    parser.add_argument("--large-uvw", action="store_true")
    parser.add_argument("--ncu-role", type=int, choices=(0, 3))
    parser.add_argument("--ablation", action="store_true")
    parser.add_argument("--case", nargs=3, metavar=("NAME", "DTYPE", "ATOMS"))
    parser.add_argument("--executions", type=int, default=1)
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != args.gpu_uuid:
        raise RuntimeError("Set CUDA_VISIBLE_DEVICES to the exact assigned UUID")
    os.sched_setaffinity(0, {args.cpu_core})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    guard = ExclusiveGuard(args.gpu_uuid)
    manifest = ROOT / "performance/fasteq_optimization/v6/scaling_20261013/manifest.json"
    source_guard = verify_manifest(manifest)
    args.output.mkdir(parents=True, exist_ok=True)
    output = args.output / "segmented_profile.json"
    report = {"status": "running", "measurement_kind": "instrumented Torch CPU/CUDA profiler",
              "normal_timing_remeasured": False, "hardware_counters": False,
              "mode_scope": "Operator feature-gradient microchain, not physical coordinate force loss",
              "gpu_uuid": args.gpu_uuid, "gpu_start": guard.check(), "source_guard": source_guard,
              "profile_driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "torch": torch.__version__, "tf32": False, "cpu_affinity": [args.cpu_core],
              "records": [], "errors": []}
    if args.ncu_role is not None:
        from fasteq.ops import segmented_product as implementation
        torch.manual_seed(199)
        case = build_case("tp_uvw", 4096, "float32", "cuda", variant="v6")
        for _ in range(2):
            result = execute_operator(case, "v6", "backward", torch)
            del result
        torch.cuda.synchronize()
        guard.check()
        original = implementation._execute
        captures = []

        def selected_role(metadata, role, required, rows, values):
            if role == args.ncu_role and not captures:
                captures.append({"role": role, "required_mask": required, "rows": rows,
                                 "shapes": [list(value.shape) for value in values]})
                torch.cuda.nvtx.range_push("ROLE_RANGE")
                try:
                    return original(metadata, role, required, rows, values)
                finally:
                    torch.cuda.nvtx.range_pop()
            return original(metadata, role, required, rows, values)

        implementation._execute = selected_role
        try:
            result = execute_operator(case, "v6", "backward", torch)
            del result
            torch.cuda.synchronize()
        finally:
            implementation._execute = original
        report.update({"status": "complete", "measurement_kind": "NCU counter collection, cold replay",
                       "hardware_counters": True, "capture": captures, "gpu_end": guard.check(),
                       "source_guard_end": verify_manifest(manifest)})
        save(args.output / f"ncu_role{args.ncu_role}_provenance.json", report)
        return
    cases = DEFAULT_CASES[:1] if args.smoke else DEFAULT_CASES
    if args.case:
        cases = ((args.case[0], args.case[1], int(args.case[2])),)
    if args.probe_primer:
        cases = (("tp_uvw", "float64", 4096),)
    if args.large_uvw:
        cases = (("tp_uvw", "float32", 65536),)
    if args.ablation:
        cases = (("tp_uvw", "float32", 4096),)
    for name, dtype, atoms in cases:
        torch.manual_seed(199)
        guard.check()
        case = build_case(name, atoms, dtype, "cuda")
        modes = ("force_loss",) if args.large_uvw or args.ablation else ("forward", "backward", "force_loss")
        for mode in modes:
            experiments = (("original", False), ("v6", False), ("original", True), ("v6", True)) if args.ablation else (
                ("original", False), ("v6", False))
            for label, freeze_weights in experiments:
                selected = {"original": "baseline", "v6": "candidate"}[label]
                if freeze_weights:
                    for parameter in case[selected + "_parameters"]:
                        parameter.requires_grad_(False)
                    case[selected + "_parameters"] = ()
                for _ in range(2):
                    guard.check()
                    result = execute_operator(case, label, mode, torch)
                    del result
                torch.cuda.synchronize()
                guard.check()
                with annotations(mode) as calls, torch.profiler.profile(
                        activities=[torch.profiler.ProfilerActivity.CPU,
                                    torch.profiler.ProfilerActivity.CUDA],
                        record_shapes=False, profile_memory=False) as profiler:
                    with torch.profiler.record_function("profiler_primer"):
                        torch.cuda._sleep(1000)
                    for _ in range(args.executions):
                        with torch.profiler.record_function("micro::" + mode):
                            result = execute_operator(case, label, mode, torch)
                            del result
                    torch.cuda.synchronize()
                gpu_after = guard.check()
                summary = summarize(profiler)
                row = {"operator": name, "dtype": dtype, "atoms": atoms, "rows": case["metadata"]["rows"],
                       "mode": mode, "backend": label, "executions": args.executions,
                       "diagnostic_weights_frozen": freeze_weights,
                       "annotation_calls": calls, "gpu_after": gpu_after,
                       "normal_timing_evidence": existing_normal_point(name, dtype, atoms, mode),
                       **summary}
                if label == "v6":
                    traced_kernel_count = sum(event["count"] for event in summary["kernels"]
                                              if event["name"] == "run")
                    row["capture_complete"] = traced_kernel_count == len(calls)
                    row["expected_segmented_kernels"] = len(calls)
                    row["observed_segmented_kernels"] = traced_kernel_count
                else:
                    row["capture_complete"] = summary["kernel_count"] > 0
                if atoms == 256 and name == "tp_uvw" and mode == "force_loss":
                    path = args.output / ("uvw_N256_" + label + "_force_loss_trace.json")
                    profiler.export_chrome_trace(str(path))
                    row["trace_path"] = str(path)
                report["records"].append(row)
                save(output, report)
                print("PROFILE", name, dtype, atoms, mode, label,
                      "device_us", round(summary["summed_kernel_device_us"], 2),
                      "kernels", summary["kernel_count"], flush=True)
                del profiler
                gc.collect()
        del case
        gc.collect()
        torch.cuda.empty_cache()
    report["source_guard_end"] = verify_manifest(manifest)
    if report["source_guard_end"] != source_guard:
        raise RuntimeError("Frozen source drift")
    report["gpu_end"] = guard.check()
    report["status"] = "complete"
    save(output, report)


if __name__ == "__main__":
    main()
