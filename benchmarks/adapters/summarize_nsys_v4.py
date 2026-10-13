"""Summarize warmed NVTX steps from Nsight SQLite without adding overlaps."""

import argparse
import json
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np


def interval_union(intervals):
    stop, total = None, 0
    for start, end in sorted(intervals):
        if stop is None or start >= stop:
            total += end - start
        elif end > stop:
            total += end - stop
        stop = max(end, stop) if stop is not None else end
    return total


def summarize(path):
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    names = dict(connection.execute("SELECT id,value FROM StringIds"))
    tables = {
        row[0]
        for row in connection.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
    }

    def events(table):
        return (
            [dict(row) for row in connection.execute("SELECT * FROM " + table)]
            if table in tables
            else []
        )

    kernels = events("CUPTI_ACTIVITY_KIND_KERNEL")
    copies = events("CUPTI_ACTIVITY_KIND_MEMCPY")
    memsets = events("CUPTI_ACTIVITY_KIND_MEMSET")
    apis = events("CUPTI_ACTIVITY_KIND_RUNTIME") + events("CUPTI_ACTIVITY_KIND_DRIVER")
    markers = [
        row
        for row in events("NVTX_EVENTS")
        if (row.get("text") or names.get(row.get("textId"), "")).startswith("step::")
        and row.get("end") is not None
    ]
    steps = []
    for marker in markers:
        start, end = marker["start"], marker["end"]
        text = marker.get("text") or names[marker["textId"]]
        _, model, dtype, mode, backend, iteration = text.split("::")

        def within(values, start=start, end=end):
            return [
                value
                for value in values
                if value["start"] >= start and value["end"] <= end
            ]

        ks, cs, ms, calls = (
            within(values) for values in (kernels, copies, memsets, apis)
        )
        kernel_intervals = [(value["start"], value["end"]) for value in ks]
        all_gpu_intervals = [
            (value["start"], value["end"]) for value in (*ks, *cs, *ms)
        ]
        kernel_groups, api_groups = defaultdict(list), defaultdict(list)
        for value in ks:
            kernel_groups[names[value["demangledName"]]].append(value)
        for value in calls:
            api_groups[names[value["nameId"]]].append(value)
        total_ns = end - start
        kernel_union_ns, all_union_ns = (
            interval_union(kernel_intervals),
            interval_union(all_gpu_intervals),
        )
        steps.append(
            {
                "model": model,
                "dtype": dtype,
                "mode": mode,
                "backend": backend,
                "iteration": int(iteration),
                "marker_wall_ms": total_ns / 1e6,
                "kernel_count": len(ks),
                "kernel_sum_ms": sum(value["end"] - value["start"] for value in ks)
                / 1e6,
                "kernel_union_ms": kernel_union_ns / 1e6,
                "gpu_work_union_ms": all_union_ns / 1e6,
                "gpu_work_union_fraction": all_union_ns / total_ns,
                "no_recorded_gpu_work_ms": (total_ns - all_union_ns) / 1e6,
                "copy_count": len(cs),
                "copy_bytes": sum(value["bytes"] for value in cs),
                "memset_count": len(ms),
                "kernel_groups": [
                    {
                        "name": name,
                        "count": len(values),
                        "sum_ms": sum(value["end"] - value["start"] for value in values)
                        / 1e6,
                        "registers_per_thread": sorted(
                            {value["registersPerThread"] for value in values}
                        ),
                        "local_memory_per_thread_bytes": sorted(
                            {value["localMemoryPerThread"] for value in values}
                        ),
                        "dynamic_shared_memory_bytes": sorted(
                            {value["dynamicSharedMemory"] for value in values}
                        ),
                    }
                    for name, values in kernel_groups.items()
                ],
                "api_groups": [
                    {
                        "name": name,
                        "count": len(values),
                        "sum_cpu_ms": sum(
                            value["end"] - value["start"] for value in values
                        )
                        / 1e6,
                    }
                    for name, values in api_groups.items()
                ],
            }
        )
    grouped = defaultdict(list)
    for row in steps:
        grouped[(row["model"], row["dtype"], row["mode"], row["backend"])].append(row)
    fields = (
        "marker_wall_ms",
        "kernel_count",
        "kernel_sum_ms",
        "kernel_union_ms",
        "gpu_work_union_ms",
        "gpu_work_union_fraction",
        "no_recorded_gpu_work_ms",
        "copy_count",
        "copy_bytes",
    )
    averages = []
    for key, values in grouped.items():
        row = dict(zip(("model", "dtype", "mode", "backend"), key, strict=True))
        row["recorded_steps"] = len(values)
        row.update(
            {
                field: {
                    "median": float(np.median([value[field] for value in values])),
                    "min": min(value[field] for value in values),
                    "max": max(value[field] for value in values),
                }
                for field in fields
            }
        )
        averages.append(row)
    return {
        "sqlite": str(path),
        "steps": steps,
        "summary": averages,
        "notes": [
            "NVTX spans and CUDA timelines are collected under Nsight, not normal latency.",
            "No-recorded-GPU-work is an instrumented interval complement, not Python CPU time.",
            "Busy fraction is time union of recorded kernels/copy/memset, not a hardware utilization counter.",
            "NVTX wraps execute plus one final device synchronization; no module hooks or torch.profiler.",
        ],
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("paths", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = {"reports": [summarize(path) for path in args.paths]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    for entry in report["reports"]:
        for row in entry["summary"]:
            print(
                row["model"],
                row["dtype"],
                row["mode"],
                row["backend"],
                *[
                    round(row[key]["median"], 4)
                    for key in (
                        "marker_wall_ms",
                        "kernel_sum_ms",
                        "gpu_work_union_fraction",
                    )
                ],
            )


if __name__ == "__main__":
    main()
