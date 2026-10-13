"""Compact frozen Prophet evidence without mixing annotation spans and kernels."""

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    path = args.directory / "profile.json"
    report = json.loads(path.read_text())
    assert len(report["results"]) == 6
    summary = {
        "notes": [
            "Normal latency is unprofiled paired wall time.",
            "GPU costs are sums of raw trace kernel durations, separate from normal latency.",
            "CPU ranges are instrumented and nested; do not add FFN/Gate/Norm ranges.",
            "CUDA user annotation spans include queue gaps and are excluded from CPU ranges.",
        ],
        "results": [],
    }
    for case in report["results"]:
        row = {
            "dtype": case["dtype"],
            "mode": case["mode"],
            "normal_wall_ms": {
                label: case["normal_timing"][label]["wall_ms"]
                for label in ("mixed", "v3")
            },
            "normal_cuda_timeline_ms": {
                label: case["normal_timing"][label]["cuda_event_ms"]
                for label in ("mixed", "v3")
            },
            "paired_mixed_over_v3": case["paired_mixed_over_v3"],
            "validation": case["validation"],
            "profiles": {},
        }
        for profile in case["profiles"]:
            families = defaultdict(lambda: {"count": 0, "duration_us": 0.0})
            for kernel in profile["cuda_kernel_groups"]:
                name = kernel["name"]
                group = (
                    "norm_parameter_reduce_rows"
                    if name == "_tree_reduce_rows"
                    else "norm_double"
                    if name == "_double_kernel"
                    else "norm_first"
                    if name == "_backward_kernel"
                    else "norm_forward"
                    if name == "_forward_kernel"
                    else "gate_double"
                    if name.startswith("gate_double")
                    else "gate_first"
                    if name == "gate_backward_kernel"
                    else "gate_forward"
                    if name == "gate_forward_kernel"
                    else "other"
                )
                families[group]["count"] += kernel["count"]
                families[group]["duration_us"] += kernel["duration_us"]
            profile["kernel_families"] = dict(families)
            scopes = [
                x
                for x in profile["shape_groups"]
                if x["name"].startswith("detail::") and x["cpu_total_us"] > 0
            ]
            decomposition = {
                x["name"]: {
                    k: x[k]
                    for k in ("count", "self_cpu_us", "cpu_total_us", "device_total_us")
                }
                for x in scopes
            }
            gate_norm = [x for x in scopes if "ResidualGateNorm::" in x["name"]]
            p = {
                "cuda_kernel_launch_count": profile["cuda_kernel_launch_count"],
                "sum_cuda_kernel_duration_us": profile["sum_cuda_kernel_duration_us"],
                "kernel_families": dict(families),
                "cpu_ranges": decomposition,
                "residual_gate_norm_forward_cpu_total_us": sum(
                    x["cpu_total_us"] for x in gate_norm
                ),
                "residual_gate_norm_forward_calls": sum(x["count"] for x in gate_norm),
                "shape_specific_autograd": [
                    x
                    for x in profile["shape_groups"]
                    if x["cpu_total_us"] > 0
                    and x["name"]
                    in (
                        "FastGateFunction",
                        "FastGateBackwardFunction",
                        "FastGateBackwardFunctionBackward",
                        "_Norm",
                        "_NormBackward",
                        "_NormBackwardBackward",
                        "_Norms",
                        "_NormsBackward",
                    )
                ],
            }
            row["profiles"][profile["backend"]] = p
        summary["results"].append(row)
    archive = Path(report["metadata"]["v3"]["archive"])
    mirror = Path(report["metadata"]["v3"]["loaded_package"])
    matched = {}
    for entry in report["metadata"]["v3"]["manifest"]["files"]:
        relative = Path(entry["path"])
        if relative.parts[:2] != ("src", "FastEq"):
            continue
        source = mirror / Path(*relative.parts[2:])
        if relative.as_posix() == "src/FastEq/fasteq/cuda/jit.py":
            data = (
                (archive / relative)
                .read_text()
                .replace('name="fasteq_cuda_jit"', 'name="fasteq_cuda_jit_v3_profile"')
                .encode()
            )
        else:
            data = (archive / relative).read_bytes()
        assert source.read_bytes() == data, relative
        matched[str(relative)] = hashlib.sha256(source.read_bytes()).hexdigest()
    report["metadata"]["v3"]["post_run_frozen_package_sha256"] = matched
    report["metadata"]["profile_summary_note"] = summary["notes"]
    path.write_text(json.dumps(report, indent=2) + "\n")
    (args.directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for row in summary["results"]:
        print(
            row["dtype"],
            row["mode"],
            {
                label: row["normal_wall_ms"][label]["median"]
                for label in ("mixed", "v3")
            },
            row["paired_mixed_over_v3"]["median"],
        )


if __name__ == "__main__":
    main()
