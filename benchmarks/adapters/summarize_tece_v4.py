"""CPU-only tables from the separately collected TECE v4 traces and timings."""

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path


def coarse(owner):
    return owner.split("::")[1].split("/")[0] if owner.startswith("family::") else "unclassified"


def native_role(row, descriptor, nodes, dtype):
    """Exact grid/metadata matching for the two profiled large shared maps."""
    if row["name"] != "run":
        return None
    paths = descriptor["paths"]
    if not paths:
        return None
    grid = row["args"]["grid"]
    input_groups = {(p[0], p[3], p[5]) for p in paths}
    output_groups = {(p[2], p[4], p[5]) for p in paths}
    weights = {(p[1], p[3], p[4]) for p in paths}
    dx_tiles = sum((ci + 31) // 32 * ((m + 3) // 4) for _, ci, m in input_groups)
    dy_tiles = sum((co + 31) // 32 * ((m + 3) // 4) for _, co, m in output_groups)
    weight_tiles = sum((ci * co + 127) // 128 for _, ci, co in weights)
    if grid[:2] == [nodes, dx_tiles] and row["args"].get("shared memory"):
        return "ROLE0 dX scalar shared transpose"
    if grid[0] < nodes and grid[1] == weight_tiles:
        return "ROLE1 dW scalar chunk partials"
    if grid[:2] == [weight_tiles, 1]:
        return "ROLE3 dW chunk reducer"
    mma_tiles = sum((co + 7) // 8 for _, co, _ in output_groups)
    if dtype == "float64" and grid[:2] == [(nodes * max(p[5] for p in paths) + 63) // 64, mma_tiles]:
        return "ROLE5 forward/dY FP64 MMA"
    if grid[:2] == [nodes, dy_tiles]:
        return "ROLE2 forward/dY scalar"
    return "unresolved native role"


def gemm_inputs(trace, rows, paths, nodes):
    events = json.loads(Path(trace).read_text())["traceEvents"]
    launch = {e.get("args", {}).get("correlation"): e for e in events
              if e.get("ph") == "X" and "LaunchKernel" in e.get("name", "")}
    cpu = defaultdict(list)
    for event in events:
        if event.get("ph") == "X" and event.get("cat") == "cpu_op":
            cpu[event["tid"]].append(event)
    result = []
    for row in rows:
        if "gemm" not in row["name"].lower():
            continue
        runtime = launch[row["args"]["correlation"]]
        matches = [e for e in cpu[runtime["tid"]] if e["ts"] <= runtime["ts"]
                   and e["ts"] + e["dur"] >= runtime["ts"] + runtime["dur"]]
        operation = min(matches, key=lambda e: e["dur"]) if matches else {}
        shapes = operation.get("args", {}).get("Input Dims", [])
        role = "unresolved"
        for _, _, _, ci, co, dim, _ in paths:
            matrix_rows = nodes * dim
            if shapes[:2] == [[matrix_rows, ci], [ci, co]]:
                role = "forward/dY"
            elif shapes[:2] == [[matrix_rows, co], [co, ci]]:
                role = "dX"
            elif shapes[:2] == [[ci, matrix_rows], [matrix_rows, co]]:
                role = "dW"
        result.append({**row, "aten_operator": operation.get("name"), "aten_input_shapes": shapes, "role_from_mm_shapes": role})
    return result


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0])
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()
    report = json.loads(args.input.read_text())
    directory = args.input.parent
    normal, profiles, families, phases, linears, mm_rows = [], [], [], [], [], []
    hot_descriptor = {}
    for case in report["results"]:
        dtype, mode = case["dtype"], case["mode"]
        for backend, timing in case["normal_timing"].items():
            normal.append({"dtype": dtype, "mode": mode, "backend": backend,
                           "wall_median_ms": timing["wall_ms"]["median"], "wall_q25_ms": timing["wall_ms"]["q25"],
                           "wall_q75_ms": timing["wall_ms"]["q75"], "event_median_ms": timing["cuda_event_ms"]["median"]})
        for profile in case["profiles"]:
            backend = profile["backend"]
            summary = {"dtype": dtype, "mode": mode, "backend": backend,
                       "kernel_count": len(profile["kernel_rows"]),
                       "kernel_sum_ms": profile["sum_cuda_kernel_duration_us"] / 1000,
                       "kernel_union_busy_ms": profile["kernel_union_busy_us"] / 1000,
                       "profile_timeline_ms": profile["first_to_last_kernel_timeline_us"] / 1000,
                       "profile_timeline_gap_ms": profile["timeline_gap_us"] / 1000,
                       "direct_annotation_duration_fraction": profile["direct_annotation_kernel_duration_us"] / profile["sum_cuda_kernel_duration_us"],
                       "direct_annotation_kernel_count": profile["direct_annotation_kernel_count"],
                       "unclassified_kernel_count": profile["unclassified_kernel_count"],
                       "unprofiled_peak_increment_mib": profile["unprofiled_peak_increment_bytes"] / 2**20,
                       "profile_peak_increment_mib": profile["profile_peak_increment_bytes"] / 2**20,
                       "copy_count": len(profile["copy_rows"]),
                       "copy_total_ms": sum(x["duration_us"] for x in profile["copy_rows"]) / 1000,
                       "copy_bytes": sum(x["args"].get("bytes", 0) for x in profile["copy_rows"]),
                       "runtime_launch_cpu_ms": sum(x["duration_us"] for x in profile["cuda_runtime_launch_groups"].values()) / 1000,
                       "profile_step_cpu_ms": profile["whole_step_marker"][0]["cpu_total_us"] / 1000,
                       "profile_step_self_cpu_ms": profile["whole_step_marker"][0]["self_cpu_us"] / 1000}
            profiles.append(summary)
            for row in profile["exclusive_gpu_families"]:
                if row["category"] == "kernel":
                    families.append({"dtype": dtype, "mode": mode, "backend": backend,
                                     "family": row["family"], "kernel_count": row["count"],
                                     "kernel_ms": row["duration_us"] / 1000})
            grouped = defaultdict(lambda: [0, 0.0])
            for row in profile["kernel_rows"]:
                owner = row["owner"].split("::")
                phase = owner[3] if len(owner) > 3 and owner[2] != "aten_fallback" else "unclassified"
                grouped[(coarse(row["owner"]), phase)][0] += 1
                grouped[(coarse(row["owner"]), phase)][1] += row["duration_us"]
            for (fam, phase), (count, duration) in grouped.items():
                phases.append({"dtype": dtype, "mode": mode, "backend": backend, "family": fam,
                               "phase": phase, "kernel_count": count, "kernel_ms": duration / 1000})
            if mode != "force_loss" or backend != "v4":
                continue
            for module in ("linear_up", "linear_coefs"):
                path = "readout_fn.representation.interactions.1.rejector." + module
                descriptor = profile["module_descriptors"]["linear_descriptor::v4/" + path]
                hot_descriptor[path] = descriptor
                rows = [r for r in profile["kernel_rows"] if "v4/" + path + "::" in r["owner"]]
                roles = defaultdict(lambda: {"count": 0, "duration_us": 0.0, "grid": None, "block": None, "shared_bytes": None})
                for row in rows:
                    role = native_role(row, descriptor, report["metadata"]["edges"], dtype)
                    if role is None:
                        continue
                    key = (role, row["owner"].split("::")[3])
                    result = roles[key]
                    result["count"] += 1
                    result["duration_us"] += row["duration_us"]
                    result["grid"], result["block"] = row["args"]["grid"], row["args"]["block"]
                    result["shared_bytes"] = row["args"]["shared memory"]
                for (role, phase), result in roles.items():
                    linears.append({"dtype": dtype, "module": path, "role": role, "phase": phase,
                                    "kernel_count": result["count"], "kernel_ms": result["duration_us"] / 1000,
                                    "grid": result["grid"], "block": result["block"], "shared_bytes": result["shared_bytes"]})
                original = next(p for p in case["profiles"] if p["backend"] == "original")
                original_rows = [r for r in original["kernel_rows"] if "original/" + path + "::" in r["owner"]]
                for row in gemm_inputs(original["trace"], original_rows, descriptor["paths"], report["metadata"]["edges"]):
                    mm_rows.append({"dtype": dtype, "module": path, **row})
    summary = {"profile_input": str(args.input), "profile_input_sha256": hashlib.sha256(args.input.read_bytes()).hexdigest(),
               "metadata": report["metadata"], "normal_timing": normal, "profiles": profiles, "families": families,
               "phase_kernel_totals": phases, "large_linear_native_roles": linears,
               "large_linear_descriptors": hot_descriptor, "large_linear_original_gemms": mm_rows,
               "paired_speedup": [{"dtype": c["dtype"], "mode": c["mode"], **c["paired_speedup"]} for c in report["results"]],
               "validation": [{"dtype": c["dtype"], "mode": c["mode"], **c["validation"]} for c in report["results"]]}
    (directory / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    for name, rows in (("normal_timing", normal), ("profile_metrics", profiles), ("kernel_families", families),
                       ("kernel_phases", phases), ("large_linear_roles", linears)):
        write_csv(directory / (name + ".csv"), rows)
    details = []
    for case in report["results"]:
        for profile in case["profiles"]:
            grouped = defaultdict(lambda: {"kernel_count": 0, "kernel_us": 0.0})
            for row in profile["kernel_rows"]:
                owner = row["owner"].split("::")
                if len(owner) >= 4 and owner[1] == "eq_linear":
                    value = grouped[(owner[2], owner[3])]
                    value["kernel_count"] += 1
                    value["kernel_us"] += row["duration_us"]
            allocations = defaultdict(int)
            for row in profile["allocator_events"]:
                if row["bytes"] > 0:
                    allocations[coarse(row["owner"])] += row["bytes"]
            details.append({"dtype": case["dtype"], "mode": case["mode"], "backend": profile["backend"],
                            "all_linear_module_phase": [{"module": key[0], "phase": key[1], **value} for key, value in grouped.items()],
                            "cpu_annotation_calls": profile["cpu_annotation_calls"],
                            "exclusive_annotated_cpu_intervals_us": profile["exclusive_annotated_cpu_intervals_us"],
                            "cumulative_positive_allocator_bytes_by_family": dict(allocations)})
    (directory / "details.json").write_text(json.dumps(details, indent=2) + "\n")
    print("SUMMARY_DONE", len(normal), len(profiles), len(mm_rows))


if __name__ == "__main__":
    main()
