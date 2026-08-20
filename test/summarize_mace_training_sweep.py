#!/usr/bin/env python3
"""Build committed CSV and Markdown tables for the extended MACE sweep."""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "benchmark_results"
SIDES = [4, 6, 8, 10, 12, 16, 20, 24, 28, 32, 36, 40, 42, 44]


def read_summary(directory):
    path = RESULTS / directory / "summary.csv"
    if not path.exists():
        return {}
    with path.open(newline="") as handle:
        return {int(row["n_side"]): row for row in csv.DictReader(handle)}


def load_performance():
    e3nn = read_summary("final_e3nn_trainable")
    fasteq = read_summary("final_reverted_fasteq_uniform1d")
    for side in [10, 12, 16, 20]:
        e3nn.update(read_summary(f"extended_e3nn_side{side}"))
    for side in [10, 12, 16, 20, 24, 28]:
        fasteq.update(read_summary(f"extended_fasteq_side{side}"))
    return e3nn, fasteq


def load_accuracy():
    paths = {
        4: RESULTS / "final_reverted_fasteq_uniform1d_vs_e3nn_side4.json",
        **{
            side: RESULTS / f"extended_correctness_side{side}.json"
            for side in [6, 8, 10, 12, 16, 20]
        },
    }
    return {
        side: json.loads(path.read_text(encoding="utf-8"))
        for side, path in paths.items()
        if path.exists()
    }


def value(row, name):
    return None if row is None else float(row[name])


def fmt(value_, digits=3):
    if value_ is None:
        return "—"
    return f"{value_:.{digits}f}"


def sci(value_):
    if value_ is None:
        return "—"
    return f"{value_:.3e}"


def fmt_unit(value_, digits, suffix):
    if value_ is None:
        return "—"
    return f"{value_:.{digits}f}{suffix}"


def status_for(side, e_row, f_row):
    if e_row is not None and f_row is not None:
        return "双方通过"
    if side == 24:
        return "e3nn 实测 OOM；FastEq 通过"
    if side == 28:
        return "e3nn 超实测 OOM 边界；FastEq 通过"
    if side == 32:
        return "FastEq 实测 OOM；e3nn 已超边界"
    return "双方均超实测 OOM 边界，未重复运行"


def build_performance_rows(e3nn, fasteq):
    rows = []
    for side in SIDES:
        e_row = e3nn.get(side)
        f_row = fasteq.get(side)
        atoms = int(e_row["num_atoms"]) if e_row else (
            int(f_row["num_atoms"]) if f_row else 3 * side**3
        )
        edges = int(e_row["num_edges"]) if e_row else (
            int(f_row["num_edges"]) if f_row else None
        )
        e_total = value(e_row, "total_cuda_ms_median_ms")
        f_total = value(f_row, "total_cuda_ms_median_ms")
        speedup = e_total / f_total if e_total is not None and f_total else None
        time_saved = (
            100.0 * (e_total - f_total) / e_total
            if e_total is not None and f_total is not None
            else None
        )
        e_memory = value(e_row, "peak_memory_gb")
        f_memory = value(f_row, "peak_memory_gb")
        memory_saved = (
            100.0 * (e_memory - f_memory) / e_memory
            if e_memory is not None and f_memory is not None
            else None
        )
        rows.append(
            {
                "box_side": side,
                "num_atoms": atoms,
                "num_edges": edges,
                "e3nn_forward_ms": value(e_row, "forward_cuda_ms_median_ms"),
                "e3nn_backward_ms": value(e_row, "backward_cuda_ms_median_ms"),
                "e3nn_total_ms": e_total,
                "fasteq_forward_ms": value(f_row, "forward_cuda_ms_median_ms"),
                "fasteq_backward_ms": value(f_row, "backward_cuda_ms_median_ms"),
                "fasteq_total_ms": f_total,
                "e3nn_over_fasteq_speedup": speedup,
                "fasteq_time_saved_percent": time_saved,
                "e3nn_peak_memory_gib": e_memory,
                "fasteq_peak_memory_gib": f_memory,
                "fasteq_memory_saved_percent": memory_saved,
                "status": status_for(side, e_row, f_row),
            }
        )
    return rows


def build_accuracy_rows(performance_rows, accuracy):
    rows = []
    for perf in performance_rows:
        side = perf["box_side"]
        result = accuracy.get(side)
        if result is None:
            status = "无 e3nn 参考：e3nn OOM" if side in [24, 28] else "双方超显存边界"
            rows.append(
                {
                    "box_side": side,
                    "num_atoms": perf["num_atoms"],
                    "num_edges": perf["num_edges"],
                    "energy_max_abs": None,
                    "energy_max_rel": None,
                    "forces_max_abs": None,
                    "forces_max_rel": None,
                    "loss_max_abs": None,
                    "loss_max_rel": None,
                    "worst_gradient": None,
                    "gradient_max_abs": None,
                    "gradient_max_rel": None,
                    "common_parameter_tensors": None,
                    "status": status,
                }
            )
            continue
        tensor_errors = result["tensor_errors"]
        gradient_error = result["worst_gradient_error"]
        rows.append(
            {
                "box_side": side,
                "num_atoms": result["num_atoms"],
                "num_edges": result["num_edges"],
                "energy_max_abs": tensor_errors["energy"]["max_abs"],
                "energy_max_rel": tensor_errors["energy"]["max_rel_to_reference_max"],
                "forces_max_abs": tensor_errors["forces"]["max_abs"],
                "forces_max_rel": tensor_errors["forces"]["max_rel_to_reference_max"],
                "loss_max_abs": tensor_errors["loss"]["max_abs"],
                "loss_max_rel": tensor_errors["loss"]["max_rel_to_reference_max"],
                "worst_gradient": result["worst_gradient"],
                "gradient_max_abs": gradient_error["max_abs"],
                "gradient_max_rel": gradient_error["max_rel_to_reference_max"],
                "common_parameter_tensors": result["common_parameter_tensors_compared"],
                "status": result["status"],
            }
        )
    return rows


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path, performance, accuracy):
    lines = [
        "# MACE force-training extended sweep: e3nn vs FastEq Uniform1D",
        "",
        "Protocol: MACE-OFF23 Small, FP32, NVIDIA H100 80GB, 20 warmups, "
        "50 timed repeats, CUDA-event medians. The timed step is "
        "`training=True` force forward plus `mean(force^2).backward()`; graph "
        "construction, transfers, `zero_grad`, and optimizer update are excluded.",
        "",
        "## Table 1. End-to-end force-training performance",
        "",
        "| Atoms | Edges | e3nn FWD ms | e3nn BWD ms | e3nn total ms | FastEq FWD ms | FastEq BWD ms | FastEq total ms | e3nn/FastEq | FastEq time saved | e3nn GiB | FastEq GiB | Memory saved | Status |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---|",
    ]
    for row in performance:
        lines.append(
            f"| {row['num_atoms']} | {row['num_edges'] or '—'} | "
            f"{fmt(row['e3nn_forward_ms'])} | {fmt(row['e3nn_backward_ms'])} | "
            f"{fmt(row['e3nn_total_ms'])} | {fmt(row['fasteq_forward_ms'])} | "
            f"{fmt(row['fasteq_backward_ms'])} | {fmt(row['fasteq_total_ms'])} | "
            f"{fmt_unit(row['e3nn_over_fasteq_speedup'], 2, 'x')} | "
            f"{fmt_unit(row['fasteq_time_saved_percent'], 1, '%')} | "
            f"{fmt(row['e3nn_peak_memory_gib'])} | {fmt(row['fasteq_peak_memory_gib'])} | "
            f"{fmt_unit(row['fasteq_memory_saved_percent'], 1, '%')} | {row['status']} |"
        )
    lines.extend(
        [
            "",
            "## Table 2. FastEq numerical and force-loss gradient error vs e3nn",
            "",
            "| Atoms | Edges | Energy abs | Energy rel | Forces abs | Forces rel | Loss abs | Loss rel | Worst comparable gradient | Grad abs | Grad rel | Status |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|:---|---:|---:|:---|",
        ]
    )
    for row in accuracy:
        lines.append(
            f"| {row['num_atoms']} | {row['num_edges'] or '—'} | "
            f"{sci(row['energy_max_abs'])} | {sci(row['energy_max_rel'])} | "
            f"{sci(row['forces_max_abs'])} | {sci(row['forces_max_rel'])} | "
            f"{sci(row['loss_max_abs'])} | {sci(row['loss_max_rel'])} | "
            f"{row['worst_gradient'] or '—'} | {sci(row['gradient_max_abs'])} | "
            f"{sci(row['gradient_max_rel'])} | "
            f"{'通过' if row['status'] == 'pass' else row['status']} |"
        )
    lines.extend(
        [
            "",
            "Notes:",
            "",
            "- e3nn first failed at 41,472 atoms; FastEq first failed at 98,304 atoms.",
            "- Larger rows were not repeatedly executed after the monotonic OOM boundary was established.",
            "- Accuracy uses `atol=5e-5`, `rtol=5e-4`, compares 20 directly name-aligned parameter tensors, and explicitly covers all eight `conv_tp_weights` tensors. Symmetric-contraction parameters are packed differently between e3nn and cuEquivariance and are reported separately in the source JSON files.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    e3nn, fasteq = load_performance()
    performance = build_performance_rows(e3nn, fasteq)
    accuracy = build_accuracy_rows(performance, load_accuracy())
    write_csv(RESULTS / "extended_training_performance.csv", performance)
    write_csv(RESULTS / "extended_training_accuracy.csv", accuracy)
    write_markdown(RESULTS / "extended_training_tables.md", performance, accuracy)
    print(RESULTS / "extended_training_tables.md")


if __name__ == "__main__":
    main()
