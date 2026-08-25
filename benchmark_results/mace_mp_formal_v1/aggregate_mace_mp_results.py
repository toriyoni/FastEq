#!/usr/bin/env python3
"""Aggregate the preserved MACE-MP force-training benchmark directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


SIZES = ("small", "medium", "large")
BACKENDS = ("e3nn", "fasteq_uniform1d")


def read_one(path: Path) -> dict[str, str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise RuntimeError(f"expected one row in {path}, found {len(rows)}")
    return rows[0]


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty CSV: {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--correctness-root", type=Path, required=True)
    args = parser.parse_args()

    all_rows: list[dict[str, object]] = []
    indexed: dict[tuple[int, str, str], dict[str, str]] = {}
    for side in (2, 4, 6):
        for size in SIZES:
            for backend in BACKENDS:
                run_dir = args.root / f"side{side}_{size}_{backend}"
                row = read_one(run_dir / "summary.csv")
                metadata = json.loads((run_dir / "metadata.json").read_text())
                indexed[(side, size, backend)] = row
                all_rows.append(
                    {
                        "model_size": size,
                        "model_sha256": metadata["model_sha256"],
                        **row,
                    }
                )

    comparisons: list[dict[str, object]] = []
    for side in (2, 4, 6):
        for size in SIZES:
            e3nn = indexed[(side, size, "e3nn")]
            fasteq = indexed[(side, size, "fasteq_uniform1d")]
            e3nn_total = float(e3nn["total_cuda_ms_median_ms"])
            fasteq_total = float(fasteq["total_cuda_ms_median_ms"])
            e3nn_mem = float(e3nn["peak_memory_gb"])
            fasteq_mem = float(fasteq["peak_memory_gb"])
            correctness = json.loads(
                (args.correctness_root / f"{size}_side2.json").read_text()
            )
            comparisons.append(
                {
                    "model_size": size,
                    "box_side": side,
                    "num_atoms": e3nn["num_atoms"],
                    "num_edges": e3nn["num_edges"],
                    "e3nn_total_median_ms": f"{e3nn_total:.6f}",
                    "fasteq_total_median_ms": f"{fasteq_total:.6f}",
                    "speedup_e3nn_over_fasteq": f"{e3nn_total / fasteq_total:.6f}",
                    "e3nn_peak_memory_gb": f"{e3nn_mem:.6f}",
                    "fasteq_peak_memory_gb": f"{fasteq_mem:.6f}",
                    "memory_reduction_pct": f"{100.0 * (1.0 - fasteq_mem / e3nn_mem):.3f}",
                    "force_max_abs_error_side2": correctness["tensor_errors"]["forces"]["max_abs"],
                    "loss_max_abs_error_side2": correctness["tensor_errors"]["loss"]["max_abs"],
                    "worst_common_gradient_max_abs_error_side2": correctness["worst_gradient_error"]["max_abs"],
                    "correctness_status_side2": correctness["status"],
                }
            )

    write_csv(args.root / "summary_all.csv", all_rows)
    write_csv(args.root / "comparison.csv", comparisons)


if __name__ == "__main__":
    main()
