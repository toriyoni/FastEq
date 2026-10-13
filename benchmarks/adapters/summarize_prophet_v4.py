"""Create the Prophet v4 profile's normal timing and exclusive GPU work figures."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    raw = json.loads((directory / "profile.json").read_text())
    analysis = json.loads((directory / "analysis.json").read_text())
    table = [
        "| dtype / mode | Original ms | Mixed ms | v3 ms | v4 ms | Paired v3/v4 [Q25,Q75] | Paired original/v4 | Paired mixed/v4 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    rows = []
    for case in raw["results"]:
        times = {key: value["wall_ms"] for key, value in case["normal_timing"].items()}
        ratios = case["paired_speedup"]
        v3 = ratios["v3_over_v4"]
        table.append(
            f"| {case['dtype']} / {case['mode']} | "
            + " | ".join(
                f"{times[key]['median']:.3f}"
                for key in ("original", "mixed", "v3", "v4")
            )
            + f" | {v3['median']:.3f} [{v3['q25']:.3f},{v3['q75']:.3f}]"
            + f" | {ratios['original_over_v4']['median']:.3f}"
            + f" | {ratios['mixed_over_v4']['median']:.3f} |"
        )
        rows.append(
            {
                "dtype": case["dtype"],
                "mode": case["mode"],
                "wall_ms": times,
                "paired_speedup": ratios,
            }
        )
    (directory / "normal_timing.md").write_text("\n".join(table) + "\n")
    (directory / "summary.json").write_text(
        json.dumps(
            {
                "source_digest": raw["metadata"]["source_digest"],
                "normal_timing": rows,
                "notes": [
                    "Ratios are medians of per-round paired wall time ratios.",
                    "GPU kernel sums are instrumented work, not normal wall time or Amdahl fractions.",
                ],
            },
            indent=2,
        )
        + "\n"
    )

    modes = ("energy", "forces", "force_loss")
    colors = ("#9a9a9a", "#e69838", "#6086bc", "#327d64")
    figure, axes = plt.subplots(1, 2, figsize=(11, 3.6), constrained_layout=True)
    for dtype, axis in zip(("float32", "float64"), axes):
        for index, backend in enumerate(("original", "mixed", "v3", "v4")):
            values = [
                next(
                    case
                    for case in rows
                    if case["dtype"] == dtype and case["mode"] == mode
                )["wall_ms"][backend]
                for mode in modes
            ]
            medians = np.array([value["median"] for value in values])
            errors = np.array(
                [
                    [value["median"] - value["q25"] for value in values],
                    [value["q75"] - value["median"] for value in values],
                ]
            )
            axis.bar(
                np.arange(3) + (index - 1.5) * 0.19,
                medians,
                0.19,
                label=backend,
                color=colors[index],
                yerr=errors,
                capsize=2,
            )
        axis.set_xticks(range(3), modes)
        axis.set_title(f"Prophet C64 N256 {dtype}")
        axis.set_ylabel("Normal wall time (ms), median and IQR")
        axis.grid(axis="y", alpha=0.2)
    axes[0].legend(ncol=4, fontsize=9)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"normal_timing.{extension}", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    mechanisms = [
        "elementwise",
        "native_generated_cuda",
        "index_scatter_sort",
        "dense_gemm",
        "tensor_product_cuda",
        "fused_gate_norm_cuda",
        "norm_cuda",
        "ffn_cuda",
        "other",
    ]
    palette = plt.get_cmap("tab10").colors
    for dtype, axis in zip(("float32", "float64"), axes):
        cases = [
            next(
                case
                for case in analysis["results"]
                if case["dtype"] == dtype
                and case["mode"] == "force_loss"
                and case["backend"] == backend
            )
            for backend in ("original", "mixed", "v3", "v4")
        ]
        sums = []
        for case in cases:
            grouped = defaultdict(float)
            for row in case["exclusive_kernel_mechanisms"]:
                key = row["mechanism"] if row["mechanism"] in mechanisms else "other"
                grouped[key] += row["duration_us"] / 1000
            sums.append(grouped)
        bottom = np.zeros(4)
        for index, mechanism in enumerate(mechanisms):
            values = np.array([group[mechanism] for group in sums])
            axis.bar(
                range(4),
                values,
                bottom=bottom,
                label=mechanism.replace("_cuda", "").replace("_", " "),
                color=palette[index],
            )
            bottom += values
        axis.set_xticks(range(4), ("original", "mixed", "v3", "v4"))
        axis.set_title(f"Prophet C64 N256 {dtype} force loss")
        axis.set_ylabel("Single instrumented step: GPU kernel sum (ms)")
    axes[1].legend(loc="upper left", bbox_to_anchor=(1, 1), fontsize=8)
    for extension in ("png", "pdf"):
        figure.savefig(directory / f"kernel_work.{extension}", dpi=180)
    plt.close(figure)


if __name__ == "__main__":
    main()
