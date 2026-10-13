"""Summarize measured three-way FastEq reuse comparisons without mixing runs."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    target = args.root / "performance/fasteq_reuse"
    source = target / "models_final/models.json"
    data = json.loads(source.read_text())
    rows = data["results"]
    assert len(rows) == 36, "A partial benchmark is not a final report"
    assert len({(r["model"], r["dtype"], r["nodes"], r["mode"]) for r in rows}) == 36
    flat = []
    for row in rows:
        entry = {
            key: row[key]
            for key in ("model", "dtype", "nodes", "mode", "visible_device")
        }
        for backend in ("original", "mixed", "fasteq"):
            entry[backend + "_wall_ms"] = row[backend]["wall_ms"]["median"]
        for comparison in ("original_over_fasteq", "mixed_over_fasteq"):
            for statistic in ("median", "q25", "q75"):
                entry[comparison + "_" + statistic] = row["paired_speedup"][comparison][
                    statistic
                ]
        flat.append(entry)
    with (target / "models.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flat[0]))
        writer.writeheader()
        writer.writerows(flat)
    summary = {
        "source": str(source),
        "case_count": len(rows),
        "timing": data["metadata"]["protocol"],
        "runs": data["metadata"]["runs"],
        "definition": "Within-round wall ratios, medians and IQR; CUDA event is a stream timeline, not summed kernel time",
        "N256": [r for r in flat if r["nodes"] == 256],
        "ranges": {},
    }
    for comparison in ("original_over_fasteq", "mixed_over_fasteq"):
        values = [row[comparison + "_median"] for row in flat]
        summary["ranges"][comparison] = {"min": min(values), "max": max(values)}
    (target / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharey="row")
    for i, dtype in enumerate(("float32", "float64")):
        for j, model in enumerate(("tace", "tece", "prophet")):
            ax = axes[i, j]
            chosen = [
                next(
                    r
                    for r in rows
                    if r["nodes"] == 256
                    and r["dtype"] == dtype
                    and r["model"] == model
                    and r["mode"] == mode
                )
                for mode in ("energy", "forces", "force_loss")
            ]
            x = np.arange(3)
            for k, backend in enumerate(("original", "mixed", "fasteq")):
                ys = [r[backend]["wall_ms"]["median"] for r in chosen]
                low = [y - r[backend]["wall_ms"]["q25"] for y, r in zip(ys, chosen)]
                high = [r[backend]["wall_ms"]["q75"] - y for y, r in zip(ys, chosen)]
                ax.bar(
                    x + (k - 1) * 0.25,
                    ys,
                    0.25,
                    label=backend,
                    yerr=[low, high],
                    capsize=2,
                )
            ax.set_xticks(x, ("energy", "forces", "force loss"))
            ax.set_title(model.upper() + " / " + dtype)
            if j == 0:
                ax.set_ylabel("wall latency (ms)")
    axes[0, 0].legend(fontsize=9)
    fig.suptitle(
        "H100 / C64 / N256 / prebuilt neighbors / paired eager timing\nMedian and IQR; force loss excludes optimizer update"
    )
    fig.tight_layout()
    fig.savefig(target / "model_latency.png", dpi=170)
    fig.savefig(target / "model_latency.pdf")
    print(json.dumps(summary["ranges"], indent=2))


if __name__ == "__main__":
    main()
