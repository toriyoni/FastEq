"""Summarize measured latency; profiler kernel sums stay separate."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(__file__).resolve().parents[4]
    )
    args = parser.parse_args()
    root = args.root
    out = root / "performance"
    models = json.loads((out / "models_final/models.json").read_text())
    operators = json.loads((out / "operators/results.json").read_text())
    profiles = json.loads((out / "profile_prophet/profile.json").read_text())
    assert len(models["results"]) == 36, len(models["results"])
    assert len(operators["results"]) == 96 and not operators["errors"]
    report = {
        "date": "2026-10-07",
        "excluded_runs": [
            "models_interrupted_gpu0: another compute task arrived; GPU sharing/OOM; entire run excluded",
            "models_prophet/models_tace_tece: concurrent initial compilation pilot, excluded",
            "operators_pilot_tp: pilot replaced with quiet warmed TP measurements",
        ],
        "model_protocol": models["metadata"],
        "operator_protocol": operators["metadata"],
        "models": models["results"],
        "operators": operators["results"],
        "profiler_note": "Separate instrumented run. Kernel duration sums are not eager latency or speedup.",
        "profile_summaries": [
            {
                k: r[k]
                for k in (
                    "mode",
                    "backend",
                    "cuda_kernel_launch_count",
                    "sum_cuda_kernel_duration_us",
                )
            }
            for r in profiles["results"]
        ],
    }
    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    index = {
        (r["model"], r["dtype"], r["nodes"], r["mode"]): r for r in models["results"]
    }
    labels = [
        (m, mode)
        for m in ("tace", "tece", "prophet")
        for mode in ("energy", "forces", "force_loss")
    ]
    fig, axes = plt.subplots(
        2, 2, figsize=(13, 9), sharex=True, constrained_layout=True
    )
    [r["paired_wall_speedup"]["q25"] for r in models["results"]]
    highs = [r["paired_wall_speedup"]["q75"] for r in models["results"]]
    xlim = (0.0, max(highs) + 0.2)
    for i, dtype in enumerate(("float32", "float64")):
        for j, nodes in enumerate((64, 256)):
            ax = axes[i, j]
            rows = [index[m, dtype, nodes, mode] for m, mode in labels]
            med = np.array([r["paired_wall_speedup"]["median"] for r in rows])
            lo = np.array([r["paired_wall_speedup"]["q25"] for r in rows])
            hi = np.array([r["paired_wall_speedup"]["q75"] for r in rows])
            y = np.arange(len(rows))
            colors = ["#27866c" if x >= 1 else "#bb554d" for x in med]
            ax.barh(
                y,
                med,
                color=colors,
                alpha=0.85,
                xerr=np.stack((med - lo, hi - med)),
                capsize=3,
            )
            ax.axvline(1, color="black", linestyle="--", linewidth=1)
            ax.set_yticks(
                y,
                [m.upper() + " / " + mode.replace("_", " ") for m, mode in labels],
                fontsize=9,
            )
            ax.invert_yaxis()
            ax.set_xlim(*xlim)
            ax.grid(axis="x", alpha=0.2)
            ax.set_title(f"{dtype.upper()} / {nodes} atoms / C64")
            ax.set_xlabel("Paired eager wall-time speedup (baseline / adapted)")
            for yy, x in zip(y, med):
                ax.text(x + 0.02, yy, f"{x:.2f}x", va="center", fontsize=9)
    fig.suptitle(
        "Equivariant CUDA adaptation: end-to-end H100 latency\n"
        "2 layers, lmax=2, 12 prebuilt neighbors/atom; error bars = paired IQR",
        fontsize=13,
    )
    fig.savefig(out / "model_speedup.png", dpi=180)
    fig.savefig(out / "model_speedup.pdf")
    print("LARGE GRAPH MODEL RESULTS (N256, baseline -> adapted milliseconds):")
    for dtype in ("float32", "float64"):
        for m in ("tace", "tece", "prophet"):
            print(m, dtype)
            for mode in ("energy", "forces", "force_loss"):
                r = index[m, dtype, 256, mode]
                b, a = (
                    r["baseline"]["wall_ms"]["median"],
                    r["adapted"]["wall_ms"]["median"],
                )
                print(f"  {mode:10} {b:.3f} -> {a:.3f} ms; {r['wall_speedup']:.3f}x")
    # Update the existing technical environment README with current evidence.
    text = """\n## 2026-10-07 性能实测\n\n完整数据见 `performance/summary.json`；整模型 CSV 在 `performance/models_final/models.csv`，独立算子 CSV 在 `performance/operators/results.csv`，图在 `performance/model_speedup.png`。计时使用 H100、CUDA FP32/FP64、64 通道、lmax=2、两层、64/256 原子、每原子预建 12 条最近邻边。TACE/TECE 的 ACE correlation=2，TECE 第二层为 so2，保留 RRA/ECE；Prophet 两个 cutoff 为 2/4。随机权重固定，原模型与适配模型逐项验证后测量。\n\n编译与预热已排除，所有缓存准备好后，单进程串行测整模型，每批前后检查所选 GPU 是否有其他计算任务，检查放在计时区间外。初次 GPU0 上的正式测量中途被外来任务占用并 OOM，整轮数据归档为 `models_interrupted_gpu0` 且完全排除；最终36项来自换卡后的完整测量。每项 7 组配对样本、每组 5 次迭代，交替执行 baseline/adapted。下表取各后端墙钟中位数，速比为两者之比；原始样本及配对速比 IQR 在 JSON 中。CUDA event 同时记录设备时间轴，含 eager 派发造成的空闲间隙，并非内核耗时总和。计时包含位置复制、几何与图处理；预建邻居列表的成本不在其中。force_loss 包含 zero_grad、能量/力、loss 与 backward，不含 optimizer 更新和应力。\n\n### 256 原子端到端墙钟中位数\n\n| 模型 | 精度 | 模式 | 原实现 ms | 适配 ms | 加速比 |\n| --- | --- | --- | ---: | ---: | ---: |\n"""
    for dtype in ("float32", "float64"):
        for m in ("tace", "tece", "prophet"):
            for mode in ("energy", "forces", "force_loss"):
                r = index[m, dtype, 256, mode]
                text += f"| {m} | {dtype} | {mode} | {r['baseline']['wall_ms']['median']:.3f} | {r['adapted']['wall_ms']['median']:.3f} | {r['wall_speedup']:.3f}x |\n"
    text += """\n速比大于 1 表示加速，小于 1 表示变慢。结果仅适用于这里的形状、dtype 与 eager 模式，不推断预训练训练收敛或 MD 全流程速度。基线为原模型对应路径，TECE 的 Wigner-D 在原模型中已有 EQX CUDA；其收益不能计为此次新增适配。\n\n独立算子网格共 96 项：8 类算子 × 2 精度 × 2 行数 × 3 模式。UUU 的行数在 ACE 中对应节点，UVU/SH 常对应边；1024/8192 是独立吞吐规模，不能直接当作相同整模型图。算子加速比是配对批次比值的中位数，原始耗时中位数的比可能略有不同。显存记录的是 PyTorch allocated 的调用峰值增量，排除常驻权重/缓存，不代表 DRAM 流量；部分 TP 前向和 O2 ECE 二阶组合的峰值会增加。\n\n### Prophet 独立 profiler 诊断\n\n`performance/profile_prophet/profile.json` 和四份 Chrome traces 记录 FP32/N256/C64 的另一次仪器化采集，不能用 profiler 运行时间替代正常延迟。\n\n| 模式 | 后端 | 内核启动次数 | 累计内核时间 ms |\n| --- | --- | ---: | ---: |\n"""
    for r in profiles["results"]:
        text += f"| {r['mode']} | {r['backend']} | {r['cuda_kernel_launch_count']} | {r['sum_cuda_kernel_duration_us'] / 1000:.3f} |\n"
    text += """\nProfiler 显示 内核启动次数与累计时间减少，但短内核的包装、派发及参数准备仍有明显主机成本；energy 的 zeros 从 6 增至 45、arange 从 0 增至 8。双方 energy 均有 24 次动态 nonzero/stream sync，force_loss 均有 48 次。这支持主机派发与原模型同步限制整模型收益的推断，不能据此精确计算 Python 占比。整体 cat/clone/reshape 次数下降，证据不支持把回退归咎于总体布局操作增加。本轮只测性能，没有更改适配算子实现。\n"""
    readme = root / "README.md"
    marker = "\n## 2026-10-07 性能实测\n"
    prior = readme.read_text().split(marker)[0]
    prior = prior.replace("当前不宣称速度提升。", "该阶段只验证精度；性能结论见后文实测。")
    prior = prior.replace("未衡量运行速度或完整训练收敛。", "该阶段未衡量运行速度或完整训练收敛；后续性能测量见下文。")
    readme.write_text(prior + text)


if __name__ == "__main__":
    main()
