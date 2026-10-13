"""Summarize the complete, source-frozen three-way eager model experiment."""

import argparse
import csv
import hashlib
import json
import xml.etree.ElementTree as ET
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root
    output = root / "performance/optimization"
    report = json.loads((output / "models_final/models.json").read_text())
    rows = report["results"]
    required = {
        (m, d, n, mode)
        for m in ("tace", "tece", "prophet")
        for d in ("float32", "float64")
        for n in (64, 256)
        for mode in ("energy", "forces", "force_loss")
    }
    assert {(r["model"], r["dtype"], r["nodes"], r["mode"]) for r in rows} == required
    assert len(rows) == len(required) == 36
    for path, expected in report["metadata"]["optimized_source_sha256"].items():
        assert hashlib.sha256((root / path).read_bytes()).hexdigest() == expected, path
    fields = ["model", "dtype", "nodes", "mode"]
    for backend in ("original", "v1", "optimized"):
        fields += [
            f"{backend}_wall_ms",
            f"{backend}_wall_q25_ms",
            f"{backend}_wall_q75_ms",
            f"{backend}_event_ms",
        ]
    fields += [
        "v1_over_optimized_paired_median",
        "original_over_optimized_paired_median",
        "v1_over_optimized_paired_q25",
        "v1_over_optimized_paired_q75",
    ]
    with (output / "models_final/models.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            flat = {key: row[key] for key in fields[:4]}
            for backend in ("original", "v1", "optimized"):
                for stat, suffix in (("median", ""), ("q25", "_q25"), ("q75", "_q75")):
                    flat[f"{backend}_wall{suffix}_ms"] = row[backend]["wall_ms"][stat]
                flat[f"{backend}_event_ms"] = row[backend]["cuda_event_ms"]["median"]
            speed = row["paired_speedup"]
            flat.update(
                v1_over_optimized_paired_median=speed["v1_over_optimized"]["median"],
                original_over_optimized_paired_median=speed["original_over_optimized"][
                    "median"
                ],
                v1_over_optimized_paired_q25=speed["v1_over_optimized"]["q25"],
                v1_over_optimized_paired_q75=speed["v1_over_optimized"]["q75"],
            )
            writer.writerow(flat)
    xml = ET.parse(root / "validation/optimized/pytest.xml").getroot()
    suites = [xml] if xml.tag == "testsuite" else list(xml)
    precision = {
        key: sum(int(s.attrib.get(key, 0)) for s in suites)
        for key in ("tests", "failures", "errors", "skipped")
    }
    assert precision["failures"] == precision["errors"] == precision["skipped"] == 0
    worst = {}
    for dtype in ("float32", "float64"):
        values = [r["validation"]["optimized"] for r in rows if r["dtype"] == dtype]
        worst[dtype] = {
            key: max(v.get(key, 0) for v in values)
            for key in ("energy", "forces", "gradient_relative_l2")
        }
    summary = {
        "metadata": report["metadata"],
        "precision": precision,
        "worst_model_errors": worst,
        "model_case_count": len(rows),
        "results": rows,
        "excluded": [
            "models_interrupted_review",
            "Intermediate operator/model samples are used only for per-change attribution, not final whole-model latency.",
        ],
    }
    profile_path = output / "profile_prophet/profile.json"
    if profile_path.exists():
        profile = json.loads(profile_path.read_text())
        summary["profiler_diagnostic"] = [
            {
                key: row[key]
                for key in (
                    "mode",
                    "backend",
                    "cuda_kernel_launch_count",
                    "sum_cuda_kernel_duration_us",
                    "trace",
                )
            }
            for row in profile["results"]
        ]
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.4), sharey=True)
    modes = ("energy", "forces", "force_loss")
    names = ("Original", "First CUDA version", "Optimized CUDA")
    for i, dtype in enumerate(("float32", "float64")):
        for j, model in enumerate(("tace", "tece", "prophet")):
            selected = {
                r["mode"]: r
                for r in rows
                if r["model"] == model and r["dtype"] == dtype and r["nodes"] == 256
            }
            ax = axes[i, j]
            for k, (backend, color) in enumerate(
                zip(("original", "v1", "optimized"), ("#7b8794", "#f2a65a", "#278e7b"))
            ):
                values = [
                    selected[mode][backend]["wall_ms"]["median"]
                    / selected[mode]["original"]["wall_ms"]["median"]
                    for mode in modes
                ]
                ax.bar(
                    np.arange(3) + (k - 1) * 0.24,
                    values,
                    width=0.23,
                    label=names[k],
                    color=color,
                )
            ax.set_title(f"{model.upper()} / {dtype.upper()}")
            ax.set_xticks(np.arange(3), ("Energy", "Forces", "Force loss"))
            ax.axhline(1, color="#444444", linewidth=0.65, linestyle="--")
            ax.grid(axis="y", alpha=0.15)
            ax.set_axisbelow(True)
            if j == 0:
                ax.set_ylabel("Wall latency / original (lower is better)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
    fig.suptitle("H100 eager models: N=256, C=64, lmax=2, two layers", y=0.94)
    fig.tight_layout(rect=(0, 0, 1, 0.89))
    fig.savefig(output / "model_latency.png", dpi=180)
    fig.savefig(output / "model_latency.pdf")
    lines = [
        "# 等变 CUDA 优化结果",
        "",
        "本轮优化在独立环境中完成，复用 FastEq Uniform1D 与 EQX CUDA；没有修改原 FastEq 工作仓库。普通标量 MLP、softmax、邻居构建及图语义保留。",
        "",
        "实现包括 C++ 线性执行计划与原生 autograd、按需求导、共享权重梯度分块累加、通用等变程序的 C++ 执行计划与 warp 归约、索引/常量缓存、张量积布局转换及无用权重拼接的消除、O(2)/ECE 耦合接入同一执行计划，以及 Prophet 残差/Gate/RMSNorm 融合。Prophet 融合仅在 Block 内部启用，单独调用 TPConvolution 保留原输出；融合路径的 Gate/Norm 子模块 hook 调用边界会变化。",
        "",
        f"精度回归：{precision['tests']} 项通过，0 失败、0 跳过。FP32/FP64 的输出、一阶与二阶导数及三个模型的能量、力、含力损失参数梯度均校验；TACE/TECE 集成检查还包含应力/virials。未放宽原有容差。逐模型 JSON 和 JUnit 位于 `validation/optimized`。",
        "",
        "## 同一进程的三版本端到端配对",
        "",
        "H100、64 通道、lmax=2、两层、N64/256、每原子预建 12 条最近邻边，固定随机权重和几何。每项先校验，再对三个版本分别完整预热 5 次；12 轮轮换顺序，每轮每版本 5 次。采集墙钟和 CUDA event，不使用 profiler/CUDA Graph 计时。编译、预热及邻居构建排除；位置复制、几何和模型执行计入；force_loss 另含 zero_grad、loss 与 backward，不含 optimizer 更新和应力。",
        "",
        "每轮前后检查所选 GPU 是否有外部计算进程，源文件 SHA256 在案例前后再次校验。审查中发生源变动的第一轮全部归档排除，最终 36 项均重新采样，不能将跨运行绝对时间相减归因某个优化。",
        "",
        "下表是 N256：时间为各版本墙钟中位数，速比为每轮配对比值的中位数，所以速比可能与所列时间直接相除略有不同。完整样本、IQR、CUDA event 和误差在 `models_final/models.json`，平铺结果在 CSV。",
        "",
        "| 模型 | 精度 | 模式 | 原模型 ms | 第一版 ms | 优化版 ms | 配对加速 vs 第一版 | 配对加速 vs 原模型 |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in sorted(
        (r for r in rows if r["nodes"] == 256),
        key=lambda r: (
            ("tace", "tece", "prophet").index(r["model"]),
            r["dtype"],
            modes.index(r["mode"]),
        ),
    ):
        numbers = [
            row[backend]["wall_ms"]["median"]
            for backend in ("original", "v1", "optimized")
        ]
        speed = row["paired_speedup"]
        lines.append(
            f"| {row['model']} | {row['dtype']} | {row['mode']} | {numbers[0]:.3f} | {numbers[1]:.3f} | {numbers[2]:.3f} | {speed['v1_over_optimized']['median']:.3f}× | {speed['original_over_optimized']['median']:.3f}× |"
        )
    if "profiler_diagnostic" in summary:
        lines += [
            "",
            "## 独立 profiler 诊断",
            "",
            "另一次 Prophet FP32/N256/C64 采集显示实际启动次数和累计内核时间减少。仪器化时间不替代正常端到端延迟，也不能从两者差值计算 Python 占比。原模型的动态 nonzero/stream sync 在 energy 中仍为 24 次、force_loss 中为 48 次，图操作保留。完整 Chrome traces 和 CPU 算子记录在 profile_prophet。",
            "",
            "| 模式 | 后端 | CUDA 内核启动数 | 累计内核时间 ms |",
            "| --- | --- | ---: | ---: |",
        ]
        for row in summary["profiler_diagnostic"]:
            lines.append(
                f"| {row['mode']} | {row['backend']} | {row['cuda_kernel_launch_count']} | {row['sum_cuda_kernel_duration_us'] / 1000:.3f} |"
            )
    lines += [
        "",
        "## 证据边界与复现",
        "",
        "上述为正常 eager 模型计时。独立算子实验分别保存在 linear、edge_program、fusion 和 o2_final 子目录；其加速不能代替模型收益。CUDA event 包含派发空隙，不能当作纯内核执行时间；warp/分块累加的操作数量变化也不是硬件计数器。随机小模型精度与固定权重含力梯度测量不等于预训练模型、完整训练收敛或 MD 全流程验证。",
        "",
        "第一版源代码及哈希保存在 v1_snapshot；基准只改变导入位置和 dispatcher namespace 以并存两个版本，共享模型无关的编译/启动 ABI，算法、准备及反向公式均加载原快照。优化源码哈希随结果保存，汇总再次验证。",
        "",
        "```bash",
        f"source {root}/activate.sh",
        "# 选择空闲 GPU，并保持生产源码不变。",
        "export CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1",
        'python "$TACE_PROPHET_ROOT/src/FastEq/benchmarks/adapters/benchmark_optimization.py" --models prophet tace tece --nodes 64 256 --channels 64 --warmup 5 --rounds 12 --iterations 5 --output "$TACE_PROPHET_ROOT/performance/optimization/reproduced"',
        "```",
        "",
        "完整可复用补丁及提交基线、SHA256 在 patches/optimized-manifest.json；原第一版补丁保留，完整优化补丁从记录的 HEAD 应用，不与第一版补丁叠加。TACE 缓存修复仍单列。",
    ]
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(f"Verified {len(rows)} model cases; wrote report, CSV, JSON, PNG and PDF")


if __name__ == "__main__":
    main()
