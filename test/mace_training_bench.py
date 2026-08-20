#!/usr/bin/env python3
"""Benchmark MACE force-loss training with e3nn or FastEq.

The timed step builds forces with ``training=True`` and differentiates a
force-only loss through them.  Consequently the backward phase contains the
second derivatives required by force training.  Model loading, graph building,
host-to-device transfer, and gradient clearing are outside the timed region.
"""

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.profiler import ProfilerActivity, profile, record_function

from mace.calculators import MACECalculator
from mace_bench_new import build_water_box, prepare_model_input


def stats(values):
    a = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(a.mean()),
        "median_ms": float(np.median(a)),
        "std_ms": float(a.std()),
        "p10_ms": float(np.percentile(a, 10)),
        "p90_ms": float(np.percentile(a, 90)),
    }


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_rev(path):
    try:
        return subprocess.check_output(
            ["git", "-C", str(path), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "unavailable"


def package_version(name):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not-installed"


def force_training_step(model, batch):
    output = model(
        batch,
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    forces = output["forces"]
    if forces is None or forces.grad_fn is None:
        raise RuntimeError("training=True did not produce differentiable forces")
    loss = forces.square().mean()
    loss.backward()
    return loss, output


def benchmark_one(model, batch, warmup, repeats, required_grad_name_fragment=None):
    model.train()
    for _ in range(warmup):
        model.zero_grad(set_to_none=True)
        loss, output = force_training_step(model, batch)
        del loss, output
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    raw = []
    for repeat in range(repeats):
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        total_start = torch.cuda.Event(enable_timing=True)
        forward_end = torch.cuda.Event(enable_timing=True)
        backward_start = torch.cuda.Event(enable_timing=True)
        backward_end = torch.cuda.Event(enable_timing=True)
        total_end = torch.cuda.Event(enable_timing=True)
        wall_start = time.perf_counter()

        total_start.record()
        output = model(
            batch,
            training=True,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
        forward_end.record()
        forces = output["forces"]
        if forces is None or forces.grad_fn is None:
            raise RuntimeError("training=True did not produce differentiable forces")
        loss = forces.square().mean()
        backward_start.record()
        loss.backward()
        backward_end.record()
        total_end.record()
        total_end.synchronize()
        wall_end = time.perf_counter()

        raw.append(
            {
                "repeat": repeat,
                "forward_cuda_ms": total_start.elapsed_time(forward_end),
                "backward_cuda_ms": backward_start.elapsed_time(backward_end),
                "total_cuda_ms": total_start.elapsed_time(total_end),
                "total_wall_ms": (wall_end - wall_start) * 1000.0,
                "loss": float(loss.detach()),
            }
        )
        del loss, forces, output

    parameter_grads = [
        (name, parameter.grad)
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    ]
    if not parameter_grads:
        raise RuntimeError("force-loss backward produced no model parameter gradients")
    if required_grad_name_fragment is not None and not any(
        required_grad_name_fragment in name for name, _ in parameter_grads
    ):
        raise RuntimeError(
            "force-loss backward produced no gradient for parameters containing "
            f"{required_grad_name_fragment!r}"
        )
    finite = all(
        parameter.grad is None or torch.isfinite(parameter.grad).all().item()
        for parameter in model.parameters()
    )
    if not finite:
        raise RuntimeError("non-finite parameter gradient after force-loss backward")
    grad_stats = {
        "parameter_tensors_with_grad": len(parameter_grads),
        "parameter_numel_with_grad": sum(grad.numel() for _, grad in parameter_grads),
    }
    return raw, torch.cuda.max_memory_allocated() / (1024**3), grad_stats


def profile_one(model, batch, trace_path, backend):
    model.zero_grad(set_to_none=True)
    activities = [ProfilerActivity.CPU, ProfilerActivity.CUDA]
    with profile(
        activities=activities, record_shapes=True, profile_memory=True
    ) as prof:
        with record_function(f"mace_{backend}_force_training_step"):
            loss, output = force_training_step(model, batch)
            torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=80)
    trace_path.with_suffix(".txt").write_text(table, encoding="utf-8")
    del loss, output


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument(
        "--backend",
        choices=("e3nn", "cueq", "fasteq", "fasteq_uniform1d"),
        default="e3nn",
    )
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--box-sides", type=int, nargs="+", default=[4, 6, 8])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=20)
    parser.add_argument("--density", type=float, default=1.0)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--profile", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    calculator = MACECalculator(
        model_paths=str(args.model),
        device=args.device,
        default_dtype=args.dtype,
        compile_mode=None,
        enable_cueq=args.backend != "e3nn",
        cueq_optimize_all=args.backend != "e3nn",
        cueq_optimize_channelwise=False,
        cueq_use_fasteq=args.backend == "fasteq",
        cueq_fasteq_uniform1d_only=args.backend == "fasteq_uniform1d",
        enable_oeq=False,
    )
    # MACECalculator freezes parameters for inference.  Re-enable them so the
    # measured force-loss backward is a genuine model-training operation and
    # includes mixed position/parameter second derivatives.
    for model in calculator.models:
        for parameter in model.parameters():
            parameter.requires_grad_(True)

    summary_rows = []
    raw_rows = []
    for side in args.box_sides:
        atoms = build_water_box(side, density_g_cm3=args.density, seed=1234 + side)
        model, batch, edges = prepare_model_input(atoms, calculator)
        raw, peak_memory_gb, grad_stats = benchmark_one(
            model,
            batch,
            args.warmup,
            args.repeats,
            required_grad_name_fragment=(
                "conv_tp_weights" if args.backend == "fasteq_uniform1d" else None
            ),
        )
        for row in raw:
            raw_rows.append({"n_side": side, "num_atoms": len(atoms), "num_edges": edges, **row})
        result = {
            "backend": args.backend,
            "dtype": args.dtype,
            "n_side": side,
            "num_atoms": len(atoms),
            "num_edges": edges,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "peak_memory_gb": peak_memory_gb,
            **grad_stats,
        }
        for field in ("forward_cuda_ms", "backward_cuda_ms", "total_cuda_ms", "total_wall_ms"):
            result.update({f"{field}_{key}": value for key, value in stats([r[field] for r in raw]).items()})
        summary_rows.append(result)
        print(
            f"{args.backend} {args.dtype} atoms={len(atoms)} edges={edges} "
            f"forward={result['forward_cuda_ms_median_ms']:.3f} ms "
            f"backward={result['backward_cuda_ms_median_ms']:.3f} ms "
            f"total={result['total_cuda_ms_median_ms']:.3f} ms "
            f"peak={peak_memory_gb:.3f} GiB"
        )
        if args.profile:
            profile_one(
                model,
                batch,
                args.output_dir
                / f"profile_{args.backend}_{args.dtype}_side{side}.json",
                args.backend,
            )
        del batch, model, atoms
        torch.cuda.empty_cache()

    for name, rows in (("summary.csv", summary_rows), ("raw.csv", raw_rows)):
        with (args.output_dir / name).open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

    repo_root = Path(__file__).resolve().parents[1]
    metadata = {
        "command": " ".join(os.sys.argv),
        "hostname": platform.node(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "cuequivariance": package_version("cuequivariance"),
        "cuequivariance_torch": package_version("cuequivariance-torch"),
        "cuequivariance_ops_torch_cu12": package_version(
            "cuequivariance-ops-torch-cu12"
        ),
        "gpu": torch.cuda.get_device_name(),
        "backend": args.backend,
        "model": str(args.model.resolve()),
        "model_sha256": sha256(args.model),
        "fasteq_commit": git_rev(repo_root),
        "mace_commit": git_rev(repo_root / "3rdparty" / "mace"),
        "trainable_parameters": sum(
            parameter.numel()
            for model in calculator.models
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
        "timed_region": "training=True force forward + mean(force^2).backward; zero_grad excluded",
    }
    (args.output_dir / "metadata.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
