"""Paired v3/new FastEq FFN with parameter gradients and force-loss chain."""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from benchmark_models import assert_exclusive_gpu, one_batch, stats
from e3nn import o3
from prophet.model import EQFFN
from v3_snapshot import load_v3

from fasteq.adapters import adapt_model


def run(model, scalar, features, mode):
    with torch.set_grad_enabled(mode != "energy"):
        s, x = model(scalar, features)
        if mode == "energy":
            return s, x
        energy = s.sin().square().mean() + x.tanh().square().mean()
        leaves = (scalar, features, *model.parameters())
        if mode == "backward":
            return torch.autograd.grad(energy, leaves)
        first = torch.autograd.grad(energy, (scalar, features), create_graph=True)
        return torch.autograd.grad(
            energy + sum(g.square().mean() for g in first), leaves
        )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = Path(os.environ["TACE_PROPHET_ROOT"])
    old_adapt, provenance = load_v3(root)
    paths = [
        root / "src/FastEq" / name
        for name in (
            "fasteq/ops/modulation.py",
            "fasteq/cuda/triton_launch.py",
            "fasteq/adapters/programs.py",
            "fasteq/ops/equivariant_norm.py",
            "fasteq/cuda/src/norm_plan.cpp",
            "fasteq/cuda/jit.py",
            "fasteq/cuda/src/jit_runtime.cpp",
        )
    ]

    def source_hashes():
        return {
            str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in paths
        }

    frozen_sources = source_hashes()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    report = {
        "metadata": {
            "v3": provenance,
            "gpu": str(torch.cuda.get_device_properties(0).uuid),
            "scope": "FFN including unchanged scalar MLP; force-loss gradients of inputs and all parameters",
            "source_sha256": frozen_sources,
            "protocol": {"warmup": 5, "rounds": 7, "iterations": 5},
        },
        "results": [],
    }
    a.output.parent.mkdir(parents=True, exist_ok=True)
    for dtype in (torch.float32, torch.float64):
        torch.manual_seed(1701)
        module = EQFFN(o3.Irreps("64x0e+64x1o+64x2e")).cuda().to(dtype)
        old, new = old_adapt(module), adapt_model(module)
        for n in (64, 256):
            scalar = torch.randn(n, 64, device="cuda", dtype=dtype, requires_grad=True)
            features = torch.randn(
                n, 576, device="cuda", dtype=dtype, requires_grad=True
            )
            for mode in ("energy", "backward", "force_loss"):
                assert source_hashes() == frozen_sources
                functions = [
                    lambda m=m, scalar=scalar, features=features, mode=mode: run(
                        m, scalar, features, mode
                    )
                    for m in (old, new)
                ]
                before, after = (fn() for fn in functions)
                max_error = 0.0
                for x, y in zip(before, after, strict=True):
                    torch.testing.assert_close(
                        x,
                        y,
                        atol=3e-5 if dtype == torch.float32 else 2e-10,
                        rtol=3e-4 if dtype == torch.float32 else 2e-9,
                    )
                    max_error = max(max_error, (x - y).abs().max().item())
                for fn in functions:
                    for _ in range(5):
                        fn()
                torch.cuda.synchronize()
                samples = [[], []]
                for block in range(7):
                    assert_exclusive_gpu()
                    for i in (0, 1) if block % 2 == 0 else (1, 0):
                        samples[i].append(one_batch(functions[i], 5))
                    assert_exclusive_gpu()
                assert source_hashes() == frozen_sources
                row = {
                    "dtype": str(dtype),
                    "nodes": n,
                    "mode": mode,
                    "max_abs_error": max_error,
                    "v3": stats([x["wall_ms"] for x in samples[0]]),
                    "optimized": stats([x["wall_ms"] for x in samples[1]]),
                    "paired_speedup": stats(
                        [
                            x["wall_ms"] / y["wall_ms"]
                            for x, y in zip(*samples, strict=True)
                        ]
                    ),
                    "samples": samples,
                }
                report["results"].append(row)
                a.output.write_text(json.dumps(report, indent=2) + "\n")
                print(
                    dtype,
                    n,
                    mode,
                    row["v3"]["median"],
                    row["optimized"]["median"],
                    row["paired_speedup"]["median"],
                    flush=True,
                )


if __name__ == "__main__":
    main()
