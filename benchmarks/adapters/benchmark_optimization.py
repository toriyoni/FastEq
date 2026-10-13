"""Interleaved original/v1/optimized model timing in the same CUDA process."""

import argparse
import gc
import hashlib
import json
import os
import time
from pathlib import Path

import torch
from benchmark_models import (
    assert_exclusive_gpu,
    build_data,
    build_model,
    configure,
    execute,
    one_batch,
    stats,
    validate_pair,
)
from v1_snapshot import load_v1

from fasteq.adapters import adapt_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    p.add_argument("--models", nargs="+", default=["tace", "tece", "prophet"])
    p.add_argument("--nodes", nargs="+", type=int, default=[64, 256])
    p.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    p.add_argument("--modes", nargs="+", default=["energy", "forces", "force_loss"])
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--rounds", type=int, default=9)
    p.add_argument("--iterations", type=int, default=5)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert a.warmup > 0 and a.rounds >= 3 and a.iterations > 0
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    v1_adapt, provenance = load_v1(a.root)
    source_paths = [a.root / row["path"] for row in provenance["manifest"]["files"]]
    source_paths.extend(
        (a.root / "src/FastEq/fasteq/adapters" / name)
        for name in ("cache.py", "fusion.py")
    )

    def source_hashes():
        return {
            str(path.relative_to(a.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        }

    frozen_sources = source_hashes()
    a.output.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "comparison": "same weights, graph, process: original, first CUDA adaptation, optimized CUDA adaptation",
            "model_config": {
                "channels": a.channels,
                "layers": 2,
                "lmax": 2,
                "correlation": 2,
                "neighbors": 12,
            },
            "protocol": {
                "warmup": a.warmup,
                "rounds": a.rounds,
                "iterations": a.iterations,
                "order": "three-way rotating order per round; wall + CUDA-event timeline; no profiler/capture",
            },
            "scope": "geometry + cloned positions + model; force_loss adds zero_grad/loss/backward, no optimizer; prebuilt neighbors; no stress",
            "gpu_guard": "NVML exclusive compute-process check before and after each three-way round",
            "v1": provenance,
            "optimized_source_sha256": frozen_sources,
        },
        "results": [],
    }
    for name in a.models:
        for dtype_name in a.dtypes:
            dtype = getattr(torch, dtype_name)
            original = build_model(name, a.channels, dtype, 2)
            models = [original, v1_adapt(original), adapt_model(original)]
            for nodes in a.nodes:
                data = build_data(nodes, 12, dtype)
                for mode in a.modes:
                    print(f"START {name} {dtype_name} N{nodes} {mode}", flush=True)
                    assert source_hashes() == frozen_sources, (
                        "Sources changed during benchmarking"
                    )
                    for model in models:
                        configure(model, name, mode)
                    assert_exclusive_gpu()
                    validation = {
                        "v1": validate_pair(
                            [models[0], models[1]], name, data, mode, dtype
                        ),
                        "optimized": validate_pair(
                            [models[0], models[2]], name, data, mode, dtype
                        ),
                    }
                    fns = [
                        lambda m=m, name=name, data=data, mode=mode: execute(
                            m, name, data, mode
                        )
                        for m in models
                    ]
                    warmup = []
                    for fn in fns:
                        before = time.perf_counter()
                        for _ in range(a.warmup):
                            out = fn()
                            del out
                        torch.cuda.synchronize()
                        warmup.append(time.perf_counter() - before)
                    samples = [[], [], []]
                    for block in range(a.rounds):
                        assert_exclusive_gpu()
                        order = [(block + i) % 3 for i in range(3)]
                        if block % 2:
                            order.reverse()
                        for i in order:
                            samples[i].append(one_batch(fns[i], a.iterations))
                        assert_exclusive_gpu()
                    assert source_hashes() == frozen_sources, (
                        "Sources changed during benchmarking; discard case"
                    )
                    metrics = {
                        label: {
                            "wall_ms": stats([s["wall_ms"] for s in ss]),
                            "cuda_event_ms": stats([s["cuda_event_ms"] for s in ss]),
                            "samples": ss,
                            "warmup_seconds": cold,
                        }
                        for label, ss, cold in zip(
                            ("original", "v1", "optimized"), samples, warmup
                        )
                    }
                    ratios = {
                        "original_over_optimized": stats(
                            [
                                x["wall_ms"] / y["wall_ms"]
                                for x, y in zip(samples[0], samples[2])
                            ]
                        ),
                        "v1_over_optimized": stats(
                            [
                                x["wall_ms"] / y["wall_ms"]
                                for x, y in zip(samples[1], samples[2])
                            ]
                        ),
                        "original_over_v1": stats(
                            [
                                x["wall_ms"] / y["wall_ms"]
                                for x, y in zip(samples[0], samples[1])
                            ]
                        ),
                    }
                    report["results"].append(
                        {
                            "model": name,
                            "dtype": dtype_name,
                            "nodes": nodes,
                            "mode": mode,
                            "validation": validation,
                            **metrics,
                            "paired_speedup": ratios,
                        }
                    )
                    (a.output / "models.json").write_text(
                        json.dumps(report, indent=2) + "\n"
                    )
                    print(
                        "DONE "
                        + " -> ".join(
                            f"{metrics[label]['wall_ms']['median']:.3f}ms"
                            for label in ("original", "v1", "optimized")
                        )
                        + f"; v1/new {ratios['v1_over_optimized']['median']:.3f}x",
                        flush=True,
                    )
            del models, original, data, fns
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
