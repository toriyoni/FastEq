"""Interleaved original/mixed/v3/optimized FastEq model timing in the same CUDA process."""

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
from mixed_snapshot import load_mixed
from v3_snapshot import load_v3

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
    p.add_argument("--resume", action="store_true")
    a = p.parse_args()
    assert a.warmup > 0 and a.rounds >= 3 and a.iterations > 0
    assert os.environ.get("CUDA_LAUNCH_BLOCKING") != "1"
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    mixed_adapt, provenance = load_mixed(a.root)
    v3_adapt, v3_provenance = load_v3(a.root)
    source_paths = []
    for relative in (
        "src/FastEq/fasteq/adapters",
        "src/FastEq/fasteq/ops",
        "src/FastEq/fasteq/triton",
        "src/FastEq/fasteq/cuda",
        "src/tace/tace/models",
        "src/tace/eqx",
        "src/prophet/prophet",
    ):
        source_paths.extend(
            path
            for path in (a.root / relative).rglob("*")
            if path.is_file()
            and path.suffix in (".py", ".cpp", ".cu", ".h", ".cuh")
            and "__pycache__" not in path.parts
            and "generated" not in str(path)
        )

    def source_hashes():
        return {
            str(path.relative_to(a.root)): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in source_paths
        }

    frozen_sources = source_hashes()
    source_digest = hashlib.sha256(
        json.dumps(frozen_sources, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    driver_hashes = {
        name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
        for name in (
            "benchmark_fasteq_optimized_models.py",
            "benchmark_models.py",
            "mixed_snapshot.py",
            "v3_snapshot.py",
        )
    }
    a.output.mkdir(parents=True, exist_ok=True)
    report = {
        "metadata": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(),
            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
            "comparison": "same weights, graph, process: original, mixed CUDA, frozen FastEq v3, optimized FastEq v4",
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
                "order": "four-way rotating order per round; wall + CUDA-event timeline; no profiler/capture",
            },
            "scope": "geometry + cloned positions + model; force_loss adds zero_grad/loss/backward, no optimizer; prebuilt neighbors; no stress",
            "gpu_guard": "NVML exclusive compute-process check before and after each four-way round",
            "mixed": provenance,
            "v3": v3_provenance,
            "fasteq_source_sha256": frozen_sources,
            "benchmark_driver_sha256": driver_hashes,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
        },
        "results": [],
    }
    output_path = a.output / "models.json"
    current_run = {
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "gpu": torch.cuda.get_device_name(),
    }
    report["metadata"]["runs"] = [current_run]
    if a.resume and output_path.exists():
        old = json.loads(output_path.read_text())
        for key in (
            "torch",
            "cuda",
            "model_config",
            "protocol",
            "fasteq_source_sha256",
            "benchmark_driver_sha256",
            "cpu_affinity",
        ):
            assert old["metadata"][key] == report["metadata"][key], key
        for row in old["results"]:
            row.setdefault("visible_device", old["metadata"]["visible_device"])
        report["results"] = old["results"]
        runs = old["metadata"].get(
            "runs",
            [
                {
                    "visible_device": old["metadata"]["visible_device"],
                    "gpu": old["metadata"]["gpu"],
                }
            ],
        )
        report["metadata"]["runs"] = runs + [current_run]
        report["metadata"]["resume"] = (
            "Only completed cases with matching source hashes/configuration/protocol retained; each case compares all four implementations in one process/GPU."
        )
    done = {
        (row["model"], row["dtype"], row["nodes"], row["mode"])
        for row in report["results"]
    }
    for name in a.models:
        for dtype_name in a.dtypes:
            if all(
                (name, dtype_name, n, mode) in done for n in a.nodes for mode in a.modes
            ):
                continue
            dtype = getattr(torch, dtype_name)
            original = build_model(name, a.channels, dtype, 2)
            models = [
                original,
                mixed_adapt(original),
                v3_adapt(original),
                adapt_model(original),
            ]
            for nodes in a.nodes:
                data = build_data(nodes, 12, dtype)
                for mode in a.modes:
                    if (name, dtype_name, nodes, mode) in done:
                        continue
                    print(f"START {name} {dtype_name} N{nodes} {mode}", flush=True)
                    assert source_hashes() == frozen_sources, (
                        "Sources changed during benchmarking"
                    )
                    for model in models:
                        configure(model, name, mode)
                    assert_exclusive_gpu()
                    validation = {
                        "mixed": validate_pair(
                            [models[0], models[1]], name, data, mode, dtype
                        ),
                        "v3": validate_pair(
                            [models[0], models[2]], name, data, mode, dtype
                        ),
                        "fasteq": validate_pair(
                            [models[0], models[3]], name, data, mode, dtype
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
                    samples = [[], [], [], []]
                    for block in range(a.rounds):
                        assert_exclusive_gpu()
                        order = [(block + i) % 4 for i in range(4)]
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
                            ("original", "mixed", "v3", "fasteq"), samples, warmup
                        )
                    }
                    ratios = {
                        "original_over_fasteq": stats(
                            [
                                x["wall_ms"] / y["wall_ms"]
                                for x, y in zip(samples[0], samples[3])
                            ]
                        ),
                        "mixed_over_fasteq": stats(
                            [
                                x["wall_ms"] / y["wall_ms"]
                                for x, y in zip(samples[1], samples[3])
                            ]
                        ),
                        "v3_over_fasteq": stats(
                            [
                                x["wall_ms"] / y["wall_ms"]
                                for x, y in zip(samples[2], samples[3])
                            ]
                        ),
                        "original_over_mixed": stats(
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
                            "visible_device": current_run["visible_device"],
                            "fasteq_source_sha256_digest": source_digest,
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
                            for label in ("original", "mixed", "v3", "fasteq")
                        )
                        + f"; v3/new {ratios['v3_over_fasteq']['median']:.3f}x; mixed/new {ratios['mixed_over_fasteq']['median']:.3f}x",
                        flush=True,
                    )
            del models, original, data, fns
            gc.collect()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
