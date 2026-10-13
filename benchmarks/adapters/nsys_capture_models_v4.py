"""Warmed four-way CUDA timelines with only whole-step NVTX annotations.

Run under nsys --capture-range=cudaProfilerApi. Normal timing and validation
precede cudaProfilerStart. This does not install module/profiler hooks.
"""

import argparse
import hashlib
import json
import os
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
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"])
    )
    parser.add_argument("--model", choices=("tace", "tece", "prophet"), required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), required=True)
    parser.add_argument(
        "--mode", choices=("energy", "forces", "force_loss"), required=True
    )
    parser.add_argument("--nodes", type=int, default=256)
    parser.add_argument("--steps", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert_exclusive_gpu()
    frozen = json.loads(
        (
            args.root / "performance/fasteq_optimization/v4_snapshot/manifest.json"
        ).read_text()
    )["files_sha256"]

    def check_sources():
        for path, expected in frozen.items():
            assert (
                hashlib.sha256((args.root / path).read_bytes()).hexdigest() == expected
            ), path

    check_sources()
    mixed, _ = load_mixed(args.root)
    v3, _ = load_v3(args.root)
    original = build_model(args.model, 64, getattr(torch, args.dtype), 2)
    models = [original, mixed(original), v3(original), adapt_model(original)]
    labels = ("original", "mixed", "v3", "v4")
    data = build_data(args.nodes, 12, getattr(torch, args.dtype))
    for model in models:
        configure(model, args.model, args.mode)
    validation = {
        label: validate_pair(
            [original, model], args.model, data, args.mode, getattr(torch, args.dtype)
        )
        for label, model in zip(labels[1:], models[1:], strict=True)
    }
    for model in models:
        for _ in range(5):
            execute(model, args.model, data, args.mode)
    torch.cuda.synchronize()
    samples = {label: [] for label in labels}
    for block in range(7):
        assert_exclusive_gpu()
        order = [(block + i) % 4 for i in range(4)]
        if block % 2:
            order.reverse()
        for i in order:
            samples[labels[i]].append(
                one_batch(
                    lambda model=models[i]: execute(model, args.model, data, args.mode),
                    3,
                )
            )
        assert_exclusive_gpu()
    report = {
        "model": args.model,
        "dtype": args.dtype,
        "mode": args.mode,
        "nodes": args.nodes,
        "edges": int(data["edge_index"].shape[1]),
        "cpu_affinity": sorted(os.sched_getaffinity(0)),
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "source_sha256": frozen,
        "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "validation": validation,
        "normal_timing": {
            label: {
                "wall_ms": stats([s["wall_ms"] for s in values]),
                "cuda_event_ms": stats([s["cuda_event_ms"] for s in values]),
                "samples": values,
            }
            for label, values in samples.items()
        },
        "capture_steps_per_backend": args.steps,
        "capture_note": "No torch.profiler, module hooks or emit_nvtx; one NVTX range per full step. Captured timelines remain instrumented, separate from normal timing.",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    check_sources()
    torch.cuda.profiler.start()
    try:
        for label, model in zip(labels, models, strict=True):
            for step in range(args.steps):
                assert_exclusive_gpu()
                torch.cuda.synchronize()
                marker = (
                    f"step::{args.model}::{args.dtype}::{args.mode}::{label}::{step}"
                )
                with torch.cuda.nvtx.range(marker):
                    output = execute(model, args.model, data, args.mode)
                    del output
                    torch.cuda.synchronize()
                assert_exclusive_gpu()
    finally:
        torch.cuda.profiler.stop()
    check_sources()
    print("CAPTURE_DONE", args.model, args.dtype, args.mode, flush=True)


if __name__ == "__main__":
    main()
