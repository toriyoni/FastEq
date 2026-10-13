"""Isolated paired FastEq general CUDA versus reused FP64 MMA plan forward."""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
from benchmark_models import assert_exclusive_gpu
from benchmark_operators import measure
from e3nn import o3

from fasteq.ops.indexed_linear import IndexedLinear, _plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", type=int, default=[1024])
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "/public-data/zhouxibo/zxb/tace_prophet_fasteq/performance/native_fasteq/linear/mma_paired_final.json"
        ),
    )
    args = parser.parse_args()
    if any(nodes < 1024 for nodes in args.nodes):
        parser.error("The frozen MMA branch requires N>=1024; small-N falls back.")
    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    assert_exclusive_gpu()
    package = Path(__file__).resolve().parents[2] / "fasteq"
    source_paths = [
        package / "ops/indexed_linear.py",
        package / "cuda/jit.py",
        package / "cuda/src/indexed_linear.cu",
        package / "cuda/src/indexed_linear_plan.cpp",
        package / "cuda/src/jit_runtime.cpp",
        package / "cuda/src/jit_runtime.h",
        package / "cuda/src/equi_linear_kernels/ptx_inst.cuh",
    ]

    def source_hashes():
        return {
            str(path.relative_to(package.parent)): hashlib.sha256(
                path.read_bytes()
            ).hexdigest()
            for path in source_paths
        }

    frozen_sources = source_hashes()
    metadata = {
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "source_sha256": frozen_sources,
        "scope": "native C++ plan forward; FP64 shared element weights; no EQX backend",
        "gpu_guard": "NVML exclusive compute-process check before and after each paired batch",
        "protocol": {
            "iterations": args.iterations,
            "batches": args.batches,
            "warmup": 5,
            "order": "alternating generic/MMA order per batch",
            "excluded": "compilation, validation, warmup, NVML/source checks",
        },
    }
    rows = []
    for channels in (64, 128):
        for layout in ("mul_ir", "ir_mul"):
            ir = o3.Irreps(
                [(channels, (degree, (-1) ** degree)) for degree in range(3)]
            )
            module = IndexedLinear(
                o3.Linear(ir, ir, internal_weights=False), layout=layout
            )
            for nodes in args.nodes:
                assert source_hashes() == frozen_sources
                assert_exclusive_gpu()
                x = torch.randn(
                    nodes, module.input_dim, device="cuda", dtype=torch.float64
                )
                w = torch.randn(
                    1, 1, module.weight_numel, device="cuda", dtype=torch.float64
                )
                ids = torch.zeros(nodes, device="cuda", dtype=torch.int64)
                mma = _plan(module.metadata, x.dtype, x.device)
                generic = _plan(module.metadata + " ", x.dtype, x.device)
                generic.configure_tensor_core(
                    torch.empty(0, 4, device="cuda", dtype=torch.int32), 5
                )
                functions = [
                    lambda p=p, x=x, w=w, ids=ids: p.forward(x, w, ids)
                    for p in (generic, mma)
                ]
                a, b = (fn() for fn in functions)
                torch.testing.assert_close(a, b, atol=2e-11, rtol=2e-10)
                for fn in functions:
                    for _ in range(5):
                        fn()
                times = [[], []]
                for batch in range(args.batches):
                    assert_exclusive_gpu()
                    for i in ((0, 1) if batch % 2 == 0 else (1, 0)):
                        _, wall = measure(functions[i], args.iterations)
                        times[i].append(wall)
                    assert_exclusive_gpu()
                assert source_hashes() == frozen_sources, (
                    "Sources changed during benchmarking; discard case"
                )
                row = {
                    "channels": channels,
                    "nodes": nodes,
                    "layout": layout,
                    "generic_wall_us": statistics.median(times[0]),
                    "mma_wall_us": statistics.median(times[1]),
                    "paired_speedup": statistics.median(
                        [a / b for a, b in zip(times[0], times[1], strict=True)]
                    ),
                    "max_abs_error": (a - b).abs().max().item(),
                    "wall_samples_us": times,
                }
                rows.append(row)
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(
                    json.dumps({"metadata": metadata, "results": rows}, indent=2)
                )
                print(row, flush=True)


if __name__ == "__main__":
    main()
