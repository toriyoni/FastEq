"""Paired e3nn versus general FastEq Linear (CUDA only; no EQX backend)."""

import argparse
import hashlib
import json
import os
import statistics
import subprocess
from pathlib import Path

import torch
from benchmark_models import assert_exclusive_gpu
from benchmark_operators import compare, measure, operation
from e3nn import o3

from fasteq.ops.indexed_linear import IndexedLinear


class Map(torch.nn.Module):
    def __init__(self, desc, weight, indices, accelerated):
        super().__init__()
        self.desc = desc
        self.weight = torch.nn.Parameter(weight.clone())
        self.register_buffer("indices", indices, persistent=False)
        self.accelerated = accelerated
        self.plan = IndexedLinear(desc)

    def forward(self, x):
        if self.accelerated:
            return self.plan(x, self.weight, self.indices)
        weights = (
            self.weight[0, 0]
            if self.desc.shared_weights
            else self.weight[self.indices, 0]
        )
        return self.desc(x, weights)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nodes", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    parser.add_argument("--elements", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
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
    info = {
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "timing": "paired alternating eager batches; compilation/checks/warmup excluded",
        "backend": "FastEq native CUDA tiled indexed Linear, FP32 uses FP32 arithmetic",
        "source_sha256": frozen_sources,
        "gpu_guard": "NVML exclusive compute-process check before and after each paired batch",
        "protocol": {
            "iterations": args.iterations,
            "batches": args.batches,
            "warmup": 5,
            "order": "alternating reference/FastEq order per batch",
            "tf32": False,
        },
        "gpu_state_start": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
                "--format=csv",
            ],
            text=True,
        ),
    }
    rows = []
    for name in args.dtypes:
        dtype = getattr(torch, name)
        for n in args.nodes:
            for elements in args.elements:
                ir = o3.Irreps(
                    [(args.channels, (degree, (-1) ** degree)) for degree in range(3)]
                )
                desc = (
                    o3.Linear(
                        ir, ir, internal_weights=False, shared_weights=elements == 1
                    )
                    .cuda()
                    .to(dtype)
                )
                indices = torch.arange(n, device="cuda") % elements
                weight = torch.randn(
                    elements, 1, desc.weight_numel, device="cuda", dtype=dtype
                )
                models = [
                    Map(desc, weight, indices, accelerated).cuda()
                    for accelerated in (False, True)
                ]
                x = torch.randn(n, ir.dim, device="cuda", dtype=dtype)
                inputs = [x.detach().clone().requires_grad_() for _ in models]
                cotangent = torch.randn_like(x)
                for mode in ("forward", "backward", "force_loss"):
                    assert source_hashes() == frozen_sources, (
                        "Sources changed during benchmarking"
                    )
                    assert_exclusive_gpu()
                    functions = [
                        lambda m=m, t=t, cotangent=cotangent, mode=mode: operation(
                            m, [t], cotangent, mode
                        )
                        for m, t in zip(models, inputs, strict=True)
                    ]
                    error = compare(*[fn() for fn in functions], dtype)
                    for fn in functions:
                        for _ in range(5):
                            fn()
                    events, walls = [[], []], [[], []]
                    for batch in range(args.batches):
                        assert_exclusive_gpu()
                        for i in (0, 1) if batch % 2 == 0 else (1, 0):
                            event, wall = measure(functions[i], args.iterations)
                            events[i].append(event)
                            walls[i].append(wall)
                        assert_exclusive_gpu()
                    assert source_hashes() == frozen_sources, (
                        "Sources changed during benchmarking; discard case"
                    )
                    row = {
                        "dtype": name,
                        "nodes": n,
                        "elements": elements,
                        "channels": args.channels,
                        "mode": mode,
                        "e3nn_event_us": statistics.median(events[0]),
                        "fasteq_event_us": statistics.median(events[1]),
                        "e3nn_wall_us": statistics.median(walls[0]),
                        "fasteq_wall_us": statistics.median(walls[1]),
                        "paired_speedup": statistics.median(
                            [a / b for a, b in zip(walls[0], walls[1], strict=True)]
                        ),
                        "event_samples_us": events,
                        "wall_samples_us": walls,
                        "max_abs_error": error,
                    }
                    rows.append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(
                        json.dumps({"metadata": info, "results": rows}, indent=2)
                    )
                    print(row, flush=True)
    info["gpu_state_end"] = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory",
            "--format=csv",
        ],
        text=True,
    )
    args.output.write_text(json.dumps({"metadata": info, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
