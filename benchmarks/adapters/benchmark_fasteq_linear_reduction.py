"""TECE-sized shared dW: unchanged atomic branch versus bounded partial reduction.

This isolates two existing FastEq CUDA branches, rather than timing a model.
The upstream gradient is scaled by 1/N; model force-loss QA is separate.
"""

import argparse
import hashlib
import json
import os
import statistics
from pathlib import Path

import torch
from benchmark_models import assert_exclusive_gpu
from benchmark_operators import measure

from fasteq.cuda.jit import kernels
from fasteq.ops.indexed_linear import _groups, _plan


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--batches", type=int, default=7)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(20261009)
    assert_exclusive_gpu()
    # Establish a CUDA context before compiling this direct launch benchmark.
    torch.empty(0, device="cuda")
    directory = Path(__file__).resolve().parents[2] / "fasteq/cuda/src"
    body = (directory / "indexed_linear.cu").read_text()
    info = {
        "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
        "visibility": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "scope": "isolated dW at E3072, shared W; unchanged role4 atomic versus adaptive native role1",
        "gpu_guard": "NVML exclusive check before and after every paired batch",
        "upstream_scale": "1/N",
        "source_sha256": {
            name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
            for name in ("indexed_linear.cu", "indexed_linear_plan.cpp")
        },
        "protocol": {"iterations": args.iterations, "batches": args.batches, "warmup": 3},
    }
    rows = []
    for dtype in (torch.float32, torch.float64):
        scalar = "float" if dtype == torch.float32 else "double"
        source = f"#define FASTEQ_LINEAR_SCALAR {scalar}\n#define FASTEQ_LINEAR_ROLE 4\n" + body
        atomic_kernel = kernels([source], torch.device("cuda:0"))[source]
        for two_paths in (False, True):
            n = 3072
            paths = [(0, 0, 0, 384, 576, 1, 384**-0.5)]
            if two_paths:
                paths.append((384, 221184, 576, 256, 384, 2, 0.0625))
            din, dout, numel = (896, 1344, 319488) if two_paths else (384, 576, 221184)
            metadata = repr((din, dout, numel, 1, "ir_mul", tuple(paths)))
            x = torch.randn(n, din, device="cuda", dtype=dtype)
            y = torch.randn(n, dout, device="cuda", dtype=dtype) / n
            w = torch.empty(1, 1, numel, device="cuda", dtype=dtype)
            types = torch.zeros(n, device="cuda", dtype=torch.int64)
            ps = torch.tensor([p[:6] for p in paths], device="cuda", dtype=torch.int32)
            scales = torch.tensor([p[6] for p in paths], device="cuda", dtype=dtype)
            groups, ids, tiles, _ = _groups(paths, 1, 1)
            gs = torch.tensor(groups, device="cuda", dtype=torch.int32)
            pi = torch.tensor(ids, device="cuda", dtype=torch.int32)
            ts = torch.tensor(tiles, device="cuda", dtype=torch.int32)
            plan = _plan(metadata, dtype, x.device)
            stream = torch.cuda.current_stream().cuda_stream

            def atomic(
                x=x, y=y, types=types, ps=ps, scales=scales, gs=gs, pi=pi, ts=ts,
                n=n, din=din, dout=dout, numel=numel, kernel=atomic_kernel, stream=stream,
            ):
                output = x.new_zeros((1, 1, numel))
                packed = [
                    x.data_ptr(), 0, y.data_ptr(), types.data_ptr(), output.data_ptr(),
                    ps.data_ptr(), scales.data_ptr(), gs.data_ptr(), pi.data_ptr(), ts.data_ptr(),
                    n, din, dout, numel, 1, 1, 1, ts.shape[0], (n + 31) // 32, 32,
                ]
                kernel.launch(packed, n, ts.shape[0], 128, stream)
                return output

            functions = [atomic, lambda p=plan, t=types, x=x, w=w, y=y: p.contract(1, t, x, w, y)]
            old, new = (fn() for fn in functions)
            atol, rtol = (3e-5, 5e-5) if dtype == torch.float32 else (2e-11, 2e-10)
            torch.testing.assert_close(old, new, atol=atol, rtol=rtol)
            for fn in functions:
                for _ in range(3):
                    fn()
            times = [[], []]
            for batch in range(args.batches):
                assert_exclusive_gpu()
                for i in (0, 1) if batch % 2 == 0 else (1, 0):
                    _, wall = measure(functions[i], args.iterations)
                    times[i].append(wall)
                assert_exclusive_gpu()
            row = {
                "dtype": str(dtype), "nodes": n, "din": din, "dout": dout,
                "numel": numel, "atomic_wall_us": statistics.median(times[0]),
                "adaptive_wall_us": statistics.median(times[1]),
                "paired_speedup": statistics.median(
                    a / b for a, b in zip(times[0], times[1], strict=True)
                ),
                "max_abs_error": (old - new).abs().max().item(), "wall_samples_us": times,
            }
            rows.append(row)
            print(row, flush=True)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps({"metadata": info, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
