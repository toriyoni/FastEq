"""Opt-in FP32 SIMT candidate versus independent frozen v5.

Normal timing is rejected on a shared GPU. Explicit shared precision mode
records no timings; optional CUPTI records establish launch identity only.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from benchmark_linear_v5 import errors, force_loss, metadata
from benchmark_models import assert_exclusive_gpu, one_batch, stats
from fasteq.ops.indexed_linear import _plan
from v5_snapshot import load_v5


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--precision-only-shared", action="store_true")
    parser.add_argument("--launch-proof", action="store_true")
    parser.add_argument("--nodes", nargs="+", type=int, default=[3073])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float64"), default=["float32"])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    os.sched_setaffinity(0, {160})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["FASTEQ_EXPERIMENTAL_LINEAR_SIMT"] = "1"
    _plan.cache_clear()
    shared_reason = None
    if args.precision_only_shared:
        try:
            assert_exclusive_gpu()
        except RuntimeError as exception:
            shared_reason = str(exception)
    else:
        assert_exclusive_gpu()  # no exception-catching or bypass in timing mode
    owned = [args.root / "src/FastEq/fasteq" / name for name in (
        "ops/indexed_linear.py", "cuda/src/indexed_linear.cu", "cuda/src/indexed_linear_plan.cpp")]

    def hashes():
        return {str(path.relative_to(args.root)): hashlib.sha256(path.read_bytes()).hexdigest() for path in owned}

    frozen = hashes()
    _, provenance = load_v5(args.root)
    baseline_factory = sys.modules["fasteq_v5.ops.indexed_linear"]._plan
    report = {"metadata": {"scope": "shared GPU precision only" if args.precision_only_shared else "exclusive GPU paired operator timing",
              "performance": None, "opt_in": "FASTEQ_EXPERIMENTAL_LINEAR_SIMT=1 before first plan creation",
              "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid), "cpu_affinity": [160],
              "normal_timing_guard_rejection": shared_reason,
              "source_sha256": frozen, "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "baseline": provenance, "tf32": False}, "results": []}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for name in ("up", "coefs"):
            din, dout, numel = (896, 1344, 319488) if name == "up" else (384, 576, 221184)
            for nodes in args.nodes:
                assert hashes() == frozen
                torch.cuda.reset_peak_memory_stats()
                torch.manual_seed(20261013)
                x, y = [torch.randn(nodes, size, device="cuda", dtype=dtype) for size in (din, dout)]
                w = torch.randn(1, 1, numel, device="cuda", dtype=dtype)
                ids = torch.zeros(nodes, device="cuda", dtype=torch.int64)
                plans = [factory(metadata(name), dtype, x.device) for factory in (baseline_factory, _plan)]
                for direction, role in (("dx", 0), ("dw", 1), ("forward", 2), ("force_loss", None)):
                    if role is None:
                        leaves = [(x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()) for _ in plans]
                        functions = [lambda p=p, a=a, b=b, ids=ids: force_loss(p, a, b, ids)
                                     for p, (a, b) in zip(plans, leaves, strict=True)]
                    else:
                        functions = [lambda p=p, role=role, ids=ids, x=x, w=w, y=y: p.contract(role, ids, x, w, y) for p in plans]
                    reference, candidate = [fn() for fn in functions]
                    pairs = list(zip(candidate, reference, strict=True)) if role is None else [(candidate, reference)]
                    precision = [errors(a, b) for a, b in pairs]
                    for a, b in pairs:
                        torch.testing.assert_close(a, b, atol=3e-5 if dtype == torch.float32 else 2e-11,
                                                   rtol=5e-5 if dtype == torch.float32 else 2e-10)
                    row = {"dtype": dtype_name, "map": name, "nodes": nodes, "direction": direction,
                           "precision": precision, "performance": None,
                           "peak_allocated_bytes": torch.cuda.max_memory_allocated()}
                    if args.launch_proof and role is not None:
                        with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]) as profile:
                            functions[1]()
                            torch.cuda.synchronize()
                        path = args.output.parent / f"{dtype_name}_{name}_{nodes}_{direction}.launch.trace.json"
                        profile.export_chrome_trace(str(path))
                        events = json.loads(path.read_text())["traceEvents"]
                        row["launch_proof"] = {"note": "CUPTI identity only; durations are not used as performance",
                            "trace": str(path), "kernels": [{"name": e["name"], "grid": e["args"].get("grid"),
                                "block": e["args"].get("block"), "shared_bytes": e["args"].get("shared memory")}
                                for e in events if e.get("cat") == "kernel" and e.get("ph") == "X"]}
                    if not args.precision_only_shared:
                        for fn in functions:
                            for _ in range(5):
                                fn()
                        samples = [[], []]
                        for block in range(args.rounds):
                            assert_exclusive_gpu()
                            assert hashes() == frozen
                            for i in ((0, 1) if block % 2 == 0 else (1, 0)):
                                samples[i].append(one_batch(functions[i], 3 if role is None else args.iterations))
                            assert_exclusive_gpu()
                            assert hashes() == frozen
                        row["performance"] = {"samples": {"v5": samples[0], "v6_opt_in": samples[1]},
                            "wall_ms": {label: stats([s["wall_ms"] for s in sample])
                                        for label, sample in zip(("v5", "v6_opt_in"), samples, strict=True)},
                            "paired_speedup": stats([a["wall_ms"] / b["wall_ms"] for a, b in zip(*samples, strict=True)])}
                    report["results"].append(row)
                    assert hashes() == frozen
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(dtype_name, name, nodes, direction, precision, "timing", row["performance"] is not None, flush=True)
    report["completed"] = True
    args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
