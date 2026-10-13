"""Paired TECE Linear directions against independently compiled frozen v4.

This is an operator benchmark, including a genuine energy/force-loss chain;
it does not measure full-model latency or hardware counters.
"""

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import torch
from benchmark_models import assert_exclusive_gpu, one_batch, stats
from v4_snapshot import load_v4

from fasteq.ops.indexed_linear import _plan


def metadata(name, layout="ir_mul", experts=1):
    paths = [(0, 0, 0, 384, 576, 1, 384**-0.5)]
    if name == "up":
        paths.append((384 * experts, 221184, 576 * experts, 256, 384, 2, 0.0625))
    din, dout, numel = (896, 1344, 319488) if name == "up" else (384, 576, 221184)
    return repr((din * experts, dout * experts, numel, experts, layout, tuple(paths)))


def force_loss(plan, x, w, ids):
    output = plan.forward(x, w, ids)
    energy = output.square().mean()
    force = -torch.autograd.grad(energy, x, create_graph=True)[0]
    objective = 0.1 * energy.square() + force.square().mean()
    return torch.autograd.grad(objective, (x, w))


def errors(actual, expected):
    return {"max_absolute": (actual - expected).abs().max().item(),
            "relative_l2": (actual.double() - expected.double()).norm().item()
            / max(expected.double().norm().item(), 1e-300)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nodes", nargs="+", type=int, default=[3072])
    parser.add_argument("--maps", nargs="+", choices=("up", "coefs"), default=["up", "coefs"])
    parser.add_argument("--dtypes", nargs="+", choices=("float32", "float64"), default=["float64"])
    parser.add_argument("--rounds", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--cpu", type=int, default=160)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert_exclusive_gpu()
    source_paths = [args.root / "src/FastEq/fasteq" / name for name in (
        "ops/indexed_linear.py", "cuda/src/indexed_linear.cu", "cuda/src/indexed_linear_plan.cpp")]

    def hashes():
        return {str(p.relative_to(args.root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in source_paths}

    frozen = hashes()
    _, provenance = load_v4(args.root)
    baseline_plan = sys.modules["fasteq_v4.ops.indexed_linear"]._plan
    report = {"metadata": {"gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
              "cpu_affinity": sorted(os.sched_getaffinity(0)), "torch": torch.__version__, "cuda": torch.version.cuda,
              "source_sha256": frozen, "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "baseline": provenance, "scope": "standalone real TECE map shapes, directions and full input/weight force-loss gradients; not full model",
              "protocol": {"rounds": args.rounds, "iterations": args.iterations, "force_loss_iterations": 3,
                           "warmup": 5, "order": "alternating v4/v5 each paired round",
                           "guard": "NVML exclusive and own three-source checks before/after every round",
                           "excluded": "compile, validation, warmup, GPU/source guards"}}, "results": []}
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        for name in args.maps:
            code = metadata(name)
            for nodes in args.nodes:
                torch.manual_seed(20261012)
                plans = [baseline_plan(code, dtype, torch.device("cuda:0")), _plan(code, dtype, torch.device("cuda:0"))]
                din, dout, numel = (896, 1344, 319488) if name == "up" else (384, 576, 221184)
                x = torch.randn(nodes, din, device="cuda", dtype=dtype)
                y = torch.randn(nodes, dout, device="cuda", dtype=dtype)
                w = torch.randn(1, 1, numel, device="cuda", dtype=dtype)
                ids = torch.zeros(nodes, device="cuda", dtype=torch.int64)
                for direction in ("dx", "dw", "forward", "force_loss"):
                    role = {"dx": 0, "dw": 1, "forward": 2}.get(direction)
                    if role is not None:
                        functions = [lambda p=p, role=role, ids=ids, x=x, w=w, y=y: p.contract(role, ids, x, w, y) for p in plans]
                    else:
                        leaves = [(x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()) for _ in plans]
                        functions = [lambda p=p, a=a, b=b, ids=ids: force_loss(p, a, b, ids)
                                     for p, (a, b) in zip(plans, leaves, strict=True)]
                    actual, expected = (fn() for fn in functions)
                    values = list(zip(actual, expected, strict=True)) if role is None else [(actual, expected)]
                    precision = [errors(a, b) for a, b in values]
                    atol, rtol = (3e-5, 5e-5) if dtype == torch.float32 else (2e-11, 2e-10)
                    for a, b in values:
                        torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
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
                    row = {"dtype": dtype_name, "map": name, "nodes": nodes, "din": din, "dout": dout, "numel": numel,
                           "direction": direction, "precision": precision, "samples": {"v4": samples[0], "v5": samples[1]},
                           "wall_ms": {label: stats([z["wall_ms"] for z in sample]) for label, sample in zip(("v4", "v5"), samples, strict=True)},
                           "cuda_event_ms": {label: stats([z["cuda_event_ms"] for z in sample]) for label, sample in zip(("v4", "v5"), samples, strict=True)},
                           "paired_wall_speedup": stats([a["wall_ms"] / b["wall_ms"] for a, b in zip(*samples, strict=True)]),
                           "paired_event_speedup": stats([a["cuda_event_ms"] / b["cuda_event_ms"] for a, b in zip(*samples, strict=True)])}
                    report["results"].append(row)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(dtype_name, name, nodes, direction, {k: v["median"] for k, v in row["wall_ms"].items()}, row["paired_wall_speedup"], flush=True)


if __name__ == "__main__":
    main()
