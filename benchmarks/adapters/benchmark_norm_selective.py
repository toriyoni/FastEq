"""Guarded paired forces/force-loss timing against the verified v3 Norm snapshot."""

import argparse
import hashlib
import importlib.util
import json
import os
import statistics
import sys
from copy import deepcopy
from pathlib import Path

import torch
from benchmark_models import assert_exclusive_gpu
from benchmark_operators import compare, measure
from e3nn import o3

from fasteq.ops.equivariant_norm import NormPlan


def load_v3_norm(root):
    archive = root / "performance/fasteq_optimization/v3_snapshot"
    manifest = json.loads((archive / "manifest.json").read_text())
    relative = "src/FastEq/fasteq/ops/equivariant_norm.py"
    row = next(row for row in manifest["files"] if row["path"] == relative)
    path = archive / relative
    assert hashlib.sha256(path.read_bytes()).hexdigest() == row["sha256"]
    name = "fasteq.ops._v3_norm_paired"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.NormPlan, {"path": str(path), "sha256": row["sha256"]}


def operation(plan, module, x, cotangent, mode):
    y = plan(module, x)
    energy = (y * cotangent).mean() + 0.1 * y.square().mean()
    force = -torch.autograd.grad(energy, x, create_graph=mode == "force_loss")[0]
    if mode == "forces":
        return energy, force
    loss = force.square().mean()
    gradients = torch.autograd.grad(loss, (x, *module.parameters()), allow_unused=True)
    return energy, force, loss, *gradients


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--nodes", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--batches", type=int, default=9)
    parser.add_argument("--iterations", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    torch.set_num_threads(1)
    torch.manual_seed(17081)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert_exclusive_gpu()
    old_norm, provenance = load_v3_norm(args.root)
    files = [
        args.root / "src/FastEq/fasteq" / relative
        for relative in (
            "ops/equivariant_norm.py", "triton/fused_equivariant_layer_norm.py",
            "cuda/src/norm_plan.cpp", "cuda/src/jit_runtime.cpp", "cuda/jit.py",
        )
    ]

    def hashes():
        return {str(path.relative_to(args.root)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in files}

    frozen = hashes()
    metadata = {
        "gpu": torch.cuda.get_device_name(), "uuid": str(torch.cuda.get_device_properties(0).uuid),
        "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"), "v3_norm": provenance,
        "source_sha256": frozen, "channels": args.channels,
        "protocol": {"warmup": 5, "batches": args.batches, "iterations": args.iterations,
                     "order": "alternating within paired batches", "tf32": False},
        "guard": "NVML exclusive check before and after each paired batch; source checks per case",
        "scope": "Norm forward + grad(E,x); force-loss adds indirect dy chain and all parameter adjoints; no optimizer",
    }
    rows = []
    for dtype in (torch.float32, torch.float64):
        for kind in ("tace", "prophet"):
            if kind == "tace":
                from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm
                module = EquivariantMergeLayerNorm([0, 1, 2], args.channels).cuda().to(dtype)
            else:
                from prophet.layer_norm import RMSLayerNorm
                ir = o3.Irreps([(args.channels, (degree, (-1) ** degree)) for degree in range(3)])
                module = RMSLayerNorm(ir, centering=False).cuda().to(dtype)
            models = [deepcopy(module), deepcopy(module)]
            plans = [cls(m, tace=kind == "tace") for cls, m in zip((old_norm, NormPlan), models, strict=True)]
            for nodes in args.nodes:
                x = torch.randn(nodes, *plans[0].shape, device="cuda", dtype=dtype)
                inputs = [x.detach().clone().requires_grad_() for _ in models]
                cotangent = torch.randn_like(x)
                for mode in ("forces", "force_loss"):
                    assert hashes() == frozen
                    assert_exclusive_gpu()
                    functions = [
                        lambda p=p, m=m, x=x, c=cotangent, mode=mode: operation(p, m, x, c, mode)
                        for p, m, x in zip(plans, models, inputs, strict=True)
                    ]
                    error = compare(*[fn() for fn in functions], dtype)
                    for fn in functions:
                        for _ in range(5):
                            fn()
                    times, events = [[], []], [[], []]
                    for batch in range(args.batches):
                        assert_exclusive_gpu()
                        for i in (0, 1) if batch % 2 == 0 else (1, 0):
                            event, wall = measure(functions[i], args.iterations)
                            times[i].append(wall)
                            events[i].append(event)
                        assert_exclusive_gpu()
                    assert hashes() == frozen
                    ratios = [a / b for a, b in zip(times[0], times[1], strict=True)]
                    q25, _, q75 = statistics.quantiles(ratios, n=4, method="inclusive")
                    row = {
                        "kind": kind, "dtype": str(dtype), "nodes": nodes, "mode": mode,
                        "v3_wall_us": statistics.median(times[0]), "new_wall_us": statistics.median(times[1]),
                        "paired_speedup": statistics.median(ratios), "q25": q25, "q75": q75,
                        "wall_samples_us": times, "event_samples_us": events, "max_abs_error": error,
                    }
                    rows.append(row)
                    print(row, flush=True)
                    args.output.parent.mkdir(parents=True, exist_ok=True)
                    args.output.write_text(json.dumps({"metadata": metadata, "results": rows}, indent=2))


if __name__ == "__main__":
    main()
