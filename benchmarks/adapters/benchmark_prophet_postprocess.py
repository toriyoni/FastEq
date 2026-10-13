"""Isolate the equivariant residual/Gate/RMS chain with paired timing."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import torch
from benchmark_models import assert_exclusive_gpu, one_batch, stats
from e3nn import o3
from prophet.layer_norm import RMSLayerNorm
from prophet.model import Gate

from fasteq.adapters.fusion import ProphetPostprocess
from fasteq.adapters.programs import GatePlan, NormPlan


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--nodes", nargs="+", type=int, default=[64, 256])
    parser.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    args = parser.parse_args()
    torch.set_num_threads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    report = {"metadata": {"comparison": "native CUDA Gate + native CUDA RMS vs fused add/Gate/RMS", "channels": args.channels,
                            "warmup": 5, "rounds": 7, "iterations": 10}, "results": []}
    c = args.channels
    for dtype_name in args.dtypes:
        dtype = getattr(torch, dtype_name)
        ir = o3.Irreps(f"{3*c}x0e+{c}x1o+{c}x2e")
        out = o3.Irreps(f"{c}x0e+{c}x1o+{c}x2e")
        gate = Gate(ir).cuda().to(dtype)
        norm = RMSLayerNorm(out, centering=False).cuda().to(dtype)
        old_gate, old_norm = GatePlan(gate, "prophet"), NormPlan(norm)
        new = ProphetPostprocess(SimpleNamespace(linear_2=SimpleNamespace(irreps_out=ir), gate=gate, layer_norm=None), norm)
        for nodes in args.nodes:
            torch.manual_seed(2701)
            x = torch.randn(nodes, ir.dim, device="cuda", dtype=dtype, requires_grad=True)
            skip = torch.randn_like(x, requires_grad=True)
            seed = torch.randn(nodes, out.dim, device="cuda", dtype=dtype)
            functions = [
                lambda old_norm=old_norm, norm=norm, old_gate=old_gate, x=x, skip=skip: old_norm(norm, old_gate(x + skip)),
                lambda new=new, x=x, skip=skip: new(x, skip),
            ]
            params = [*norm.parameters(), x, skip]
            y0, y1 = [fn() for fn in functions]
            torch.testing.assert_close(y0, y1, atol=2e-6 if dtype == torch.float32 else 1e-12, rtol=2e-5 if dtype == torch.float32 else 1e-10)
            for mode in ("energy", "backward", "force_loss"):
                def execute(fn, *, params=params, mode=mode, seed=seed, x=x):
                    for p in params:
                        p.grad = None
                    with torch.set_grad_enabled(mode != "energy"):
                        y = fn()
                        if mode == "backward":
                            y.square().mean().backward()
                        elif mode == "force_loss":
                            force = torch.autograd.grad((y * seed).sum(), x, create_graph=True)[0]
                            (y.square().mean() + force.square().mean()).backward()
                runners = [lambda fn=fn, execute=execute: execute(fn) for fn in functions]
                for run in runners:
                    for _ in range(5):
                        run()
                torch.cuda.synchronize()
                samples = [[], []]
                for block in range(7):
                    assert_exclusive_gpu()
                    for i in (0, 1) if block % 2 == 0 else (1, 0):
                        samples[i].append(one_batch(runners[i], 10))
                    assert_exclusive_gpu()
                metrics = [{"wall_ms": stats([s["wall_ms"] for s in rows]), "samples": rows} for rows in samples]
                speedup = metrics[0]["wall_ms"]["median"] / metrics[1]["wall_ms"]["median"]
                report["results"].append({"dtype": dtype_name, "nodes": nodes, "mode": mode,
                                          "baseline": metrics[0], "fused": metrics[1], "wall_speedup": speedup})
                (args.output / "operators.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"{dtype_name} N{nodes} {mode}: {metrics[0]['wall_ms']['median']:.4f} -> {metrics[1]['wall_ms']['median']:.4f} ms, {speedup:.3f}x", flush=True)


if __name__ == "__main__":
    main()
