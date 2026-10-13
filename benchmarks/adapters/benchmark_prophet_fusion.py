"""Paired old/new equivariant post-processing with unchanged scalar networks."""

import argparse
import json
import os
from pathlib import Path

import torch
from benchmark_models import (
    assert_exclusive_gpu,
    benchmark_pair,
    build_data,
    build_model,
    configure,
    validate_pair,
)

from fasteq.adapters import adapt_model
from fasteq.adapters.fusion import install_prophet_fusions


def without_fusion(model):
    for m in model.modules():
        if hasattr(m, "_fasteq_postprocess"):
            del m._fasteq_postprocess
        if hasattr(m, "_fasteq_fused_norms"):
            del m._fasteq_fused_norms
    return model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--channels", type=int, default=64)
    p.add_argument("--nodes", nargs="+", type=int, default=[64, 256])
    p.add_argument("--dtypes", nargs="+", default=["float32", "float64"])
    p.add_argument("--modes", nargs="+", default=["energy", "forces", "force_loss"])
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--rounds", type=int, default=5)
    p.add_argument("--iterations", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    a.output.mkdir(parents=True, exist_ok=True)
    report = {"metadata": {"gpu": torch.cuda.get_device_name(),
                            "visible_device": os.environ.get("CUDA_VISIBLE_DEVICES"),
                            "channels": a.channels, "comparison": "same adapted model without vs with residual+Gate+RMS fusion",
                            "protocol": {"warmup": a.warmup, "rounds": a.rounds, "iterations": a.iterations}},
              "results": []}
    for dtype_name in a.dtypes:
        dtype = getattr(torch, dtype_name)
        original = build_model("prophet", a.channels, dtype, 2)
        old = without_fusion(adapt_model(original))
        new = without_fusion(adapt_model(original))
        ids = [id(parameter) for parameter in new.parameters()]
        keys = tuple(new.state_dict())
        fusions = install_prophet_fusions(new)
        assert ids == [id(parameter) for parameter in new.parameters()]
        assert keys == tuple(new.state_dict())
        for nodes in a.nodes:
            data = build_data(nodes, 12, dtype)
            for mode in a.modes:
                pair = [old, new]
                for model in pair:
                    configure(model, "prophet", mode)
                print(f"START {dtype_name} C={a.channels} N={nodes} {mode}", flush=True)
                assert_exclusive_gpu()
                validation = validate_pair(pair, "prophet", data, mode, dtype)
                timing = benchmark_pair(pair, "prophet", data, mode, a)
                row = {"dtype": dtype_name, "nodes": nodes, "mode": mode,
                       "fusions": fusions, "validation": validation, **timing}
                report["results"].append(row)
                (a.output / "models.json").write_text(json.dumps(report, indent=2) + "\n")
                print(f"DONE {dtype_name} N={nodes} {mode}: {timing['baseline']['wall_ms']['median']:.4f} -> {timing['adapted']['wall_ms']['median']:.4f} ms ({timing['wall_speedup']:.4f}x), {validation}", flush=True)


if __name__ == "__main__":
    main()
