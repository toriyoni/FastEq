"""Replay captured EQv3 MergeLayerNorm inputs/weights, not an end-to-end model run.

Each fixture is a weights_only-loadable dictionary with name, input, output,
state_dict and constructor kwargs. See --fixtures for the capture directory.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import socket

import torch
import triton

from benchmark_triton_equivariant_layer_norm import compare, load_source, measure
from fasteq.triton.fused_equivariant_layer_norm import from_reference


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--samples", type=int, default=20)
    args = parser.parse_args()
    source_path = Path(os.environ["FASTEQ_EQUIFORMER_V3_LAYER_NORM"])
    source_class = load_source(source_path, "EquivariantMergeLayerNorm")
    capture = json.loads((args.fixtures / "provenance.json").read_text())
    assert digest(source_path) == capture["layernorm_sha256"]
    implementation = Path(__file__).resolve().parents[1] / "fasteq/triton/fused_equivariant_layer_norm.py"
    report = {"scope": "Captured model activations and EMA weights; standalone layer replay",
              "host": socket.gethostname(), "device": torch.cuda.get_device_name(),
              "torch": torch.__version__, "triton": triton.__version__,
              "target": str(triton.runtime.driver.active.get_current_target()),
              "source_sha256": digest(source_path), "implementation_sha256": digest(implementation),
              "capture_provenance": capture,
              "warmup": args.warmup, "samples": args.samples,
              "atol": 5e-5, "rtol": 5e-4, "cases": []}
    torch.set_num_threads(8)
    for structure in ("cu108", "mixed280"):
        for layer in ("blocks_0_norm_1", "blocks_0_norm_2", "norm"):
            path = args.fixtures / structure / (layer + ".pt")
            data = torch.load(path, map_location="cpu", weights_only=True)
            source = source_class(**data["kwargs"]).cuda().eval()
            source.load_state_dict(data["state_dict"], strict=True)
            adapter = from_reference(source).eval()
            x = data["input"].cuda()
            torch.manual_seed(20260920)
            upstream = torch.randn_like(x)
            named = dict(source.named_parameters())
            for mode in ("fwd", "fwd_bwd"):
                x.requires_grad_(mode == "fwd_bwd")

                def invoke(op):
                    if mode == "fwd":
                        with torch.inference_mode():
                            return (op(x),)
                    output = op(x)
                    return (output, *torch.autograd.grad(output, (x, *named.values()), upstream))

                functions = {"torch": lambda: invoke(source), "fused": lambda: invoke(adapter)}
                expected, actual = functions["torch"](), functions["fused"]()
                names = ("output",) if mode == "fwd" else ("output", "dX", *named)
                checks = {key: compare(a, b) for key, a, b in zip(names, actual, expected)}
                checks["torch_vs_captured_output"] = compare(expected[0], data["output"].cuda())
                checks["fused_vs_captured_output"] = compare(actual[0], data["output"].cuda())
                row = {"structure": structure, "layer": data["name"], "mode": mode,
                       "shape": list(x.shape), "stride": list(x.stride()),
                       "fixture_sha256": digest(path), "kwargs": data["kwargs"],
                       "checks": checks, "pass": all(m["failures"] == 0 for m in checks.values())}
                del expected, actual
                if row["pass"]:
                    row.update(measure(functions, args))
                report["cases"].append(row)
                args.out.parent.mkdir(parents=True, exist_ok=True)
                args.out.write_text(json.dumps(report, indent=2) + "\n")
                print(structure, layer, mode, "PASS" if row["pass"] else checks, flush=True)
    assert all(row["pass"] for row in report["cases"]), "Native/captured precision failure; inspect JSON"


if __name__ == "__main__":
    main()
