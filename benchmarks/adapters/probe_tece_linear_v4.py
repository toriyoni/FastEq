"""Read-only, warmed real TECE Linear plan for one NVTX-selected ncu call.

Normal timing and Torch precision checks precede the target range. Kernel
counter collection must select the range, rather than profiling warmup.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from benchmark_models import assert_exclusive_gpu, build_model, one_batch, stats
from fasteq.adapters import adapt_model
from fasteq.ops.indexed_linear import _plan


def torch_reference(element, role, x, weight, y):
    result = torch.zeros_like(x if role == 0 else weight)
    for sx, sw, sy, ci, co, dim, coefficient in element.paths:
        features = x[:, sx : sx + ci * dim].reshape(x.shape[0], dim, ci)
        cotangent = y[:, sy : sy + co * dim].reshape(y.shape[0], dim, co)
        matrix = weight[0, 0, sw : sw + ci * co].reshape(ci, co)
        if role == 0:
            value = torch.matmul(cotangent, matrix.T).reshape(x.shape[0], -1)
            result[:, sx : sx + ci * dim] += coefficient * value
        else:
            value = torch.einsum("nmi,nmj->ij", features, cotangent)
            result[0, 0, sw : sw + ci * co] += coefficient * value.flatten()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(os.environ["TACE_PROPHET_ROOT"]))
    parser.add_argument("--role", type=int, choices=(0, 1), required=True)
    parser.add_argument("--module", choices=("linear_up", "linear_coefs"), default="linear_up")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int, default=160)
    parser.add_argument("--rounds", type=int, default=7)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    os.sched_setaffinity(0, {args.cpu})
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert_exclusive_gpu()
    frozen_path = args.root / "performance/fasteq_optimization/v4_snapshot/manifest.json"
    frozen = json.loads(frozen_path.read_text())["files_sha256"]

    def check_sources():
        for relative, expected in frozen.items():
            assert hashlib.sha256((args.root / relative).read_bytes()).hexdigest() == expected, relative

    check_sources()
    model = adapt_model(build_model("tece", 64, torch.float64, 2))
    module_path = "readout_fn.representation.interactions.1.rejector." + args.module
    module = model.get_submodule(module_path)
    element = module._fasteq_plan.element
    assert element.layout == "ir_mul" and element.num_experts == 1
    expected = (896, 1344, 319488) if args.module == "linear_up" else (384, 576, 221184)
    assert (element.input_dim, element.output_dim, element.weight_numel) == expected
    plan = _plan(element.metadata, torch.float64, torch.device("cuda:0"))
    torch.manual_seed(20261009)
    nodes = 3072
    x = torch.randn(nodes, element.input_dim, device="cuda", dtype=torch.float64)
    y = torch.randn(nodes, element.output_dim, device="cuda", dtype=torch.float64)
    weight = torch.randn(1, 1, element.weight_numel, device="cuda", dtype=torch.float64)
    types = torch.zeros(nodes, device="cuda", dtype=torch.int64)

    def call():
        return plan.contract(args.role, types, x, weight, y)

    with torch.no_grad():
        reference, output = torch_reference(element, args.role, x, weight, y), call()
        torch.testing.assert_close(output, reference, atol=2e-8, rtol=2e-7)
        precision = {
            "max_absolute_error": (output - reference).abs().max().item(),
            "relative_l2_error": ((output - reference).square().sum() / reference.square().sum()).sqrt().item(),
            "reference": "Torch per-irrep matmul/einsum with FP64 path coefficients; TF32 disabled",
        }
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        samples = []
        for _ in range(args.rounds):
            assert_exclusive_gpu()
            samples.append(one_batch(call, args.iterations))
            assert_exclusive_gpu()
        report = {
            "module": module_path, "native_plan": type(plan).__name__,
            "serialized_plan": element.metadata, "paths": element.paths,
            "nodes": nodes, "input_dim": element.input_dim, "output_dim": element.output_dim,
            "weight_numel": element.weight_numel, "weight_shape": list(weight.shape),
            "elements": 1, "experts": 1, "layout": element.layout, "role": args.role,
            "actual_branch": "ROLE0 scalar shared transpose" if args.role == 0 else "ROLE1 scalar chunk partials + ROLE3 reduce",
            "kernel_name": "run", "dtype": "float64", "seed": 20261009,
            "gpu_uuid": str(torch.cuda.get_device_properties(0).uuid),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "cpu_affinity": sorted(os.sched_getaffinity(0)),
            "frozen_manifest": str(frozen_path), "frozen_file_count": len(frozen), "source_sha256": frozen,
            "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "precision": precision,
            "normal_timing": {"wall_ms": stats([s["wall_ms"] for s in samples]),
                              "cuda_event_ms": stats([s["cuda_event_ms"] for s in samples]), "samples": samples},
            "timing_note": "Warm normal timing outside target NVTX, not ncu replay timing or model speedup.",
            "capture_note": "One contract call; role1 launches a partial kernel then a reducer. ncu launch-count1 selects only the partial kernel.",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        check_sources()
        assert_exclusive_gpu()
        torch.cuda.synchronize()
        marker = "target::" + ("up" if args.module == "linear_up" else "coefs") + "::" + ("dx" if args.role == 0 else "dw")
        with torch.cuda.nvtx.range(marker):
            output = call()
            torch.cuda.synchronize()
        assert_exclusive_gpu()
        check_sources()
        report["capture_completed"] = True
        report["target_nvtx_range"] = marker
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("PROBE_DONE", args.module, args.role, flush=True)


if __name__ == "__main__":
    main()
