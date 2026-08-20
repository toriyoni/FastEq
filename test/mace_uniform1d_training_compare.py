#!/usr/bin/env python3
"""Compare a reference backend and FastEq-Uniform1D MACE training numerics."""

import argparse
import json
from pathlib import Path

import torch

from mace.calculators import MACECalculator
from mace_bench_new import build_water_box, prepare_model_input


def make_calculator(model_path, backend, dtype):
    calculator = MACECalculator(
        model_paths=str(model_path),
        device="cuda",
        default_dtype=dtype,
        compile_mode=None,
        enable_cueq=backend != "e3nn",
        cueq_optimize_all=True,
        cueq_optimize_channelwise=False,
        cueq_use_fasteq=False,
        cueq_fasteq_uniform1d_only=backend == "fasteq_uniform1d",
        enable_oeq=False,
    )
    for model in calculator.models:
        for parameter in model.parameters():
            parameter.requires_grad_(True)
    return calculator


def run_force_loss(model, batch):
    model.train()
    model.zero_grad(set_to_none=True)
    output = model(
        batch,
        training=True,
        compute_force=True,
        compute_virials=False,
        compute_stress=False,
    )
    forces = output["forces"]
    loss = forces.square().mean()
    loss.backward()
    torch.cuda.synchronize()
    gradients = {
        name: parameter.grad.detach().clone()
        for name, parameter in model.named_parameters()
        if parameter.grad is not None
    }
    return {
        "energy": output["energy"].detach().clone(),
        "forces": forces.detach().clone(),
        "loss": loss.detach().clone(),
        "gradients": gradients,
    }


def error_stats(reference, actual):
    difference = (actual - reference).abs()
    scale = reference.abs().max().item()
    return {
        "max_abs": difference.max().item(),
        "max_rel_to_reference_max": difference.max().item() / max(scale, 1.0e-30),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--box-side", type=int, default=2)
    parser.add_argument(
        "--reference-backend", choices=("e3nn", "cueq"), default="cueq"
    )
    parser.add_argument("--atol", type=float, default=5.0e-5)
    parser.add_argument("--rtol", type=float, default=5.0e-4)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.manual_seed(0)
    atoms = build_water_box(args.box_side, seed=1234 + args.box_side)
    calculators = {
        backend: make_calculator(args.model, backend, args.dtype)
        for backend in (args.reference_backend, "fasteq_uniform1d")
    }
    runs = {}
    edges = None
    for backend, calculator in calculators.items():
        model, batch, backend_edges = prepare_model_input(atoms, calculator)
        if edges is None:
            edges = backend_edges
        elif edges != backend_edges:
            raise RuntimeError(f"edge-count mismatch: {edges} vs {backend_edges}")
        runs[backend] = run_force_loss(model, batch)

    reference = runs[args.reference_backend]
    actual = runs["fasteq_uniform1d"]
    reference_gradients = reference["gradients"]
    actual_gradients = actual["gradients"]
    missing_gradients = sorted(reference_gradients.keys() - actual_gradients.keys())
    extra_gradients = sorted(actual_gradients.keys() - reference_gradients.keys())
    common_gradients = sorted(reference_gradients.keys() & actual_gradients.keys())
    if missing_gradients or extra_gradients:
        raise RuntimeError(
            "parameter-gradient coverage mismatch: "
            f"missing={missing_gradients}, extra={extra_gradients}"
        )
    reference_conv_gradients = {
        name for name in reference_gradients if "conv_tp_weights" in name
    }
    actual_conv_gradients = {
        name for name in actual_gradients if "conv_tp_weights" in name
    }
    if reference_conv_gradients != actual_conv_gradients:
        raise RuntimeError(
            "Uniform1D driver-gradient coverage mismatch: "
            f"{args.reference_backend}={sorted(reference_conv_gradients)}, "
            f"fasteq={sorted(actual_conv_gradients)}"
        )

    tensor_stats = {
        name: error_stats(reference[name], actual[name])
        for name in ("energy", "forces", "loss")
    }
    gradient_stats = {
        name: error_stats(reference_gradients[name], actual_gradients[name])
        for name in common_gradients
    }
    worst_gradient = max(
        gradient_stats,
        key=lambda name: gradient_stats[name]["max_abs"],
    )

    torch.testing.assert_close(
        actual["energy"], reference["energy"], atol=args.atol, rtol=args.rtol
    )
    torch.testing.assert_close(
        actual["forces"], reference["forces"], atol=args.atol, rtol=args.rtol
    )
    for name in common_gradients:
        torch.testing.assert_close(
            actual_gradients[name],
            reference_gradients[name],
            atol=args.atol,
            rtol=args.rtol,
            msg=lambda message, parameter_name=name: f"{parameter_name}: {message}",
        )

    result = {
        "status": "pass",
        "reference_backend": args.reference_backend,
        "dtype": args.dtype,
        "box_side": args.box_side,
        "num_atoms": len(atoms),
        "num_edges": edges,
        "atol": args.atol,
        "rtol": args.rtol,
        "reference_parameter_tensors_with_grad": len(reference_gradients),
        "reference_parameter_numel_with_grad": sum(
            gradient.numel() for gradient in reference_gradients.values()
        ),
        "fasteq_parameter_tensors_with_grad": len(actual_gradients),
        "fasteq_parameter_numel_with_grad": sum(
            gradient.numel() for gradient in actual_gradients.values()
        ),
        "common_parameter_tensors_compared": len(common_gradients),
        "unmatched_reference_gradient_names": missing_gradients,
        "unmatched_fasteq_gradient_names": extra_gradients,
        "tensor_errors": tensor_stats,
        "worst_gradient": worst_gradient,
        "worst_gradient_error": gradient_stats[worst_gradient],
        "conv_tp_weight_gradients": sorted(reference_conv_gradients),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
