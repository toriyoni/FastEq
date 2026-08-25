#!/usr/bin/env python3
"""Isolate cuEq-vs-FastEq STC forward, first derivative, and weight double backward."""

import sys

import torch

sys.path.insert(0, "test")

import mace_uniform1d_training_compare as compare


def main():
    dtype_name = sys.argv[1] if len(sys.argv) > 1 else "float32"
    dtype = getattr(torch, dtype_name)
    calculators = {
        backend: compare.make_calculator(
            "benchmark_results/models/MACE-MP-0_small.model", backend, dtype_name
        )
        for backend in ("cueq", "fasteq_cwtp_stc")
    }
    for product_index in (0, 1):
        module = calculators["fasteq_cwtp_stc"].models[0].products[
            product_index
        ].symmetric_contractions.cuda()

        torch.manual_seed(20260825 + product_index)
        batch = 24
        base_x = torch.randn(
            batch, module.irreps_in.dim, device="cuda", dtype=dtype
        )
        # Match a water-like batch: many atoms reuse the same O/H weight rows.
        # This exercises the cross-block gather-reduction in STC double backward.
        indices = torch.arange(batch, device="cuda", dtype=torch.long) % 2

        results = {}
        for use_fasteq in (False, True):
            module.use_fasteq = use_fasteq
            module.zero_grad(set_to_none=True)
            x = base_x.detach().clone().requires_grad_(True)
            output = module(x, indices)
            grad_x = torch.autograd.grad(output.square().mean(), x, create_graph=True)[0]
            grad_x.square().mean().backward()
            torch.cuda.synchronize()
            results[use_fasteq] = (
                output.detach(),
                grad_x.detach(),
                x.grad.detach().clone(),
                module.weight.grad.detach().clone(),
            )

        print(f"product={product_index}")
        for label, tensor_index in (
            ("output", 0),
            ("grad_x", 1),
            ("input_double_grad", 2),
            ("weight_grad", 3),
        ):
            actual = results[True][tensor_index]
            reference = results[False][tensor_index]
            difference = (actual - reference).abs()
            print(
                label,
                f"shape={tuple(actual.shape)}",
                f"max_abs={difference.max().item():.9g}",
                f"mean_abs={difference.mean().item():.9g}",
                f"bad_1e-4={int((difference > 1.0e-4).sum())}",
                f"reference_max={reference.abs().max().item():.9g}",
            )

        internal_x = module.transpose_in(base_x).detach()
        projected_weight = torch.einsum(
            "zau,ab->zbu", module.weight.detach(), module.projection
        ).flatten(1)
        projected_results = {}
        for use_fasteq in (False, True):
            x = internal_x.clone().requires_grad_(True)
            weight = projected_weight.clone().requires_grad_(True)
            operator = module.ff if use_fasteq else module.f
            output = operator([weight, x], input_indices={0: indices})[0]
            grad_x = torch.autograd.grad(output.square().mean(), x, create_graph=True)[0]
            grad_x.square().mean().backward()
            projected_results[use_fasteq] = weight.grad.detach().clone()

        actual = projected_results[True]
        reference = projected_results[False]
        difference = (actual - reference).abs()
        flat_index = int(difference.argmax())
        row = flat_index // difference.shape[1]
        column = flat_index % difference.shape[1]
        print(
            "projected_weight_grad",
            f"shape={tuple(actual.shape)}",
            f"max_abs={difference.max().item():.9g}",
            f"mean_abs={difference.mean().item():.9g}",
            f"bad_1e-4={int((difference > 1.0e-4).sum())}",
            f"argmax=({row},{column})",
            f"segment={column // module.ff.u}",
            f"channel={column % module.ff.u}",
            f"reference={reference[row, column].item():.9g}",
            f"actual={actual[row, column].item():.9g}",
        )

        direct_results = {}
        output_width = module.irreps_out.dim
        fixed_grad_out = torch.randn(
            batch, output_width, device="cuda", dtype=base_x.dtype
        )
        fixed_grad_grad_x = torch.randn_like(internal_x)
        for use_fasteq in (False, True):
            x = internal_x.clone().requires_grad_(True)
            weight = projected_weight.clone().requires_grad_(True)
            grad_out = fixed_grad_out.clone().requires_grad_(True)
            operator = module.ff if use_fasteq else module.f
            output = operator([weight, x], input_indices={0: indices})[0]
            grad_x = torch.autograd.grad(
                output, x, grad_outputs=grad_out, create_graph=True
            )[0]
            direct_results[use_fasteq] = tuple(
                tensor.detach()
                for tensor in torch.autograd.grad(
                    grad_x,
                    (grad_out, x, weight),
                    grad_outputs=fixed_grad_grad_x,
                )
            )

        for label, tensor_index in (
            ("direct_d_grad_out", 0),
            ("direct_d_x", 1),
            ("direct_d_weight", 2),
        ):
            actual = direct_results[True][tensor_index]
            reference = direct_results[False][tensor_index]
            difference = (actual - reference).abs()
            print(
                label,
                f"shape={tuple(actual.shape)}",
                f"max_abs={difference.max().item():.9g}",
                f"mean_abs={difference.mean().item():.9g}",
                f"bad_1e-4={int((difference > 1.0e-4).sum())}",
                f"reference_max={reference.abs().max().item():.9g}",
            )


if __name__ == "__main__":
    main()
