"""Cached constants stay valid across inference, training and model updates."""

from copy import deepcopy

import pytest
import torch
from e3nn import o3

from fasteq.adapters import adapt_model
from fasteq.adapters.cache import TensorCache
from fasteq.adapters.tensor_product import TensorProductPlan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def test_constant_cache_is_bounded_and_trainable():
    cache = TensorCache(capacity=2, max_numel=8)
    with torch.inference_mode():
        first = cache.get(1, lambda: torch.ones(4, device="cuda"))
    assert not first.is_inference()
    x = torch.randn_like(first, requires_grad=True)
    torch.testing.assert_close(torch.autograd.grad((x * first).sum(), x)[0], first)
    assert cache.get(1, lambda: None) is first
    cache.get(2, lambda: torch.ones(2, device="cuda"))
    cache.get(3, lambda: torch.ones(3, device="cuda"))
    assert list(cache.values) == [2, 3]
    cache.get(4, lambda: torch.ones(9, device="cuda"))
    assert list(cache.values) == [2, 3]


@pytest.mark.parametrize("tace", [True, False])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_norm_cache_model_updates(tace, dtype):
    from prophet.layer_norm import RMSLayerNorm
    from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm

    reference = (
        (
            EquivariantMergeLayerNorm([0, 1, 2], 4)
            if tace
            else RMSLayerNorm(o3.Irreps("4x0e+4x1o+4x2e"))
        )
        .cuda()
        .to(dtype)
    )
    model = adapt_model(reference)
    shape = (5, 9, 4) if tace else (5, 36)
    x = torch.randn(shape, device="cuda", dtype=dtype)
    with torch.inference_mode():
        model(x)
    # Warmed constants can be saved by autograd; affine values must be fresh.
    for native, original in zip(model.parameters(), reference.parameters()):
        with torch.no_grad():
            original.uniform_(0.2, 1.2)
            native.copy_(original)
    if tace:
        with torch.no_grad():
            reference.balance_degree_weight.mul_(1.3)
        model.load_state_dict(reference.state_dict())
    for current in (model, deepcopy(model)):
        a, b = [x.clone().requires_grad_() for _ in range(2)]
        expected, actual = reference(a), current(b)
        tol = 5e-5 if dtype == torch.float32 else 2e-8
        torch.testing.assert_close(actual, expected, atol=tol, rtol=tol)
        leaves = (a, *reference.parameters()), (b, *current.parameters())
        gradients = [
            torch.autograd.grad(y.square().sum(), ps, create_graph=True)
            for y, ps in zip((expected, actual), leaves)
        ]
        for ref, got in zip(*gradients):
            torch.testing.assert_close(got, ref, atol=tol * 10, rtol=tol * 10)
        hessians = [
            torch.autograd.grad(sum(g.square().sum() for g in gs), ps)
            for gs, ps in zip(gradients, leaves)
        ]
        for ref, got in zip(*hessians):
            torch.testing.assert_close(got, ref, atol=tol * 50, rtol=tol * 50)
    model.to(torch.float64 if dtype == torch.float32 else torch.float32)
    assert (
        model(x.to(next(model.parameters()).dtype)).dtype
        == next(model.parameters()).dtype
    )


def test_layout_permutation_and_identity():
    module = o3.TensorProduct(
        "4x0e+4x1o+4x2e",
        "0e+1o+2e",
        "4x0e+4x1o+4x2e",
        [(0, 0, 0, "uvu", True)],
        internal_weights=False,
        shared_weights=False,
    )
    plan = TensorProductPlan(module)
    for which, irreps in enumerate(
        (module.irreps_in1, module.irreps_in2, module.irreps_out)
    ):
        x = torch.arange(3 * irreps.dim, device="cuda").reshape(3, irreps.dim)
        expected = torch.cat(
            [
                x[:, sl].reshape(3, c, ir.dim).transpose(1, 2).flatten(1)
                for (c, ir), sl in zip(irreps, irreps.slices())
            ],
            -1,
        )
        actual = plan._layout(x, which)
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(plan._layout(actual, which, True), x)
        if which == 1:
            assert actual is x


@pytest.mark.parametrize("weighted", [False, True])
def test_tp_inference_constants_then_train(weighted):
    reference = (
        o3.TensorProduct(
            "4x0e+4x1o",
            "0e+1o",
            "4x0e+4x1o",
            [(0, 0, 0, "uvu", weighted), (1, 0, 1, "uvu", weighted)],
            internal_weights=False,
            shared_weights=False,
        )
        .cuda()
        .double()
    )
    model = adapt_model(reference)
    xs = [
        torch.randn(3, n, device="cuda", dtype=torch.float64)
        for n in (16, 4, reference.weight_numel)
    ]
    with torch.inference_mode():
        model(*xs)
    results = []
    for m in (reference, model):
        inputs = [x.clone().requires_grad_() for x in xs]
        out = m(*inputs)
        gradients = torch.autograd.grad(
            out.square().sum(), inputs, create_graph=True, allow_unused=True
        )
        curvature = torch.autograd.grad(
            sum(g.square().sum() for g in gradients if g is not None),
            inputs,
            allow_unused=True,
        )
        results.append((out, gradients, curvature))
    for ref_group, got_group in zip(*results):
        if torch.is_tensor(ref_group):
            torch.testing.assert_close(got_group, ref_group, atol=2e-8, rtol=2e-7)
        else:
            for ref, got in zip(ref_group, got_group):
                if ref is None:
                    assert got is None or got.numel() == 0
                else:
                    torch.testing.assert_close(got, ref, atol=2e-8, rtol=2e-7)
