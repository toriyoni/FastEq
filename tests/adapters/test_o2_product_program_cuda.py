"""O(2) channel-product CUDA programs preserve broadcasts and training caches."""

import pytest
import torch
from eqx.o2 import TensorProduct
from test_equivariant_cuda import check

from fasteq.adapters import adapt_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def module(weighted=True, internal=True, shared=True):
    return TensorProduct(
        "4x1me", "4x1me", "4x0ee+4x0oe+4x2me",
        [(0, 0, i, "uuu", weighted) for i in range(3)],
        internal_weights=internal, shared_weights=shared,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "leading_shapes",
    [
        ((5,), (5,), ()),
        ((1,), (1,), (5,)),
        ((2, 1), (1, 3), (2, 3)),
        ((2, 3), (), (1,)),
        ((), (), ()),
    ],
)
def test_o2_channel_product_external_broadcast(dtype, leading_shapes):
    m = module(internal=False, shared=False).to(device="cuda", dtype=dtype)
    args = [
        torch.randn(*shape, size, device="cuda", dtype=dtype)
        for shape, size in zip(leading_shapes, (8, 8, m.weight_numel))
    ]
    check(
        m, args,
        atol=3e-5 if dtype == torch.float32 else 2e-9,
        rtol=3e-5 if dtype == torch.float32 else 2e-8,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("weighted", [True, False])
def test_o2_internal_and_unweighted(dtype, weighted):
    m = module(weighted=weighted).to(device="cuda", dtype=dtype)
    x = torch.randn(3, 8, device="cuda", dtype=dtype)
    check(m, [x, torch.randn_like(x)],
          atol=3e-5 if dtype == torch.float32 else 2e-9,
          rtol=3e-5 if dtype == torch.float32 else 2e-8)


@pytest.mark.parametrize("weighted", [True, False])
def test_o2_empty_edge_gradient(weighted):
    reference = module(weighted=weighted).cuda().double()
    native = adapt_model(reference)
    results = []
    for m in (reference, native):
        inputs = [torch.empty(0, 8, device="cuda", dtype=torch.float64, requires_grad=True) for _ in range(2)]
        out = m(*inputs)
        leaves = inputs + list(m.parameters())
        gradients = torch.autograd.grad(out.sum(), leaves, allow_unused=True)
        results.append((out, gradients))
    torch.testing.assert_close(results[0][0], results[1][0])
    for ref, actual in zip(results[0][1], results[1][1]):
        if ref is None or actual is None:
            remaining = actual if ref is None else ref
            assert remaining is None or torch.count_nonzero(remaining) == 0
        else:
            torch.testing.assert_close(actual, ref)


def test_o2_inference_then_training_and_dtype_change():
    reference = module().cuda().double()
    native = adapt_model(reference)
    x, y = [torch.randn(5, 8, device="cuda", dtype=torch.float64) for _ in range(2)]
    with torch.inference_mode():
        torch.testing.assert_close(native(x, y), reference(x, y))
    for m in (reference, native):
        m.train()
        m.zero_grad(set_to_none=True)
        inputs = [value.detach().clone().requires_grad_() for value in (x, y)]
        out = m(*inputs)
        first = torch.autograd.grad(out.square().sum(), inputs, create_graph=True)
        sum(g.square().sum() for g in first).backward()
    for expected, actual in zip(reference.parameters(), native.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad, atol=2e-8, rtol=2e-7)
    reference.float()
    native.float()
    torch.testing.assert_close(native(x.float(), y.float()), reference(x.float(), y.float()),
                               atol=3e-5, rtol=3e-5)
