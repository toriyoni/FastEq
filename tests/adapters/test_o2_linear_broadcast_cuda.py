"""O(2) linear broadcasting must include external weights and biases."""

import pytest
import torch
from eqx.o2 import Linear
from test_equivariant_cuda import check

from fasteq.adapters import adapt_model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("shared_weights", [True, False])
@pytest.mark.parametrize(
    "leading_shapes",
    [
        ((1,), (1,), (5,)),  # Bias alone introduces the batch dimension.
        ((2, 1), (1, 3), (2, 3)),  # All three operands participate.
        ((2, 3), (), ()),  # Shared parameters stay compact.
        ((), (3,), ()),  # External batched weights override the module hint.
    ],
)
def test_o2_external_broadcast(shared_weights, leading_shapes):
    torch.manual_seed(407)
    m = Linear(
        "2x0ee+2x1me",
        "3x0ee+3x1me",
        internal_weights=False,
        shared_weights=shared_weights,
        biases=True,
    ).double()
    args = [
        torch.randn(*shape, size, device="cuda", dtype=torch.float64)
        for shape, size in zip(
            leading_shapes, (m.irreps_in.dim, m.weight_numel, m.bias_numel)
        )
    ]
    check(m, args)


def test_o2_broadcast_fp32():
    m = Linear(
        "2x0ee", "3x0ee", internal_weights=False, shared_weights=True, biases=True
    )
    args = [
        torch.randn(2, 1, 2, device="cuda"),
        torch.randn(1, 3, m.weight_numel, device="cuda"),
        torch.randn(2, 3, m.bias_numel, device="cuda"),
    ]
    check(m, args, atol=3e-5, rtol=3e-5)


def test_o2_external_bias_required():
    m = Linear(
        "2x0ee", "3x0ee", internal_weights=False, shared_weights=False, biases=True
    ).cuda()
    x = torch.randn(2, 2, device="cuda")
    w = torch.randn(2, m.weight_numel, device="cuda")
    for candidate in (m, adapt_model(m)):
        with pytest.raises(RuntimeError, match="Biases must be provided"):
            candidate(x, w)
