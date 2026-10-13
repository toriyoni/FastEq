"""Native host plans preserve selective and repeated differentiation."""

import pytest
import torch
from e3nn import o3
from eqx.o3.linear import ElementLinear as TorchElementLinear
from eqx.o3.linear import MoEElementLinear as TorchMoEElementLinear

from fasteq.ops.indexed_linear import IndexedLinear as ElementLinear
from fasteq.ops.indexed_linear import MoEIndexedLinear as MoEElementLinear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("elements", [1, 3])
@pytest.mark.parametrize("nodes", [17, 129])
@pytest.mark.parametrize("selected", ["features", "weight", "both"])
def test_native_linear_selective_gradients(dtype, experts, elements, nodes, selected):
    torch.manual_seed(710)
    # The unmatched output irrep and unused input irrep leave holes that must
    # still be zero after removing initialization for fully covered maps.
    desc = o3.Linear("5x0e+4x1o+2x2e", "3x0e+6x1o+2x3o", internal_weights=False)
    native = ElementLinear(desc) if experts == 1 else MoEElementLinear(desc, experts)
    reference = (
        TorchElementLinear(desc, "torch")
        if experts == 1
        else TorchMoEElementLinear(desc, experts, "torch")
    )
    ids = torch.arange(nodes, device="cuda") % elements
    base = torch.randn(nodes, 2 * native.input_dim, device="cuda", dtype=dtype)
    x = base[:, ::2].detach().requires_grad_()
    w = torch.randn(
        elements, experts, desc.weight_numel, device="cuda", dtype=dtype
    ).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    wr = w.detach().clone().requires_grad_()
    values = [native(x, w, ids), reference(xr, wr, ids)]
    atol, rtol = (3e-5, 5e-5) if dtype == torch.float32 else (2e-11, 2e-10)
    torch.testing.assert_close(*values, atol=atol, rtol=rtol)
    requested = (
        [0] if selected == "features" else [1] if selected == "weight" else [0, 1]
    )
    first, second = [], []
    for y, inputs in zip(values, ([x, w], [xr, wr])):
        grads = torch.autograd.grad(
            y.square().mean(), [inputs[i] for i in requested], create_graph=True
        )
        first.append(grads)
        second.append(
            torch.autograd.grad(
                sum(g.square().mean() for g in grads), inputs, allow_unused=True
            )
        )
    for results in (first, second):
        for a, b in zip(*results):
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)


def test_native_linear_saved_variable_version():
    desc = o3.Linear("4x0e+4x1o", "4x0e+4x1o", internal_weights=False)
    native = ElementLinear(desc)
    x = torch.randn(3, desc.irreps_in.dim, device="cuda", requires_grad=True)
    w = torch.randn(1, desc.weight_numel, device="cuda", requires_grad=True)
    y = native(x, w, torch.zeros(3, device="cuda", dtype=torch.int64))
    with torch.no_grad():
        w.add_(1)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        torch.autograd.grad(y.sum(), x)


@pytest.mark.parametrize("nested_enable_grad", [False, True])
def test_native_linear_inference_mode(nested_enable_grad):
    desc = o3.Linear("3x0e+3x1o", "3x0e+3x1o", internal_weights=False)
    native = ElementLinear(desc)
    reference = TorchElementLinear(desc, "torch")
    x = torch.randn(4, desc.irreps_in.dim, device="cuda", requires_grad=True)
    w = torch.randn(1, desc.weight_numel, device="cuda", requires_grad=True)
    ids = torch.zeros(4, device="cuda", dtype=torch.int64)
    with torch.inference_mode(), torch.set_grad_enabled(nested_enable_grad):
        actual, expected = native(x, w, ids), reference(x, w, ids)
    torch.testing.assert_close(actual, expected)
    assert actual.is_inference() and expected.is_inference()
    assert not actual.requires_grad and not expected.requires_grad
    assert actual.grad_fn is None and expected.grad_fn is None


def test_native_forward_destination_has_no_edge():
    desc = o3.Linear("3x0e+3x1o", "3x0e+3x1o", internal_weights=False)
    native = ElementLinear(desc)
    x = torch.randn(4, desc.irreps_in.dim, device="cuda", requires_grad=True)
    w = torch.randn(1, desc.weight_numel, device="cuda", requires_grad=True)
    ids = torch.zeros(4, device="cuda", dtype=torch.int64)
    y = native(x, w, ids)
    assert y.grad_fn.name() == "FastEqIndexedLinearBackward"
    assert len(y.grad_fn.next_functions) == 2
    assert y.grad_fn.next_functions[0][0] is not None
    assert y.grad_fn.next_functions[1][0] is not None
    torch.autograd.grad(y.square().sum(), (x, w))
