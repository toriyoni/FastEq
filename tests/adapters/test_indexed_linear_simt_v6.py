"""FP32 shared-tile accuracy, including the physical force-loss adjoint."""

import pytest
import torch
from fasteq.ops.indexed_linear import _plan, indexed_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def opt_in_simt(monkeypatch):
    monkeypatch.setenv("FASTEQ_EXPERIMENTAL_LINEAR_SIMT", "1")
    _plan.cache_clear()
    yield
    _plan.cache_clear()


def reference(metadata, x, weight, ids):
    _din, dout, _numel, experts, layout, paths = metadata
    result = x.new_zeros(x.shape[0], dout)
    for sx, sw, sy, ci, co, dim, scale in paths:
        part = x[:, sx : sx + ci * dim * experts]
        part = (part.reshape(-1, dim, experts, ci) if layout == "ir_mul"
                else part.reshape(-1, experts, ci, dim).permute(0, 3, 1, 2))
        blocks = []
        for expert in range(experts):
            if weight.shape[0] == 1:
                matrix = weight[0, expert, sw : sw + ci * co].reshape(ci, co)
            else:
                matrix = weight[ids, expert, sw : sw + ci * co].reshape(-1, ci, co)
            blocks.append((part[:, :, expert] @ matrix) * scale)
        value = torch.stack(blocks, dim=2)
        if layout == "mul_ir":
            value = value.permute(0, 2, 3, 1)
        result[:, sy : sy + co * dim * experts] += value.flatten(1)
    return result


def chain(metadata, pos, projection, weight, ids, native):
    x = torch.sin(pos @ projection)
    output = (indexed_linear(repr(metadata), x, weight, ids) if native
              else reference(metadata, x, weight, ids))
    energy = 0.01 * output.square().sum() / x.shape[0]
    forces = -torch.autograd.grad(energy, pos, create_graph=True)[0]
    pure = torch.autograd.grad(forces.square().mean(), (projection, weight), retain_graph=True)
    full = torch.autograd.grad(0.1 * energy.square() + forces.square().mean(), (pos, projection, weight))
    return [energy, forces, *pure, *full]


@pytest.mark.parametrize("layout,experts,elements,nodes", [
    ("ir_mul", 1, 1, 1025), ("mul_ir", 3, 1, 1025),
    ("ir_mul", 1, 2, 1025), ("mul_ir", 1, 1, 1023),
])
def test_simt_fragment_tails_experts_and_fallback_force_loss(layout, experts, elements, nodes):
    torch.manual_seed(3921)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = ((0, 0, 0, 65, 67, 1, 65**-0.5),
             (65 * experts, 4355, 67 * experts, 69, 71, 2, 69**-0.5))
    din, dout, numel = 203 * experts, 209 * experts, 9254
    metadata = (din, dout, numel, experts, layout, paths)
    pos = torch.randn(nodes, 3, device="cuda")
    projection = torch.randn(3, din, device="cuda") * 0.1
    weights = torch.randn(elements, experts, numel, device="cuda")
    ids = torch.arange(nodes, device="cuda") % elements
    outputs = []
    for native in (False, True):
        leaves = [value.detach().clone().requires_grad_() for value in (pos, projection, weights)]
        outputs.append(chain(metadata, *leaves, ids, native))
    for actual, expected in zip(outputs[1], outputs[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)


def test_simt_multiple_paths_same_output_group():
    torch.manual_seed(1237)
    torch.backends.cuda.matmul.allow_tf32 = False
    paths = ((0, 0, 0, 65, 67, 1, 65**-0.5), (65, 4355, 0, 69, 67, 1, 69**-0.5))
    metadata = (134, 67, 8978, 1, "ir_mul", paths)
    x = torch.randn(1025, 134, device="cuda", requires_grad=True)
    weight = torch.randn(1, 1, 8978, device="cuda", requires_grad=True)
    ids = torch.zeros(1025, device="cuda", dtype=torch.int64)
    xr, wr = x.detach().clone().requires_grad_(), weight.detach().clone().requires_grad_()
    actual, expected = indexed_linear(repr(metadata), x, weight, ids), reference(metadata, xr, wr, ids)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)
    for leafs, result in (((x, weight), actual), ((xr, wr), expected)):
        first = torch.autograd.grad(result.square().mean(), leafs, create_graph=True)
        second = torch.autograd.grad(sum(value.square().mean() for value in first), leafs)
        if result is actual:
            actual_gradients = [*first, *second]
        else:
            for actual_gradient, expected_gradient in zip(actual_gradients, [*first, *second], strict=True):
                torch.testing.assert_close(actual_gradient, expected_gradient, atol=3e-5, rtol=5e-5)
