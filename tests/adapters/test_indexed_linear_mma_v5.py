"""MMA derivative regression through positions, energy and full force loss."""

import pytest
import torch

from fasteq.ops.indexed_linear import indexed_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(metadata, features, weight):
    din, dout, numel, experts, layout, paths = metadata
    assert experts == 1
    pieces = []
    for sx, sw, sy, ci, co, dim, scale in paths:
        values = features[:, sx : sx + ci * dim]
        values = (values.reshape(-1, dim, ci) if layout == "ir_mul"
                  else values.reshape(-1, ci, dim).transpose(1, 2))
        result = (values @ weight[0, 0, sw : sw + ci * co].reshape(ci, co)) * scale
        if layout == "mul_ir":
            result = result.transpose(1, 2)
        pieces.append(result.flatten(1))
    return torch.cat(pieces, dim=1)


def evaluate(metadata, positions, projection, weights, native):
    x = torch.sin(positions @ projection)
    ids = torch.zeros(x.shape[0], dtype=torch.int64, device=x.device)
    output = (indexed_linear(repr(metadata), x, weights, ids) if native
              else reference(metadata, x, weights))
    energy = 0.01 * output.square().sum() / x.shape[0]
    forces = -torch.autograd.grad(energy, positions, create_graph=True)[0]
    # Only positions are requested at first. The force adjoint must still
    # reach both live projection and weights, including output-dependent dY.
    force_gradients = torch.autograd.grad(forces.square().mean(), (projection, weights), retain_graph=True)
    full_gradients = torch.autograd.grad(0.1 * energy.square() + forces.square().mean(), (positions, projection, weights))
    return [energy, forces, *force_gradients, *full_gradients]


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("large", [False, True])
def test_positions_force_loss_keeps_full_parameter_chain(layout, large):
    torch.manual_seed(2819)
    torch.backends.cuda.matmul.allow_tf32 = False
    # The small case triggers MMA while testing all fragment and final node
    # tails; the other is the exact real two-path TECE shape plus one node.
    ci0, co0, ci1, co1 = (384, 576, 256, 384) if large else (65, 67, 69, 71)
    paths = ((0, 0, 0, ci0, co0, 1, ci0**-0.5),
             (ci0, ci0 * co0, co0, ci1, co1, 2, ci1**-0.5))
    din, dout, numel = ci0 + 2 * ci1, co0 + 2 * co1, ci0 * co0 + ci1 * co1
    metadata = (din, dout, numel, 1, layout, paths)
    nodes = 3073 if large else 1025
    positions = torch.randn(nodes, 3, dtype=torch.float64, device="cuda")
    projection = torch.randn(3, din, dtype=torch.float64, device="cuda") * 0.1
    weights = torch.randn(1, 1, numel, dtype=torch.float64, device="cuda")
    results = []
    for native in (False, True):
        leaves = [value.detach().clone().requires_grad_() for value in (positions, projection, weights)]
        results.append(evaluate(metadata, *leaves, native))
    assert results[0][3].abs().sum() > 0
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
