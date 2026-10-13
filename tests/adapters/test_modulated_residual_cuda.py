"""Packed-free FastEq equivariant modulation and real force-loss adjoints."""

import pytest
import torch
from e3nn import o3
from eqx_cuda_spy import forbid_eqx_cuda

from fasteq.ops.modulation import ModulatedResidualPlan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("shape", [(0,), (7,), (2, 3)])
def test_modulated_residual_derivatives(dtype, shape):
    ir = o3.Irreps("2x0e+3x1o+1x2e")
    plan = ModulatedResidualPlan(ir)
    torch.manual_seed(164)
    with torch.inference_mode():
        plan(
            torch.zeros(*shape, ir.dim, dtype=dtype, device="cuda"),
            torch.zeros(*shape, plan.channels, dtype=dtype, device="cuda"),
            torch.zeros(*shape, ir.dim, dtype=dtype, device="cuda"),
        )
    values = [
        torch.randn(*shape, width * 2, dtype=dtype, device="cuda")[..., ::2]
        .detach()
        .requires_grad_()
        for width in (ir.dim, plan.channels, ir.dim)
    ]
    reference = [x.detach().clone().requires_grad_() for x in values]
    indices = torch.tensor(plan._index, device="cuda")
    with forbid_eqx_cuda():
        actual = plan(*values)
        expected = reference[2] + reference[0] * reference[1].index_select(-1, indices)
        atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-11, 2e-10)
        torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
        seed = torch.randn_like(actual)
        left = torch.autograd.grad(
            (actual.sin() * seed).sum(), values, create_graph=True
        )
        right = torch.autograd.grad(
            (expected.sin() * seed).sum(), reference, create_graph=True
        )
        for a, b in zip(left, right, strict=True):
            torch.testing.assert_close(a, b, atol=atol * 4, rtol=rtol * 4)
        ll = torch.autograd.grad(sum(x.square().sum() for x in left), values)
        rr = torch.autograd.grad(sum(x.square().sum() for x in right), reference)
        for a, b in zip(ll, rr, strict=True):
            torch.testing.assert_close(a, b, atol=atol * 20, rtol=rtol * 20)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_modulated_residual_force_loss(dtype):
    ir = o3.Irreps("3x0e+3x1o+3x2e")
    plan = ModulatedResidualPlan(ir)
    torch.manual_seed(632)
    position = torch.randn(9, 3, device="cuda", dtype=dtype, requires_grad=True)
    parameters = [
        torch.randn(3, width, device="cuda", dtype=dtype, requires_grad=True)
        for width in (ir.dim, plan.channels, ir.dim)
    ]
    left = (position, *parameters)
    right = tuple(x.detach().clone().requires_grad_() for x in left)
    idx = torch.tensor(plan._index, device="cuda")
    gradients = []
    for inputs, native in ((left, True), (right, False)):
        pos, projection, gating, skip = inputs
        x, g, residual = pos @ projection, torch.sin(pos @ gating), pos @ skip
        y = plan(x, g, residual) if native else residual + x * g.index_select(-1, idx)
        energy = torch.tanh(y).square().sum()
        force = -torch.autograd.grad(energy, pos, create_graph=True)[0]
        gradients.append(torch.autograd.grad(energy + force.square().sum(), inputs))
    for a, b in zip(*gradients, strict=True):
        torch.testing.assert_close(
            a,
            b,
            atol=3e-4 if dtype == torch.float32 else 2e-10,
            rtol=3e-3 if dtype == torch.float32 else 2e-9,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_ffn_all_parameter_force_loss(dtype):
    from prophet.model import EQFFN

    from fasteq.adapters import adapt_model

    torch.manual_seed(701)
    ir = o3.Irreps("3x0e+3x1o+3x2e")
    reference = EQFFN(ir).cuda().to(dtype)
    native = adapt_model(reference)
    pos = torch.randn(7, 3, device="cuda", dtype=dtype, requires_grad=True)
    scalar_projection = torch.randn(
        3, 3, device="cuda", dtype=dtype, requires_grad=True
    )
    feature_projection = torch.randn(
        3, ir.dim, device="cuda", dtype=dtype, requires_grad=True
    )
    source = (pos, scalar_projection, feature_projection)
    values = []
    for model in (reference, native):
        p, sw, fw = (value.detach().clone().requires_grad_() for value in source)
        scalar, features = model(p @ sw, p @ fw)
        energy = scalar.sin().square().mean() + features.tanh().square().mean()
        force = -torch.autograd.grad(energy, p, create_graph=True)[0]
        gradients = torch.autograd.grad(
            energy + force.square().mean(), (p, sw, fw, *model.parameters())
        )
        values.append((scalar, features, force, *gradients))
    for a, b in zip(*values, strict=True):
        torch.testing.assert_close(
            a,
            b,
            atol=4e-5 if dtype == torch.float32 else 2e-10,
            rtol=4e-4 if dtype == torch.float32 else 2e-9,
        )


def test_modulation_native_stream_and_graph():
    ir = o3.Irreps("3x0e+3x1o+3x2e")
    plan = ModulatedResidualPlan(ir)
    x = torch.randn(2, 3, ir.dim, device="cuda", dtype=torch.float64)
    gates = torch.randn(2, 3, plan.channels, device="cuda", dtype=torch.float64)
    residual = torch.randn_like(x)
    index = torch.tensor(plan._index, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        actual = plan(x, gates, residual)
        expected = residual + x * gates.index_select(-1, index)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
    assert plan._launchers and all(
        value is not None for value in plan._launchers.values()
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = plan(x, gates, residual)
    graph.replay()
    torch.testing.assert_close(result, expected, atol=2e-11, rtol=2e-10)
