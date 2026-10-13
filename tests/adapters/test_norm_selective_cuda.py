"""Autograd task pruning must preserve the indirect force-loss parameter chain."""

from copy import deepcopy

import pytest
import torch
from e3nn import o3

from fasteq.ops import equivariant_norm as native

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def module_and_plan(kind, dtype):
    if kind == "tace":
        from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm
        module = EquivariantMergeLayerNorm([0, 1, 2], 5, centering=True).cuda().to(dtype)
    else:
        from prophet.layer_norm import RMSLayerNorm
        module = RMSLayerNorm(o3.Irreps("5x0e+3x1o+2x2e"), centering=True).cuda().to(dtype)
    with torch.no_grad():
        for parameter in module.parameters():
            parameter.copy_(torch.randn_like(parameter))
    return module, native.NormPlan(module, tace=kind == "tace")


def record_launches(monkeypatch, calls, double_calls=None):
    # Observe the actual launch boundary, including a resident CUDA launch.
    original = native.NormPlan.launch

    def invoke(plan, fn, arguments, x, kwargs):
        if fn is native._backward_kernel:
            calls.append(kwargs.copy())
        if fn is native._double_kernel and double_calls is not None:
            double_calls.append(kwargs.copy())
        return original(plan, fn, arguments, x, kwargs)

    monkeypatch.setattr(native.NormPlan, "launch", invoke)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["tace", "prophet"])
def test_only_x_skips_all_affine_partials_and_reducers(dtype, kind, monkeypatch):
    torch.manual_seed(846)
    module, plan = module_and_plan(kind, dtype)
    x = torch.randn(17, *plan.shape, device="cuda", dtype=dtype, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    expected = module(xr)
    calls = []
    record_launches(monkeypatch, calls)

    def forbidden(*args, **kwargs):
        raise AssertionError("only-x Norm backward launched an affine reducer")

    monkeypatch.setattr(native, "_reduce_rows", forbidden)
    output = plan(module, x)
    seed = torch.randn_like(output)
    gradient = torch.autograd.grad(output, x, seed, create_graph=True)[0]
    reference = torch.autograd.grad(expected, xr, seed, create_graph=True)[0]
    atol, rtol = (4e-5, 4e-5) if dtype == torch.float32 else (2e-10, 2e-9)
    torch.testing.assert_close(gradient, reference, atol=atol, rtol=rtol)
    assert all(parameter.requires_grad for parameter in module.parameters())
    assert len(calls) == 1 and calls[0]["DXFLAG"]
    assert calls[0]["WIDTH"] == 0
    assert not any(calls[0]["GGRADFLAGS"] + calls[0]["BGRADFLAGS"])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["tace", "prophet"])
def test_force_loss_recovers_affine_gradients_after_pruned_first_backward(dtype, kind, monkeypatch):
    torch.manual_seed(926)
    torch.backends.cuda.matmul.allow_tf32 = False
    module, plan = module_and_plan(kind, dtype)
    reference = deepcopy(module)
    coordinates = torch.randn(19, 3, device="cuda", dtype=dtype, requires_grad=True)
    coordinates_ref = coordinates.detach().clone().requires_grad_()
    projection = torch.randn(3, plan.dim, device="cuda", dtype=dtype, requires_grad=True)
    projection_ref = projection.detach().clone().requires_grad_()
    calls, double_calls = [], []
    record_launches(monkeypatch, calls, double_calls)
    output = plan(module, (coordinates @ projection).reshape(19, *plan.shape))
    expected = reference((coordinates_ref @ projection_ref).reshape(19, *plan.shape))
    energy, energy_ref = output.square().mean(), expected.square().mean()
    force = -torch.autograd.grad(energy, coordinates, create_graph=True)[0]
    force_ref = -torch.autograd.grad(energy_ref, coordinates_ref, create_graph=True)[0]
    assert len(calls) == 1 and calls[0]["WIDTH"] == 0
    target = torch.randn_like(force)
    loss = (force - target).square().mean()
    loss_ref = (force_ref - target).square().mean()
    loss.backward()
    loss_ref.backward()
    atol, rtol = (8e-4, 8e-4) if dtype == torch.float32 else (4e-9, 4e-8)
    torch.testing.assert_close(force, force_ref, atol=atol, rtol=rtol)
    torch.testing.assert_close(projection.grad, projection_ref.grad, atol=atol, rtol=rtol)
    for actual, desired in zip(module.parameters(), reference.parameters(), strict=True):
        assert actual.grad is not None and desired.grad is not None
        torch.testing.assert_close(actual.grad, desired.grad, atol=atol, rtol=rtol)
    assert any(any(call["GGRADFLAGS"]) for call in double_calls)
    assert any(call["WIDTH"] > 0 for call in calls[1:])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_one_gamma_uses_only_its_compact_partial_width(dtype, monkeypatch):
    module, plan = module_and_plan("prophet", dtype)
    x = torch.randn(13, *plan.shape, device="cuda", dtype=dtype, requires_grad=True)
    selected = module.affine_weight[1]
    expected = torch.autograd.grad(module(x).square().sum(), selected)[0]
    calls = []
    record_launches(monkeypatch, calls)
    actual = torch.autograd.grad(plan(module, x).square().sum(), selected)[0]
    atol, rtol = (4e-4, 4e-4) if dtype == torch.float32 else (2e-9, 2e-8)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    assert not calls[0]["DXFLAG"]
    assert calls[0]["WIDTH"] == selected.numel()
    assert calls[0]["GGRADFLAGS"] == (False, True, False)
    assert not any(calls[0]["BGRADFLAGS"])
