"""FastEq norm precision, full affine gradients, and analytic double backward."""

import pytest
import torch
from e3nn import o3

from fasteq.ops.equivariant_norm import NormPlan


def test_norm_double_formula_cpu():
    torch.manual_seed(631)
    x = torch.randn(5, 14, dtype=torch.float64, requires_grad=True)
    gamma = torch.randn(6, dtype=torch.float64, requires_grad=True)
    beta = torch.randn(3, dtype=torch.float64, requires_grad=True)
    dy = torch.randn_like(x, requires_grad=True)
    index = torch.tensor([0, 1, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5, 5, 5])
    b = torch.rand(14, dtype=torch.float64) / 7
    def project(value):
        return torch.cat((value[:, :3] - value[:, :3].mean(1, keepdim=True), value[:, 3:]), 1)
    v = project(x)
    r = ((b*v*v).sum(1, keepdim=True) + 1e-5).rsqrt()
    gvec = gamma[index]
    y = r*v*gvec + torch.nn.functional.pad(beta, (0, 11))
    first = torch.autograd.grad(y, (x, gamma, beta), dy, create_graph=True)
    u, h, k = torch.randn_like(x), torch.randn_like(gamma), torch.randn_like(beta)
    expected = torch.autograd.grad(sum((a*s).sum() for a, s in zip(first, (u, h, k))),
                                   (x, dy, gamma, beta), allow_unused=True)
    g, t, hvec = dy*gvec, project(u), h[index]
    a, s = (g*v).sum(1, keepdim=True), (t*b*v).sum(1, keepdim=True)
    c = (t*g + dy*hvec*v).sum(1, keepdim=True)
    ddx = project(r*dy*hvec - b*v*r**3*c + 3*b*v*r**5*a*s - r**3*(s*g + a*b*t))
    ddy = r*(t*gvec + hvec*v) - r**3*s*v*gvec + torch.nn.functional.pad(k, (0, 11))
    ddgamma = torch.zeros_like(gamma).index_add(0, index, (dy*(r*t - r**3*s*v)).sum(0))
    for actual, reference in zip((ddx, ddy, ddgamma), expected[:3]):
        torch.testing.assert_close(actual, reference, atol=2e-12, rtol=2e-12)
    assert expected[3] is None


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["tace", "tace_duplicate", "prophet"])
@pytest.mark.parametrize("centering", [False, True])
@pytest.mark.parametrize("affine", [False, True])
def test_norm(dtype, kind, centering, affine, monkeypatch):
    from eqx.kernels import cuda
    from prophet.layer_norm import RMSLayerNorm
    from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm
    def forbidden(*args, **kwargs):
        raise AssertionError("EQX backend executed")
    monkeypatch.setattr(cuda, "runtime", forbidden)
    monkeypatch.setattr(cuda, "kernels", forbidden)
    torch.manual_seed(639)
    if kind.startswith("tace"):
        degrees = [0, 1, 2] if kind == "tace" else [0, 0, 1, 2]
        module = EquivariantMergeLayerNorm(degrees, 3, centering=centering, affine=affine).cuda().to(dtype)
        shape = (2, 3, module.dim, 3)
    else:
        module = RMSLayerNorm(o3.Irreps("3x0e+2x0o+2x1o+1x2e"), centering=centering, affine=affine).cuda().to(dtype)
        shape = (2, 3, module.irreps.dim)
    plan = NormPlan(module, kind.startswith("tace"))
    x = torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    parameters = tuple(module.parameters())
    actual = plan(module, x)
    # Prophet's original implementation requires one flattened node axis.
    reference = module(xr.reshape(-1, *plan.shape)).reshape(shape)
    atol, rtol = (4e-5, 4e-5) if dtype == torch.float32 else (2e-10, 2e-9)
    torch.testing.assert_close(actual, reference, atol=atol, rtol=rtol)
    seed = torch.randn_like(actual, requires_grad=True)
    seedr = seed.detach().clone().requires_grad_()
    first = torch.autograd.grad(actual, (x, *parameters), seed, create_graph=True)
    firstr = torch.autograd.grad(reference, (xr, *parameters), seedr, create_graph=True)
    directions = tuple(torch.randn_like(g) for g in first)
    for a, b in zip(first, firstr):
        torch.testing.assert_close(a, b, atol=atol * 10, rtol=rtol * 10)
    second = torch.autograd.grad(sum((g*d).sum() for g, d in zip(first, directions)),
                                (x, seed, *parameters), allow_unused=True)
    secondr = torch.autograd.grad(sum((g*d).sum() for g, d in zip(firstr, directions)),
                                 (xr, seedr, *parameters), allow_unused=True)
    for a, b in zip(second, secondr):
        if b is None:
            assert a is None or not torch.count_nonzero(a)
        else:
            assert a is not None
            torch.testing.assert_close(a, b, atol=atol * 20, rtol=rtol * 20)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_norm_empty_and_inference_cache(dtype):
    from prophet.layer_norm import RMSLayerNorm
    module = RMSLayerNorm(o3.Irreps("3x0e+2x1o"), centering=True).cuda().to(dtype)
    plan = NormPlan(module)
    with torch.inference_mode():
        plan(module, torch.randn(3, 9, dtype=dtype, device="cuda"))
    x = torch.randn(3, 9, dtype=dtype, device="cuda", requires_grad=True)
    y = plan(module, x)
    torch.autograd.grad(y.square().sum(), (x, *module.parameters()))
    x = torch.empty(0, 9, dtype=dtype, device="cuda", requires_grad=True)
    y = plan(module, x)
    assert y.shape == x.shape
    first = torch.autograd.grad(y.sum(), (x, *module.parameters()), create_graph=True)
    torch.autograd.grad(sum(g.sum() for g in first), (x, *module.parameters()), allow_unused=True)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_norm_live_parameters_after_warm_cache():
    from prophet.layer_norm import RMSLayerNorm
    module = RMSLayerNorm(o3.Irreps("3x0e+2x1o"), centering=True).cuda().double()
    plan = NormPlan(module)
    x = torch.randn(3, 9, device="cuda", dtype=torch.float64, requires_grad=True)
    plan(module, x)  # Warm the constant cache before replacing every parameter.
    before = tuple(module.parameters())
    state = {name: torch.full_like(value, 1.7 if "weight" in name else 0.3)
             for name, value in module.state_dict().items()}
    module.load_state_dict(state, assign=True)
    assert all(old is not new for old, new in zip(before, module.parameters()))
    actual = plan(module, x)
    torch.testing.assert_close(actual, module(x), atol=2e-10, rtol=2e-9)
    torch.autograd.grad(actual.square().sum(), (x, *module.parameters()))
    overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(True)
        before = tuple(module.parameters())
        module.float()
        assert all(old is not new for old, new in zip(before, module.parameters()))
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(overwrite)
    x = x.detach().float().requires_grad_()
    actual = plan(module, x)
    torch.testing.assert_close(actual, module(x), atol=4e-5, rtol=4e-5)
    torch.autograd.grad(actual.square().sum(), (x, *module.parameters()))
