"""Original-model comparisons for reused FastEq Gate and SH kernels."""

import pytest
import torch
from e3nn import o3
from e3nn.nn import Gate

from fasteq.ops.gate import GatePlan
from fasteq.ops.spherical_harmonics import spherical_harmonics

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def compare(fn, reference, x, dtype):
    left = x.detach().clone().requires_grad_()
    right = x.detach().clone().requires_grad_()
    actual, expected = fn(left), reference(right)
    atol, rtol = (3e-5, 3e-5) if dtype == torch.float32 else (2e-10, 2e-9)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    seed = torch.randn_like(actual, requires_grad=True)
    seed_ref = seed.detach().clone().requires_grad_()
    first = torch.autograd.grad((actual * seed).sum() + actual.square().sum() / 7, left, create_graph=True)[0]
    first_ref = torch.autograd.grad((expected * seed_ref).sum() + expected.square().sum() / 7, right, create_graph=True)[0]
    torch.testing.assert_close(first, first_ref, atol=atol * 10, rtol=rtol * 10)
    direction = torch.randn_like(first)
    second = torch.autograd.grad(first, (left, seed), direction, allow_unused=True)
    second_ref = torch.autograd.grad(first_ref, (right, seed_ref), direction, allow_unused=True)
    for actual, expected in zip(second, second_ref):
        if expected is None:
            assert actual is None or not torch.count_nonzero(actual)
        elif actual is None:
            assert not torch.count_nonzero(expected)
        else:
            torch.testing.assert_close(actual, expected, atol=atol * 30, rtol=rtol * 30)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["o3", "prophet", "o2", "o2_sign", "o3_scalar"])
def test_gate_without_eqx_runtime(dtype, kind, monkeypatch):
    from eqx.kernels import cuda
    from eqx.o2 import Gate as O2Gate
    from prophet.model import Gate as ProphetGate

    def forbidden(*args, **kwargs):
        raise AssertionError("An EQX execution backend was called")

    monkeypatch.setattr(cuda, "runtime", forbidden)
    monkeypatch.setattr(cuda, "kernels", forbidden)
    torch.manual_seed(610)
    if kind == "o3":
        module = Gate("2x0e", [torch.nn.SiLU()], "3x0e", [torch.nn.Sigmoid()], "3x1o")
    elif kind == "o3_scalar":
        module = Gate("2x0e", [torch.nn.Tanh()], "3x0e", [None], "3x0e")
    elif kind == "prophet":
        module = ProphetGate("8x0e+3x1o")
    elif kind == "o2":
        module = O2Gate("2x0ee", [torch.nn.SiLU()], "3x0ee", [torch.nn.Sigmoid()], "3x1me")
    else:
        module = O2Gate("2x0oe+3x0ee", [torch.nn.Tanh(), torch.nn.SiLU()],
                        "3x0oe+2x0ee", [torch.nn.Tanh(), torch.nn.Sigmoid()], "5x1me")
        assert module._has_signs
    module = module.cuda().to(dtype)
    plan = GatePlan(module, "o3" if kind.startswith("o3") else "o2" if kind.startswith("o2") else kind)
    # Multiple leading axes ensure custom-function flattening remains connected
    # to the original public input during the second backward.
    x = torch.randn(2, 3, plan.dim, device="cuda", dtype=dtype)
    reference = lambda x: module(x.reshape(-1, plan.dim)).reshape(*x.shape[:-1], plan.outdim)
    compare(plan, reference, x, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_sh_without_eqx_runtime(dtype, normalize, normalization, monkeypatch):
    from eqx.kernels import cuda

    def forbidden(*args, **kwargs):
        raise AssertionError("An EQX execution backend was called")

    monkeypatch.setattr(cuda, "runtime", forbidden)
    monkeypatch.setattr(cuda, "kernels", forbidden)
    degrees = [0, 1, 2, 3]
    x = torch.randn(2, 4, 3, device="cuda", dtype=dtype)
    compare(lambda x: spherical_harmonics(degrees, x, normalize, normalization),
            lambda x: o3.spherical_harmonics(degrees, x, normalize, normalization), x, dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_gate_sh_empty_origin(dtype):
    module = Gate("2x0e", [torch.nn.SiLU()], "3x0e", [torch.nn.Sigmoid()], "3x1o").cuda().to(dtype)
    plan = GatePlan(module, "o3")
    x = torch.empty(0, plan.dim, device="cuda", dtype=dtype, requires_grad=True)
    y = plan(x)
    assert y.shape == (0, plan.outdim)
    torch.autograd.grad(y.sum(), x)
    for n in (0, 1):
        x = torch.zeros(n, 3, device="cuda", dtype=dtype, requires_grad=True)
        y = spherical_harmonics([0, 1, 2], x, True, "component")
        ref = o3.spherical_harmonics([0, 1, 2], x, True, "component")
        torch.testing.assert_close(y, ref)
        torch.testing.assert_close(torch.autograd.grad(y.sum(), x)[0], torch.autograd.grad(ref.sum(), x)[0])
