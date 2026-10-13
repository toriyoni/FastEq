"""Analytic adjoints of FastEq scalar conditioning and rotation expressions."""

import pytest
import torch
from e3nn import o3

from fasteq.ops.scalar_expression import ScalarProgram, scalar_expression
from fasteq.triton.irreps_invariants import InvariantPlan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def check(actual, expected, left, right, dtype):
    atol, rtol = (2e-5, 2e-4) if dtype == torch.float32 else (2e-11, 2e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    seed = torch.randn_like(actual)
    ga = torch.autograd.grad((actual.sin() * seed).sum(), left, create_graph=True)
    gb = torch.autograd.grad((expected.sin() * seed).sum(), right, create_graph=True)
    for a, b in zip(ga, gb):
        torch.testing.assert_close(a, b, atol=atol * 5, rtol=rtol * 5)
    aa = torch.autograd.grad(sum(g.square().sum() for g in ga), left)
    bb = torch.autograd.grad(sum(g.square().sum() for g in gb), right)
    for a, b in zip(aa, bb):
        torch.testing.assert_close(a, b, atol=atol * 20, rtol=rtol * 20)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("nodes", [0, 7])
def test_invariants(dtype, nodes):
    irreps = o3.Irreps("2x0e+3x1o+1x2e")
    plan = InvariantPlan(irreps)
    # Cache creation under inference must not contaminate subsequent training.
    with torch.inference_mode():
        plan(torch.ones(nodes, irreps.dim, device="cuda", dtype=dtype))
    x = torch.randn(nodes, irreps.dim, device="cuda", dtype=dtype, requires_grad=True)
    reference = x.detach().clone().requires_grad_()
    expected = torch.cat(
        [
            reference[:, sl].reshape(nodes, mul, ir.dim).square().mean(-1)
            for (mul, ir), sl in zip(irreps, irreps.slices())
        ],
        -1,
    )
    check(plan(x), expected, (x,), (reference,), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_scalar_expression(dtype):
    p = ScalarProgram()
    a, b = p.add("input", 0, 0), p.add("input", 1, 0)
    out = p.add(
        "add",
        p.add("sin", a),
        p.add(
            "div",
            p.add("mul", a, b),
            p.add("sqrt", p.add("add", p.add("mul", b, b), p.constant(1))),
        ),
    )
    other = p.add("cos", b)
    metadata = repr((tuple(p.nodes), ((out,), (other,)), (1, 1)))
    a = torch.randn(7, 1, device="cuda", dtype=dtype, requires_grad=True)
    b = torch.randn_like(a, requires_grad=True)
    ra, rb = (x.detach().clone().requires_grad_() for x in (a, b))
    actual = scalar_expression(metadata, [a, b])[0]
    expected = ra.sin() + ra * rb / (rb.square() + 1).sqrt()
    # The unused second output exercises zero/None adjoint seed handling.
    check(actual, expected, (a, b), (ra, rb), dtype)
