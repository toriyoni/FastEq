"""FastEq frame CUDA kernels match reference math through force-loss order."""

import pytest
import torch
from fasteq.ops.local_product import LocalSplit, local_product
from fasteq.ops.rotation import rotary_product, rotate, wigner_packed

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def derivatives(fn, args):
    xs = [x.detach().clone().requires_grad_() for x in args]
    out = fn(*xs)
    out = torch.cat([x.flatten(1) for x in out], 1) if isinstance(out, tuple) else out
    objective = out.square().sum()
    if not objective.requires_grad:
        return out, [None] * len(xs), [None] * len(xs)
    first = torch.autograd.grad(objective, xs, create_graph=True, allow_unused=True)
    loss = sum(g.square().sum() for g in first if g is not None and g.requires_grad)
    second = (
        torch.autograd.grad(loss, xs, allow_unused=True)
        if torch.is_tensor(loss)
        else [None] * len(xs)
    )
    return out, first, second


def compare(actual, expected, args, dtype):
    a, b = derivatives(actual, args), derivatives(expected, args)
    for left, right in zip(a, b):
        for x, y in (
            zip(left, right) if isinstance(left, (tuple, list)) else [(left, right)]
        ):
            if x is None or y is None:
                z = y if x is None else x
                assert z is None or torch.count_nonzero(z) == 0
            else:
                torch.testing.assert_close(
                    x,
                    y,
                    atol=3e-4 if dtype == torch.float32 else 3e-8,
                    rtol=3e-4 if dtype == torch.float32 else 3e-7,
                )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fasteq_frame_rotation(dtype):
    degrees = (0, 1, 1, 1)
    mask = torch.tensor(
        [[a == b for b in degrees] for a in degrees], device="cuda", dtype=dtype
    )
    features = torch.randn(4, 4, 5, device="cuda", dtype=dtype)
    matrix = torch.randn(4, 4, 4, device="cuda", dtype=dtype)
    compare(
        lambda x, m: rotate(x, m, repr((degrees, degrees))),
        lambda x, m: torch.bmm(m * mask, x),
        [features, matrix],
        dtype,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_fasteq_rotary(dtype):
    lmax, mmax, heads, channels = 2, 1, 2, 4
    angular = 7
    values = [
        torch.randn(4, angular, channels, device="cuda", dtype=dtype) for _ in range(2)
    ]
    values.append(torch.randn(4, mmax + 1, heads, 2, device="cuda", dtype=dtype))

    def reference(q, k, p):
        n = lmax + 1
        q0 = q[:, :n].reshape(4, n, heads, channels // heads)
        k0 = k[:, :n].reshape_as(q0)
        out = (q0 * k0).sum((1, 3)) * p[:, 0, :, 0]
        offset = n
        for m in range(1, mmax + 1):
            n = lmax + 1 - m
            qm = q[:, offset : offset + 2 * n].reshape(
                4, 2, n, heads, channels // heads
            )
            km = k[:, offset : offset + 2 * n].reshape_as(qm)
            out = out + (qm * km).sum((1, 2, 4)) * p[:, m, :, 0]
            out = (
                out
                + (qm[:, 1] * km[:, 0] - qm[:, 0] * km[:, 1]).sum((1, 3))
                * p[:, m, :, 1]
            )
            offset += 2 * n
        return out

    compare(
        lambda q, k, p: rotary_product(q, k, p, lmax, mmax, heads),
        reference,
        values,
        dtype,
    )


@pytest.mark.parametrize("degree", [0, 1, 2, 3])
def test_fasteq_wigner_polynomial(degree):
    from eqx.o2 import WignerD

    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        frame = WignerD(degree, degree, method="recursive").cuda()
    finally:
        torch.set_default_dtype(previous)
    x = torch.randn(4, 3, device="cuda", dtype=torch.float64)
    compare(
        lambda v: wigner_packed(v, degree),
        lambda v: frame.forward_packed(v, method="recursive"),
        [x],
        torch.float64,
    )


def test_fasteq_local_missing_factors_and_split():
    width = 3
    metadata = repr((width, (2, 2, 2), (((0, 0, 0), 1.0), ((1, -1, 1), 0.5))))
    x, y = [torch.randn(4, 6, device="cuda", dtype=torch.float64) for _ in range(2)]
    compare(
        lambda x, y: local_product(metadata, x, y),
        lambda x, y: torch.cat((x[:, :3] * y[:, :3], 0.5 * x[:, 3:]), 1),
        [x, y],
        torch.float64,
    )
    # Gradient order two must not resurrect the missing y factor in path one.
    split = repr((width, (3, 1, 1), (((0, 0, -1), 1.0), ((2, -1, 0), 1.0))))
    x = torch.randn(4, 9, device="cuda", dtype=torch.float64)
    compare(
        lambda v: LocalSplit.apply(v, ((0, 3), (6, 3)), split),
        lambda v: (v[:, :3], v[:, 6:9]),
        [x],
        torch.float64,
    )
