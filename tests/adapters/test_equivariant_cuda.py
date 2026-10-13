"""Reference comparisons including derivatives needed for force-loss training."""

import os

os.environ.setdefault("FASTEQ_UNIFORM1D_FWD_TUNE", "0")
os.environ.setdefault("FASTEQ_UNIFORM1D_BWD_TUNE", "0")
for name in ("EQX", "OEQ", "CUE", "EQT"):
    os.environ["TACE_USE_" + name] = "0"

import pytest
import torch
from e3nn import o3
from e3nn.nn import Gate

from fasteq.adapters import adapt_model, spherical_harmonics

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def check(module, args, *, atol=2e-9, rtol=2e-8):
    reference = module.cuda()
    native = adapt_model(reference)
    assert reference.state_dict().keys() == native.state_dict().keys()
    xs = [[x.detach().clone().requires_grad_(True) for x in args] for _ in range(2)]
    outputs, gradients, hessians = [], [], []
    seed = directions = None
    for model, inputs in zip((reference, native), xs):
        leaves = inputs + list(model.parameters())
        out = model(*inputs)
        if seed is None:
            seed = torch.randn_like(out)
            directions = [torch.randn_like(x) for x in leaves]
        first = torch.autograd.grad(
            (out * seed).sum() + out.square().sum() / 7,
            leaves,
            create_graph=True,
            allow_unused=True,
        )
        loss = sum(
            (g * d).sum()
            for g, d in zip(first, directions)
            if g is not None and g.requires_grad
        )
        second = (
            torch.autograd.grad(loss, leaves, allow_unused=True)
            if torch.is_tensor(loss)
            else [None] * len(leaves)
        )
        outputs.append(out)
        gradients.append(first)
        hessians.append(second)
    torch.testing.assert_close(outputs[0], outputs[1], atol=atol, rtol=rtol)
    for groups in (gradients, hessians):
        for a, b in zip(*groups):
            if a is None:
                assert b is None or torch.count_nonzero(b) == 0
            elif b is None:
                assert torch.count_nonzero(a) == 0
            else:
                torch.testing.assert_close(a, b, atol=atol * 10, rtol=rtol * 10)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("mode", ["uvu", "uuu", "uvw"])
def test_tp(dtype, mode):
    torch.manual_seed(10)
    a = "4x0e+4x1o"
    b = "1x0e+1x1o" if mode == "uvu" else a
    instructions = [
        (0, 0, 0, mode, True),
        (1, 0, 1, mode, True),
        (0, 1, 1, mode, True),
        (1, 1, 0, mode, True),
    ]
    m = o3.TensorProduct(
        a, b, a, instructions, internal_weights=False, shared_weights=False
    ).to(dtype)
    xs = [
        torch.randn(3, n, device="cuda", dtype=dtype)
        for n in (m.irreps_in1.dim, m.irreps_in2.dim, m.weight_numel)
    ]
    check(
        m,
        xs,
        atol=3e-5 if dtype == torch.float32 else 2e-9,
        rtol=3e-5 if dtype == torch.float32 else 2e-8,
    )


@pytest.mark.parametrize("shared", [True, False])
@pytest.mark.parametrize("bias", [True, False])
def test_linear(shared, bias):
    # e3nn 0.6's einsum optimizer uses an invalid batched bias example here.
    m = o3.Linear(
        "3x0e+2x1o",
        "4x0e+3x1o",
        internal_weights=False,
        shared_weights=shared,
        biases=bias,
        _optimize_einsums=False,
    ).double()
    xs = [
        torch.randn(3, 9, device="cuda", dtype=torch.float64),
        torch.randn(
            *(() if shared else (3,)),
            m.weight_numel,
            device="cuda",
            dtype=torch.float64,
        ),
    ]
    if bias:
        xs.append(torch.randn(m.bias_numel, device="cuda", dtype=torch.float64))
    check(m, xs)


@pytest.mark.parametrize("normalize", [True, False])
@pytest.mark.parametrize("normalization", ["component", "norm", "integral"])
def test_sh(normalize, normalization):
    m = o3.SphericalHarmonics(
        list(range(4)), normalize=normalize, normalization=normalization
    )
    check(m, [torch.randn(5, 3, device="cuda", dtype=torch.float64)])


def test_sh_origin_empty():
    for n in (0, 1):
        x = torch.zeros(n, 3, device="cuda", dtype=torch.float64, requires_grad=True)
        y = spherical_harmonics([0, 1, 2], x, True, "component")
        ref = o3.spherical_harmonics([0, 1, 2], x, True, "component")
        torch.testing.assert_close(y, ref)
        torch.testing.assert_close(
            torch.autograd.grad(y.sum(), x)[0], torch.autograd.grad(ref.sum(), x)[0]
        )


def test_gates():
    from eqx.o2 import Gate as O2Gate
    from prophet.model import Gate as ProphetGate

    modules = [
        Gate("2x0e", [torch.nn.SiLU()], "3x0e", [torch.nn.Sigmoid()], "3x1o"),
        ProphetGate("8x0e+3x1o"),
        O2Gate("2x0ee", [torch.nn.SiLU()], "3x0ee", [torch.nn.Sigmoid()], "3x1me"),
    ]
    for m in modules:
        dim = m.irreps_out.dim if isinstance(m, ProphetGate) else m.irreps_in.dim
        check(m, [torch.randn(3, dim, device="cuda", dtype=torch.float64)])


@pytest.mark.parametrize("centering", [True, False])
@pytest.mark.parametrize("affine", [True, False])
def test_norm(centering, affine):
    from prophet.layer_norm import RMSLayerNorm
    from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm

    modules = [
        RMSLayerNorm(o3.Irreps("3x0e+2x1o+4x2e"), centering=centering, affine=affine),
        EquivariantMergeLayerNorm([0, 1, 2], 4, centering=centering, affine=affine),
    ]
    for m in modules:
        shape = (3, 29) if isinstance(m, RMSLayerNorm) else (3, 9, 4)
        check(m.double(), [torch.randn(*shape, device="cuda", dtype=torch.float64)])


def test_o2_tp_linear():
    from eqx.o2 import Linear, TensorProduct

    m = Linear("3x0ee+3x1me", "4x0ee+4x1me", biases=True).double()
    check(m, [torch.randn(3, 9, device="cuda", dtype=torch.float64)])
    m = TensorProduct(
        "3x1me",
        "3x1me",
        "3x0ee+3x0oe+3x2me",
        [(0, 0, i, "uuu", True) for i in range(3)],
    ).double()
    check(m, [torch.randn(3, 6, device="cuda", dtype=torch.float64) for _ in range(2)])


def test_wigner_reuse():
    from eqx.o2.wigner import WignerD

    class Frames(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.frames = WignerD(mmax=2, lmax=3, method="recursive")

        def forward(self, x):
            forward, inverse = self.frames(x)
            return torch.cat((forward.flatten(1), inverse.flatten(1)), -1)

    # Construct reference CG tables in FP64; .double() cannot undo FP32 rounding.
    previous = torch.get_default_dtype()
    try:
        torch.set_default_dtype(torch.float64)
        module = Frames()
    finally:
        torch.set_default_dtype(previous)
    check(module, [torch.randn(4, 3, device="cuda", dtype=torch.float64)])


def test_wide_norm_and_conversion():
    from tace.models._e3nn.layer_norm import EquivariantMergeLayerNorm

    m = EquivariantMergeLayerNorm([0, 1, 2, 3, 4], 64).double().cuda()
    x = torch.randn(2, 25, 64, device="cuda", dtype=torch.float64)
    check(m, [x])
    native = adapt_model(m)
    identity = [id(p) for p in native.parameters()]
    assert adapt_model(native, inplace=True) is native
    assert identity == [id(p) for p in native.parameters()]
    clone = adapt_model(native)
    clone = clone.float()
    torch.testing.assert_close(
        clone(x.float()), m.float()(x.float()), atol=3e-6, rtol=3e-5
    )
