"""Gate/normalization fusion against the unchanged Prophet torch expression."""

from types import SimpleNamespace

import pytest
import torch
from e3nn import o3
from prophet.layer_norm import RMSLayerNorm
from prophet.model import Gate, ProphetBlock, TPConvolution

from fasteq.adapters.fusion import ProphetPostprocess, install_prophet_fusions

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("pre_norm", [False, True])
@pytest.mark.parametrize("centering", [False, True])
def test_prophet_gate_norm_fusion(dtype, pre_norm, centering):
    torch.manual_seed(710)
    out_irreps = o3.Irreps("3x0e+3x1o+3x2e")
    gate_irreps = o3.Irreps("9x0e+3x1o+3x2e")
    gate = Gate(gate_irreps).cuda().to(dtype)
    before = RMSLayerNorm(gate_irreps, centering=centering).cuda().to(dtype) if pre_norm else None
    after = RMSLayerNorm(out_irreps, centering=centering).cuda().to(dtype)
    for norm in (before, after):
        if norm is not None:
            with torch.no_grad():
                for weight in norm.affine_weight:
                    weight.uniform_(0.5, 1.5)
                if norm.affine_bias is not None:
                    for bias in norm.affine_bias:
                        if bias is not None:
                            bias.uniform_(-0.2, 0.2)
    conv = SimpleNamespace(linear_2=SimpleNamespace(irreps_out=gate_irreps), gate=gate, layer_norm=before)
    plan = ProphetPostprocess(conv, after)
    parameters = tuple(plan.parameters)
    inputs = [torch.randn(5, gate_irreps.dim, dtype=dtype, device="cuda", requires_grad=True) for _ in range(2)]
    copies = [x.detach().clone().requires_grad_() for x in inputs]
    expected = copies[0] + copies[1]
    if before is not None:
        expected = before(expected)
    expected = after(gate(expected))
    actual = plan(*inputs)
    atol, rtol = (5e-6, 5e-5) if dtype == torch.float32 else (1e-11, 1e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    seed = torch.randn_like(actual)
    left = torch.autograd.grad((actual.sin() * seed).sum(), (*inputs, *parameters), create_graph=True)
    right = torch.autograd.grad((expected.sin() * seed).sum(), (*copies, *parameters), create_graph=True)
    for a, b in zip(left, right):
        torch.testing.assert_close(a, b, atol=atol * 5, rtol=rtol * 5)
    left2 = torch.autograd.grad(sum(x.square().sum() for x in left), (*inputs, *parameters))
    right2 = torch.autograd.grad(sum(x.square().sum() for x in right), (*copies, *parameters))
    for a, b in zip(left2, right2):
        torch.testing.assert_close(a, b, atol=atol * 50, rtol=rtol * 50)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_prophet_fused_plan_empty_and_no_affine(dtype):
    gate_irreps = o3.Irreps("4x0e+2x1o")
    norm = RMSLayerNorm(o3.Irreps("2x0e+2x1o"), affine=False, centering=False).cuda().to(dtype)
    conv = SimpleNamespace(linear_2=SimpleNamespace(irreps_out=gate_irreps), gate=Gate(gate_irreps).cuda().to(dtype), layer_norm=None)
    plan = ProphetPostprocess(conv, norm)
    x = torch.empty(0, gate_irreps.dim, device="cuda", dtype=dtype, requires_grad=True)
    y = plan(x, x)
    assert y.shape == (0, norm.irreps.dim)
    torch.autograd.grad(y.sum(), x)


def test_prophet_fusion_tracks_replaced_parameters():
    """An existing plan must read newly assigned checkpoint parameters."""
    ir = o3.Irreps("6x0e+2x1o+2x2e")
    gate = Gate(ir).cuda().double()
    norm = RMSLayerNorm(o3.Irreps("2x0e+2x1o+2x2e"), centering=True).cuda().double()
    owner = torch.nn.ModuleDict({"gate": gate, "norm": norm})
    plan = ProphetPostprocess(SimpleNamespace(linear_2=SimpleNamespace(irreps_out=ir), gate=gate, layer_norm=None), norm)
    original = tuple(plan.parameters)
    state = {name: value.clone() for name, value in owner.state_dict().items()}
    for name, value in state.items():
        if "affine_weight" in name:
            value.fill_(1.7)
        elif "affine_bias" in name:
            value.fill_(0.3)
    owner.load_state_dict(state, assign=True)
    assert all(old is not new for old, new in zip(original, plan.parameters))
    x = torch.randn(4, ir.dim, device="cuda", dtype=torch.float64, requires_grad=True)
    skip = torch.randn_like(x)
    actual = plan(x, skip)
    expected = norm(gate(x + skip))
    torch.testing.assert_close(actual, expected, atol=1e-11, rtol=1e-10)
    torch.autograd.grad(actual.square().sum(), (x, *plan.parameters))

    overwrite = torch.__future__.get_overwrite_module_params_on_conversion()
    try:
        torch.__future__.set_overwrite_module_params_on_conversion(True)
        original = tuple(plan.parameters)
        owner.float()
        assert all(old is not new for old, new in zip(original, plan.parameters))
    finally:
        torch.__future__.set_overwrite_module_params_on_conversion(overwrite)
    x = x.detach().float().requires_grad_()
    skip = skip.float()
    actual = plan(x, skip)
    expected = norm(gate(x + skip))
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    torch.autograd.grad(actual.square().sum(), (x, *plan.parameters))

    class Scale(torch.nn.Module):
        def forward(self, weight):
            return weight * 1.5

    torch.nn.utils.parametrize.register_parametrization(norm.affine_weight, "0", Scale())
    actual = plan(x, skip)
    expected = norm(gate(x + skip))
    torch.testing.assert_close(actual, expected, atol=5e-6, rtol=5e-5)
    torch.autograd.grad(actual.square().sum(), (x, *norm.parameters()))


def test_prophet_fusion_preserves_standalone_convolution_api():
    """Only the parent Block call may move its norm into the convolution."""
    class Projection(torch.nn.Module):
        def __init__(self, irreps):
            super().__init__()
            self.irreps_out = irreps

        def forward(self, features):
            return features

    class Product(torch.nn.Module):
        def forward(self, features, sh, weights):
            return features

    class FFN(torch.nn.Module):
        def forward(self, scalars, irreps):
            return scalars, irreps

    # The projections/product are identity stubs: no new TP shape is compiled,
    # and the real TPConvolution/ProphetBlock forward methods remain exercised.
    ir = o3.Irreps("12x0e+4x1o+4x2e")
    out = o3.Irreps("4x0e+4x1o+4x2e")
    conv = TPConvolution.__new__(TPConvolution)
    torch.nn.Module.__init__(conv)
    conv.index_weights = conv.kernel = False
    conv.avg_n_neighbors = 1.0
    conv.linear_1 = conv.skip = torch.nn.Identity()
    conv.radial_mlp = conv.sort = torch.nn.Identity()
    conv.tp = Product()
    conv.linear_2 = Projection(ir)
    conv.layer_norm = None
    conv.gate = Gate(ir).cuda().double()
    norm = RMSLayerNorm(out, centering=False).cuda().double()
    block = ProphetBlock.__new__(ProphetBlock)
    torch.nn.Module.__init__(block)
    block.cutoffs = [4.0]
    block.scalar_dim = 4
    block.tpconv = torch.nn.ModuleList([conv])
    block.norm = torch.nn.ModuleList([norm])
    block.eq_ffn = torch.nn.ModuleList([FFN()])
    block.mix_irreps = block.mix_scalars = torch.nn.Identity()
    install_prophet_fusions(block)

    x = torch.randn(3, ir.dim, device="cuda", dtype=torch.float64)
    ids = torch.arange(3, device="cuda")
    scalar = torch.zeros(3, 1, device="cuda", dtype=torch.float64)
    envelope = torch.ones(3, device="cuda", dtype=torch.float64)
    args = (x, ids, scalar, scalar, ids, ids, envelope)
    gated = conv.gate(x * 2)
    normalized = norm(gated)
    default = conv(*args)
    opted_in = conv(*args, _fasteq_fuse_norm=True)
    torch.testing.assert_close(default, gated, atol=1e-11, rtol=1e-10)
    torch.testing.assert_close(opted_in, normalized, atol=1e-11, rtol=1e-10)
    assert not torch.allclose(default, normalized)
    _, from_block = block(
        torch.zeros(3, 4, device="cuda", dtype=torch.float64), x, ids,
        scalar, [scalar], ids, ids,
        [torch.ones(3, device="cuda", dtype=torch.bool)], [envelope],
    )
    torch.testing.assert_close(from_block, normalized, atol=1e-11, rtol=1e-10)
