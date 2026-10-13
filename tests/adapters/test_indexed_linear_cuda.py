"""General FastEq Linear accuracy against e3nn, without any EQX backend."""

import pytest
import torch
from e3nn import o3

from fasteq.ops.indexed_linear import IndexedLinear, indexed_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def reference(desc, x, w, ids, experts, layout):
    chunks = []
    for expert in range(experts):
        expert_inputs = []
        for item, sl in zip(desc.irreps_in, desc.irreps_in.slices(), strict=True):
            part = x[:, sl.start * experts : sl.stop * experts]
            part = (
                part.reshape(x.shape[0], item.ir.dim, experts, item.mul).permute(
                    0, 2, 3, 1
                )
                if layout == "ir_mul"
                else part.reshape(x.shape[0], experts, item.mul, item.ir.dim)
            )
            expert_inputs.append(part[:, expert].flatten(1))
        chunks.append(desc(torch.cat(expert_inputs, dim=1), w[ids, expert]))
    outputs = []
    for item, sl in zip(desc.irreps_out, desc.irreps_out.slices(), strict=True):
        part = torch.stack([value[:, sl] for value in chunks], dim=1).reshape(
            x.shape[0], experts, item.mul, item.ir.dim
        )
        if layout == "ir_mul":
            part = part.permute(0, 3, 1, 2)
        outputs.append(part.flatten(1))
    return torch.cat(outputs, dim=1)


def differentiate(output, x, w, cotangent):
    objective = (output * cotangent).mean() + 0.1 * output.square().mean()
    first = torch.autograd.grad(objective, (x, w), create_graph=True)
    second = torch.autograd.grad(sum(g.square().mean() for g in first), (x, w))
    return first, second


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("elements", [1, 3, 19])
@pytest.mark.parametrize("nodes", [7, 65])
def test_dense_linear_forward_first_second(dtype, layout, experts, elements, nodes):
    torch.manual_seed(1742)
    torch.backends.cuda.matmul.allow_tf32 = False
    desc = (
        o3.Linear(
            "7x0e+5x1o+3x2e+2x3e",
            "4x0e+8x1o+2x2e+1x3o",
            internal_weights=False,
            shared_weights=False,
        )
        .cuda()
        .to(dtype)
    )
    native = IndexedLinear(desc, experts, layout)
    storage = torch.randn(nodes, 2 * native.input_dim, device="cuda", dtype=dtype)
    x = storage[:, ::2].detach().requires_grad_()
    w = torch.randn(
        elements, experts, desc.weight_numel, device="cuda", dtype=dtype
    ).requires_grad_()
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    ids = torch.arange(nodes, device="cuda") % elements
    actual = native(x, w, ids)
    expected = reference(desc, xr, wr, ids, experts, layout)
    atol, rtol = (3e-5, 5e-5) if dtype == torch.float32 else (2e-11, 2e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    cotangent = torch.randn_like(actual)
    results = [
        differentiate(y, a, b, cotangent)
        for y, a, b in ((actual, x, w), (expected, xr, wr))
    ]
    for a, b in zip(*results, strict=True):
        for ga, gb in zip(a, b, strict=True):
            torch.testing.assert_close(ga, gb, atol=atol, rtol=rtol)


def test_weight_gradient_does_not_unpack_unused_weight():
    desc = (
        o3.Linear(
            "4x0e+4x1o", "3x0e+6x1o", internal_weights=False, shared_weights=False
        )
        .cuda()
        .double()
    )
    native = IndexedLinear(desc)
    x = torch.randn(11, native.input_dim, device="cuda", dtype=torch.float64)
    w = torch.randn(
        3, 1, desc.weight_numel, device="cuda", dtype=torch.float64
    ).requires_grad_()
    ids = torch.arange(11, device="cuda") % 3
    actual = native(x, w, ids)
    expected = reference(desc, x, w, ids, 1, "mul_ir")
    desired = torch.autograd.grad(expected.sum(), w)[0]
    with torch.no_grad():
        w.add_(1)
    torch.testing.assert_close(
        torch.autograd.grad(actual.sum(), w)[0], desired, atol=2e-11, rtol=2e-10
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
def test_multiple_input_output_paths(dtype, layout):
    torch.manual_seed(381)
    desc = (
        o3.Linear(
            "3x0e+7x0e+4x1o+5x1o+2x2e",
            "2x0e+9x0e+6x1o+3x1o+4x2e+2x3o",
            internal_weights=False,
            shared_weights=False,
        )
        .cuda()
        .to(dtype)
    )
    native = IndexedLinear(desc, 3, layout)
    x = torch.randn(67, native.input_dim, device="cuda", dtype=dtype).requires_grad_()
    w = torch.randn(
        3, 3, desc.weight_numel, device="cuda", dtype=dtype
    ).requires_grad_()
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    ids = torch.arange(67, device="cuda") % 3
    actual, expected = native(x, w, ids), reference(desc, xr, wr, ids, 3, layout)
    atol, rtol = (3e-5, 5e-5) if dtype == torch.float32 else (2e-11, 2e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    cotangent = torch.randn_like(actual)
    results = [
        differentiate(y, a, b, cotangent)
        for y, a, b in ((actual, x, w), (expected, xr, wr))
    ]
    for a, b in zip(*results, strict=True):
        for ga, gb in zip(a, b, strict=True):
            torch.testing.assert_close(ga, gb, atol=atol, rtol=rtol)


@pytest.mark.parametrize("layout", ["mul_ir", "ir_mul"])
@pytest.mark.parametrize("experts", [1, 3])
@pytest.mark.parametrize("nodes", [1025])
def test_fp64_tensor_core_irregular_channel_tails(layout, experts, nodes):
    # CI/CO are not multiples of the m16n8k4 fragment dimensions; coefficients
    # are irrational FP64 values and two paths share the same output group.
    torch.manual_seed(558)
    desc = (
        o3.Linear(
            "67x0e+65x0e+65x1o+69x2e",
            "69x0e+72x1o+75x2e+2x3o",
            internal_weights=False,
            shared_weights=False,
        )
        .cuda()
        .double()
    )
    native = IndexedLinear(desc, experts, layout)
    x = torch.randn(
        nodes, native.input_dim, device="cuda", dtype=torch.float64
    ).requires_grad_()
    w = torch.randn(
        1, experts, desc.weight_numel, device="cuda", dtype=torch.float64
    ).requires_grad_()
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    ids = torch.zeros(nodes, device="cuda", dtype=torch.int64)
    actual, expected = native(x, w, ids), reference(desc, xr, wr, ids, experts, layout)
    torch.testing.assert_close(actual, expected, atol=2e-11, rtol=2e-10)
    cotangent = torch.randn_like(actual)
    results = [
        differentiate(y, a, b, cotangent)
        for y, a, b in ((actual, x, w), (expected, xr, wr))
    ]
    for a, b in zip(*results, strict=True):
        for ga, gb in zip(a, b, strict=True):
            torch.testing.assert_close(ga, gb, atol=2e-11, rtol=2e-10)


@pytest.mark.parametrize("nested", [False, True])
def test_inference_mode(nested):
    desc = o3.Linear("4x0e+4x1o", "4x0e+4x1o", internal_weights=False)
    native = IndexedLinear(desc)
    x = torch.randn(3, native.input_dim, device="cuda", requires_grad=True)
    w = torch.randn(1, 1, desc.weight_numel, device="cuda", requires_grad=True)
    ids = torch.zeros(3, device="cuda", dtype=torch.int64)
    with torch.inference_mode(), torch.set_grad_enabled(nested):
        y = native(x, w, ids)
    assert y.is_inference() and not y.requires_grad and y.grad_fn is None


def test_empty_nodes_and_disconnected_map():
    desc = o3.Linear("3x1o", "2x0e", internal_weights=False)
    native = IndexedLinear(desc)
    for nodes in (3, 0):
        x = torch.randn(nodes, native.input_dim, device="cuda", requires_grad=True)
        w = torch.empty(1, 1, 0, device="cuda", requires_grad=True)
        ids = torch.zeros(nodes, device="cuda", dtype=torch.int64)
        y = native(x, w, ids)
        torch.testing.assert_close(y, torch.zeros_like(y))
        gx, gw = torch.autograd.grad(y.sum(), (x, w))
        torch.testing.assert_close(gx, torch.zeros_like(x))
        assert gw.shape == w.shape


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("two_paths", [False, True])
def test_large_shared_weight_reduction_budget(dtype, two_paths):
    # The actual TECE maps cross the 64 MiB scratch budget at E=3072. An
    # irregular extra row also checks the last adaptive reduction chunk.
    torch.manual_seed(2761)
    torch.backends.cuda.matmul.allow_tf32 = False
    nodes = 3073
    paths = [(0, 0, 0, 384, 576, 1, 384**-0.5)]
    if two_paths:
        paths.append((384, 221184, 576, 256, 384, 2, 0.0625))
    din, dout, numel = (896, 1344, 319488) if two_paths else (384, 576, 221184)
    metadata = repr((din, dout, numel, 1, "ir_mul", tuple(paths)))
    x = torch.randn(nodes, din, device="cuda", dtype=dtype).requires_grad_()
    w = torch.randn(1, 1, numel, device="cuda", dtype=dtype).requires_grad_()
    xr, wr = x.detach().clone().requires_grad_(), w.detach().clone().requires_grad_()
    ids = torch.zeros(nodes, device="cuda", dtype=torch.int64)
    actual = indexed_linear(metadata, x, w, ids)
    parts = []
    for sx, sw, _, ci, co, dim, scale in paths:
        feature = xr[:, sx : sx + ci * dim].reshape(nodes, dim, ci)
        weight = wr[0, 0, sw : sw + ci * co].reshape(ci, co)
        parts.append((feature @ weight * scale).flatten(1))
    expected = torch.cat(parts, dim=1)
    atol, rtol = (3e-5, 5e-5) if dtype == torch.float32 else (2e-11, 2e-10)
    torch.testing.assert_close(actual, expected, atol=atol, rtol=rtol)
    cotangent = torch.randn_like(actual)
    results = [
        differentiate(y, a, b, cotangent)
        for y, a, b in ((actual, x, w), (expected, xr, wr))
    ]
    for a, b in zip(*results, strict=True):
        for ga, gb in zip(a, b, strict=True):
            torch.testing.assert_close(ga, gb, atol=atol, rtol=rtol)
