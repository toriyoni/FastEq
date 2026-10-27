"""Fused native-layout Linear forward, selective backward and force losses."""

import ast

import pytest
import torch
from e3nn import o3

from fasteq.ops.indexed_linear import IndexedLinear
from fasteq.ops.inference_linear import try_inference_linear
import fasteq.ops.inference_linear as serial
from fasteq.triton import so3_linear

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Hopper inference dispatch",
)

IR = "64x0e+64x1o+64x2e"


@pytest.mark.parametrize("ir_in,ir_out", [
    (IR, IR), (IR, "192x0e+64x1o+64x2e"), (IR + "+" + IR, IR),
])
def test_native_equivalence_strides_tail_and_dispatch(ir_in, ir_out, monkeypatch):
    torch.manual_seed(773)
    module = o3.Linear(ir_in, ir_out).cuda()
    op = IndexedLinear(module)
    n = 16385  # Tail of both the node and angular tiles.
    storage = torch.randn(n, 2 * op.input_dim + 1, device="cuda")
    x = storage[:, 1::2]
    ws = torch.randn(1 + 2 * op.weight_numel, device="cuda")
    w = ws[1::2].view(1, 1, -1)
    ids = torch.zeros(n, device="cuda", dtype=torch.int64)
    calls = []
    original = serial._serial_plan

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(serial, "_serial_plan", record)
    monkeypatch.setenv("FASTEQ_LINEAR_INFERENCE", "1")
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        with torch.no_grad():
            torch.backends.cuda.matmul.allow_tf32 = False
            expected = module(x, w.flatten())
            torch.backends.cuda.matmul.allow_tf32 = True
            actual = op(x, w, ids)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)
            with torch.inference_mode():
                torch.testing.assert_close(op(x, w, ids), expected, atol=3e-5, rtol=5e-5)
        assert len(calls) == 2
        assert all(c[0] == op.metadata and c[2] == n for c in calls)
        with torch.no_grad():
            monkeypatch.setenv("FASTEQ_LINEAR_INFERENCE", "0")
            torch.testing.assert_close(op(x, w, ids), expected, atol=3e-5, rtol=5e-5)
        assert len(calls) == 2
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_graph_live_weights_nondefault_stream():
    module = o3.Linear(IR + "+" + IR, IR).cuda()
    op = IndexedLinear(module)
    x = torch.randn(16385, op.input_dim, device="cuda")
    w = module.weight.detach().view(1, 1, -1)
    ids = torch.zeros(x.shape[0], device="cuda", dtype=torch.int64)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.no_grad(), torch.cuda.stream(stream):
        for _ in range(3):
            op(x, w, ids)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.no_grad(), torch.cuda.graph(graph, stream=stream):
        actual = op(x, w, ids)
    with torch.no_grad():
        w.add_(0.1)
        x.mul_(0.7)
        graph.replay()
        torch.testing.assert_close(actual, module(x), atol=3e-5, rtol=5e-5)


@pytest.mark.parametrize("ir_in,ir_out", [
    (IR, IR), (IR, "192x0e+64x1o+64x2e"), (IR + "+" + IR, IR),
])
def test_grad_enabled_keeps_force_loss_chain(ir_in, ir_out):
    from test_indexed_linear_simt_v6 import chain

    torch.manual_seed(2710)
    module = o3.Linear(ir_in, ir_out).cuda()
    op = IndexedLinear(module)
    ids = torch.zeros(16385, device="cuda", dtype=torch.int64)
    source = [torch.randn(16385, 3, device="cuda"),
              torch.randn(3, op.input_dim, device="cuda") * 0.1,
              module.weight.detach().reshape(1, 1, -1)]

    results = []
    for use_fasteq in (False, True):
        leaves = [v.clone().requires_grad_() for v in source]
        results.append(chain(ast.literal_eval(op.metadata), *leaves, ids, use_fasteq))
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)


def test_unsupported_cases_retain_existing_dispatch():
    module = o3.Linear(IR, IR).cuda()
    op = IndexedLinear(module)
    x = torch.randn(16384, op.input_dim, device="cuda")
    w = module.weight.detach().view(1, 1, -1)
    ids = torch.zeros(x.shape[0], device="cuda", dtype=torch.int64)
    # Detached inputs also work while grad mode is enabled.
    assert try_inference_linear(op.metadata, x, w, ids) is not None
    with torch.no_grad():
        assert try_inference_linear(op.metadata, x[:10], w, ids[:10]) is None
        assert try_inference_linear(op.metadata, x.double(), w.double(), ids) is None
        assert try_inference_linear(op.metadata, x, w.expand(2, -1, -1), ids) is None
        angular = IndexedLinear(module, layout="ir_mul")
        assert try_inference_linear(angular.metadata, x, w, ids) is None
        scalar = IndexedLinear(o3.Linear("64x0e", "64x0e").cuda())
        assert try_inference_linear(scalar.metadata, x[:, :64], w[:, :, :4096], ids) is None


def test_native_layout_rotation_equivariance():
    from fasteq.ops.inference_linear import _layout

    module = o3.Linear(IR + "+" + IR, IR).cuda()
    op = IndexedLinear(module)
    layout = _layout(op.metadata)
    rotation = o3.rand_matrix()
    din = module.irreps_in.D_from_matrix(rotation).cuda()
    dout = module.irreps_out.D_from_matrix(rotation).cuda()
    x = torch.randn(257, op.input_dim, device="cuda")
    with torch.no_grad():
        y = so3_linear.so3_linear_serial_cta(module.weight, x, layout, feature_layout="mul_ir")
        rotated = so3_linear.so3_linear_serial_cta(
            module.weight, x @ din.T, layout, feature_layout="mul_ir")
        torch.testing.assert_close(rotated, y @ dout.T, atol=3e-5, rtol=5e-5)


def test_upstream_default_layout_preserved():
    from fasteq.ops.inference_linear import _layout

    module = o3.Linear(IR, IR).cuda()
    op = IndexedLinear(module)
    x = torch.randn(257, op.input_dim, device="cuda")

    def angular_first(value, irreps):
        return torch.cat([value[:, sl].reshape(value.shape[0], mul, ir.dim)
                          .transpose(1, 2).flatten(1)
                          for (mul, ir), sl in zip(irreps, irreps.slices())], dim=1)

    with torch.no_grad():
        expected = angular_first(module(x), module.irreps_out)
        actual = so3_linear.so3_linear_serial_cta(
            module.weight, angular_first(x, module.irreps_in), _layout(op.metadata))
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)


def test_optional_triton_keeps_cuda_fallback(monkeypatch):
    module = o3.Linear(IR, IR).cuda()
    op = IndexedLinear(module)
    x = torch.randn(16384, op.input_dim, device="cuda")
    w = module.weight.detach().view(1, 1, -1)
    ids = torch.zeros(x.shape[0], device="cuda", dtype=torch.int64)
    monkeypatch.setattr(so3_linear, "triton", None)
    with torch.no_grad():
        assert try_inference_linear(op.metadata, x, w, ids) is None
        torch.testing.assert_close(op(x, w, ids), module(x), atol=3e-5, rtol=5e-5)


@pytest.mark.parametrize("ir_in,ir_out", [
    (IR, IR), (IR, "192x0e+64x1o+64x2e"), (IR + "+" + IR, IR),
])
def test_input_and_weight_gradients_strided(ir_in, ir_out):
    torch.manual_seed(845)
    module = o3.Linear(ir_in, ir_out).cuda()
    op = IndexedLinear(module)
    n = 16385
    x = torch.randn(n, op.input_dim * 2 + 1, device="cuda")[:, 1::2].requires_grad_()
    w = torch.randn(op.weight_numel * 2 + 1, device="cuda")[1::2].view(1, 1, -1).requires_grad_()
    ids = torch.zeros(n, device="cuda", dtype=torch.int64)
    g = (torch.randn(n, op.output_dim * 2 + 1, device="cuda") * 0.02)[:, 1::2]
    actual = op(x, w, ids)
    assert actual.grad_fn.name() == "FastEqIndexedLinearBackward"
    expected = module(x, w.flatten())
    ag = torch.autograd.grad(actual, (x, w), g)
    eg = torch.autograd.grad(expected, (x, w), g)
    for a, b in zip(ag, eg, strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=5e-5)


@pytest.mark.parametrize("selected", ["x", "w"])
def test_selective_gradient_does_not_unpack_other_operand(selected):
    module = o3.Linear(IR, IR).cuda()
    op = IndexedLinear(module)
    x = torch.randn(16385 * op.input_dim + 1, device="cuda")[1:].view(16385, -1).requires_grad_()
    w = torch.randn(op.weight_numel + 1, device="cuda")[1:].view(1, 1, -1).requires_grad_()
    ids = torch.zeros(x.shape[0], device="cuda", dtype=torch.int64)
    target = x if selected == "x" else w
    # dX uses W but not X; dW uses X but not W. Mutating the destination's
    # corresponding forward operand must not trigger its saved version check.
    mutate = x if selected == "x" else w
    expected = torch.autograd.grad(module(x, w.flatten()).sum(), target)[0]
    actual = op(x, w, ids).sum()
    with torch.no_grad():
        mutate.add_(0.1)
    torch.testing.assert_close(torch.autograd.grad(actual, target)[0], expected,
                               atol=2e-4, rtol=5e-5)


def test_first_backward_graph_live_weights():
    module = o3.Linear(IR, "192x0e+64x1o+64x2e").cuda()
    op = IndexedLinear(module)
    x = torch.randn(16385, op.input_dim, device="cuda", requires_grad=True)
    w = module.weight.detach().view(1, 1, -1).requires_grad_()
    ids = torch.zeros(x.shape[0], device="cuda", dtype=torch.int64)
    g = torch.randn(x.shape[0], op.output_dim, device="cuda") * 0.02
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            torch.autograd.grad(op(x, w, ids), (x, w), g)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = torch.autograd.grad(op(x, w, ids), (x, w), g)
    with torch.no_grad():
        x.mul_(0.6)
        w.add_(0.1)
    graph.replay()
    expected = torch.autograd.grad(module(x, w.flatten()), (x, w), g)
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, atol=1e-4, rtol=5e-5)


@pytest.mark.parametrize("feature_layout", ["ir_mul", "mul_ir"])
def test_raw_transpose_broadcast_cotangent(feature_layout):
    from test_indexed_linear_simt_v6 import reference

    module = o3.Linear(IR, IR).cuda()
    op = IndexedLinear(module, layout=feature_layout)
    metadata = ast.literal_eval(op.metadata)
    layout = serial._layout(IndexedLinear(module).metadata)
    x = torch.randn(257, op.input_dim, device="cuda", requires_grad=True)
    w = module.weight.detach().view(1, 1, -1)
    ids = torch.zeros(x.shape[0], device="cuda", dtype=torch.int64)
    g = torch.ones(1, device="cuda").expand(257, op.output_dim)
    expected = torch.autograd.grad(reference(metadata, x, w, ids), x, g)[0]
    actual = so3_linear.so3_linear_backward_x(
        w.flatten(), g, layout, feature_layout=feature_layout)
    torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)
