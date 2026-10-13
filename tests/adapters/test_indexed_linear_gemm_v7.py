"""Shared CUDA GEMM storage offsets, full precision, and physical adjoints."""

from pathlib import Path

import pytest
import torch
from test_indexed_linear_simt_v6 import chain, reference

import fasteq
from fasteq.ops.indexed_linear import _plan, indexed_linear

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def opt_in(monkeypatch):
    assert Path(fasteq.__file__).is_relative_to(Path(__file__).parents[2])
    monkeypatch.setenv("FASTEQ_EXPERIMENTAL_LINEAR_GEMM", "1")
    _plan.cache_clear()
    yield
    _plan.cache_clear()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout", ["ir_mul", "mul_ir"])
def test_storage_offsets_and_physical_parameter_chain(dtype, layout):
    torch.manual_seed(7381)
    torch.backends.cuda.matmul.allow_tf32 = False
    experts, nodes = 3, 1025
    paths = ((0, 0, 0, 65, 67, 1, 65**-0.5),
             (65 * experts, 4355, 67 * experts, 69, 71, 2, 69**-0.5))
    description = (203 * experts, 209 * experts, 9254, experts, layout, paths)
    pos = torch.randn(nodes, 3, device="cuda", dtype=dtype)
    projection = torch.randn(3, description[0], device="cuda", dtype=dtype) * 0.1
    storage = torch.randn(1 + experts * 9254, device="cuda", dtype=dtype)
    weight = storage[1:].view(1, experts, 9254)
    ids = torch.zeros(nodes, dtype=torch.int64, device="cuda")
    results = []
    for native in (False, True):
        leaves = [value.detach().requires_grad_() for value in
                  (pos.clone(), projection.clone(), weight if native else weight.clone())]
        results.append(chain(description, *leaves, ids, native))
    tolerance = {"atol": 3e-5, "rtol": 5e-5} if dtype == torch.float32 else {"atol": 2e-11, "rtol": 2e-10}
    for actual, expected in zip(results[1], results[0], strict=True):
        torch.testing.assert_close(actual, expected, **tolerance)
    # A contiguous tensor can still start inside its storage. Both feature
    # and weight views must add storage_offset, not assume allocation base.
    feature = torch.randn(nodes * description[0] + 1, device="cuda", dtype=dtype)[1:].view(nodes, description[0])
    assert feature.is_contiguous() and feature.storage_offset() == 1
    torch.testing.assert_close(indexed_linear(repr(description), feature, weight, ids),
                               reference(description, feature, weight, ids), **tolerance)


def test_fp32_does_not_follow_global_tf32_opt_in():
    torch.manual_seed(6612)
    description = (129, 131, 16899, 1, "ir_mul", ((0, 0, 0, 129, 131, 1, 129**-0.5),))
    x = torch.randn(1025, 129, device="cuda")
    w = torch.randn(1, 1, 16899, device="cuda")
    ids = torch.zeros(1025, dtype=torch.int64, device="cuda")
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        expected = reference(description, x, w, ids)
        torch.backends.cuda.matmul.allow_tf32 = True
        actual = indexed_linear(repr(description), x, w, ids)
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


def test_selected_weight_gradient_does_not_unpack_mutated_weight():
    description = (65, 67, 4355, 1, "ir_mul", ((0, 0, 0, 65, 67, 1, 65**-0.5),))
    x = torch.randn(1025, 65, device="cuda", dtype=torch.float64)
    w = torch.randn(1, 1, 4355, device="cuda", dtype=torch.float64, requires_grad=True)
    ids = torch.zeros(1025, dtype=torch.int64, device="cuda")
    expected = torch.autograd.grad(reference(description, x, w, ids).sum(), w)[0]
    actual = indexed_linear(repr(description), x, w, ids)
    with torch.no_grad():
        w.add_(1)
    torch.testing.assert_close(torch.autograd.grad(actual.sum(), w)[0], expected,
                               atol=2e-11, rtol=2e-10)


def test_batched_weight_adjoint_disables_tf32():
    description = (130, 134, 4355, 1, "ir_mul", ((0, 0, 0, 65, 67, 2, 65**-0.5),))
    x = torch.randn(8193, 130, device="cuda", requires_grad=True)
    w = torch.randn(1, 1, 4355, device="cuda", requires_grad=True)
    dy = torch.randn(8193, 134, device="cuda") * 0.01
    ids = torch.zeros(8193, dtype=torch.int64, device="cuda")
    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        expected = torch.autograd.grad(reference(description, x, w, ids), w, dy)[0]
        torch.backends.cuda.matmul.allow_tf32 = True
        actual = torch.autograd.grad(indexed_linear(repr(description), x, w, ids), w, dy)[0]
        torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous


@pytest.mark.parametrize("dtype,nodes", [(torch.float32, 8193), (torch.float64, 1025)])
def test_nondefault_stream_force_loss_graph_live_weights(dtype, nodes):
    torch.manual_seed(982)
    torch.backends.cuda.matmul.allow_tf32 = False
    description = (65, 67, 4355, 1, "ir_mul", ((0, 0, 0, 65, 67, 1, 65**-0.5),))
    source = [torch.randn(shape, dtype=dtype, device="cuda", requires_grad=True)
              for shape in ((nodes, 3), (3, 65), (1, 1, 4355))]
    ids = torch.zeros(nodes, dtype=torch.int64, device="cuda")
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            chain(description, *source, ids, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = chain(description, *source, ids, True)
    with torch.no_grad():
        source[1].mul_(0.7)
        source[2].add_(0.1)
    graph.replay()
    leaves = [value.detach().clone().requires_grad_() for value in source]
    expected = chain(description, *leaves, ids, False)
    tolerance = {"atol": 3e-5, "rtol": 5e-5} if dtype == torch.float32 else {"atol": 2e-11, "rtol": 2e-10}
    for a, b in zip(actual, expected, strict=True):
        torch.testing.assert_close(a, b, **tolerance)
