"""Scalar-Y adapter layout, shared weights and physical mixed force losses."""
import os
from pathlib import Path
import sys

import pytest
import torch
from e3nn import o3
from fasteq.adapters import adapt_model
from fasteq.ops import segmented_product as product

sys.path.insert(0, str(Path(__file__).parents[2] / "benchmarks/adapters"))
from benchmark_scaling_case import ExclusiveGuard
from eqx_cuda_spy import forbid_eqx_cuda

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def exclusive_scalar_y(monkeypatch):
    guard = ExclusiveGuard(os.environ["CUDA_VISIBLE_DEVICES"])
    guard.check()
    torch.set_num_threads(1)
    torch.manual_seed(73129)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    for suffix in ("TASK_MASK", "REDUCE", "GEMM", "PAIR"):
        monkeypatch.setenv("FASTEQ_EXPERIMENTAL_SEGMENTED_" + suffix, "1")
    monkeypatch.setenv("FASTEQ_EXPERIMENTAL_UNIFORM_SCALAR_Y", "1")
    yield
    torch.cuda.synchronize()
    guard.check()


def make_module(kind, dtype, *, external=False, mixed=False, unweighted=False, channels=7):
    weighted = not unweighted
    if kind == "o2":
        from eqx.o2 import TensorProduct
        return TensorProduct(
            f"{channels}x1me", "1x0ee", f"{channels}x1me",
            [(0, 0, 0, "u1u", weighted)],
            internal_weights=not external and weighted,
            shared_weights=not external,
        ).to(device="cuda", dtype=dtype)
    paths = [(0, j, j, "uvu", weighted and (not mixed or j != 2)) for j in range(3)]
    return o3.TensorProduct(
        f"{channels}x0e", "1x0e+1x1o+1x2e",
        f"{channels}x0e+{channels}x1o+{channels}x2e", paths,
        internal_weights=not external and weighted,
        shared_weights=not external,
    ).to(device="cuda", dtype=dtype)


def close(left, right, dtype):
    tolerance = (dict(atol=3e-5, rtol=5e-5) if dtype == torch.float32
                 else dict(atol=2e-10, rtol=2e-9))
    for a, b in zip(left, right, strict=True):
        if a is None or b is None:
            present = b if a is None else a
            assert present is None or not torch.count_nonzero(present)
        else:
            torch.testing.assert_close(a, b, **tolerance)


def chain(module, args, *, subset=None):
    leaves = tuple(args) + tuple(module.parameters())
    value = module(*args)
    objective = (value.sin() + 0.1 * value.square()).sum()
    selected = leaves if subset is None else tuple(leaves[i] for i in subset)
    first = torch.autograd.grad(objective, selected, create_graph=True, allow_unused=True)
    terms = [g.square().sum() for g in first if g is not None and g.requires_grad]
    higher = (torch.autograd.grad(sum(terms), leaves, allow_unused=True)
              if terms else (None,) * len(leaves))
    return (value, *first, *higher)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind,external,mixed,unweighted,rows", [
    ("o2", False, False, False, 1025),
    ("o2", True, False, False, 17),
    ("o2", False, False, True, 17),
    ("o3", False, False, False, 17),
    ("o3", True, True, False, 17),
    ("o3", False, False, True, 17),
])
def test_native_layout_all_weight_domains(dtype, kind, external, mixed, unweighted, rows):
    original = make_module(kind, dtype, external=external, mixed=mixed, unweighted=unweighted)
    candidate = adapt_model(original)
    assert candidate._fasteq_plan.scalar_y_small
    args = [torch.randn(width, rows, device="cuda", dtype=dtype).T.requires_grad_()
            for width in (original.irreps_in1.dim, original.irreps_in2.dim)]
    if external:
        args.append(torch.randn(original.weight_numel, rows, device="cuda", dtype=dtype).T.requires_grad_())
    close(chain(candidate, args), chain(original, args), dtype)
    close(chain(candidate, args, subset=(0,)), chain(original, args, subset=(0,)), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_shared_weight_stays_one_row_in_train_and_inference(dtype, monkeypatch):
    module = adapt_model(make_module("o2", dtype, channels=64))
    observed = []
    actual = product.segmented_product

    def record(metadata, w, x, y):
        observed.append(tuple(w.shape))
        return actual(metadata, w, x, y)

    monkeypatch.setattr(product, "segmented_product", record)
    args = [torch.randn(1025, width, device="cuda", dtype=dtype).requires_grad_()
            for width in (module.irreps_in1.dim, module.irreps_in2.dim)]
    with torch.inference_mode():
        module(*args)
    chain(module, args, subset=(0, 1))
    assert observed == [(1, 64), (1, 64)]


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_external_broadcast_and_storage_offsets(dtype):
    original = make_module("o3", dtype, external=True)
    candidate = adapt_model(original)
    shapes = ((2, 1, original.irreps_in1.dim), (1, 3, original.irreps_in2.dim),
              (2, 1, original.weight_numel))
    args = [torch.randn(torch.tensor(shape).prod().item() + 1, device="cuda", dtype=dtype)[1:]
            .reshape(shape).requires_grad_() for shape in shapes]
    close(chain(candidate, args), chain(original, args), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["o2", "o3"])
def test_coordinate_dependent_weight_physical_loss(dtype, kind):
    original = make_module(kind, dtype, external=kind == "o3", channels=64)
    candidate = adapt_model(original)
    n = 33
    coords = torch.randn(n, 3, device="cuda", dtype=dtype).requires_grad_()
    widths = (original.irreps_in1.dim, original.irreps_in2.dim, original.weight_numel)
    projections = [torch.randn(3, d, device="cuda", dtype=dtype).mul_(0.03).requires_grad_()
                   for d in widths]

    def run(module):
        x, y = (coords @ projections[0]).sin(), (coords @ projections[1]).cos()
        base = coords if kind == "o3" else coords.mean(0, keepdim=True)
        w = (base @ projections[2]).sin()
        if kind == "o2":
            w = w + module.weight
        value = module(x, y, w)
        energy = (value.sin() + 0.1 * value.square()).sum()
        force = -torch.autograd.grad(energy, coords, create_graph=True)[0]
        leaves = (coords, *projections, *tuple(module.parameters()))
        gradients = torch.autograd.grad((force - 0.13).square().mean(), leaves, allow_unused=True)
        return (value, force, *gradients)

    close(run(candidate), run(original), dtype)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_empty_and_cg_state_load(dtype):
    original = make_module("o3", dtype)
    candidate = adapt_model(original)
    empty = [torch.empty(0, width, device="cuda", dtype=dtype).requires_grad_()
             for width in (original.irreps_in1.dim, original.irreps_in2.dim)]
    close(chain(candidate, empty), chain(original, empty), dtype)
    args = [torch.randn(17, width, device="cuda", dtype=dtype).requires_grad_()
            for width in (original.irreps_in1.dim, original.irreps_in2.dim)]
    candidate(*args)
    with torch.no_grad():
        original.weight.add_(0.07)
        for name, buffer in original.named_buffers():
            if "_w3j_" in name:
                buffer.mul_(0.73)
    candidate.load_state_dict(original.state_dict())
    close(chain(candidate, args), chain(original, args), dtype)


def test_nondefault_stream_graph_live_weight_no_eqx():
    dtype = torch.float64
    original = make_module("o2", dtype, channels=64)
    candidate = adapt_model(original)
    args = [torch.randn(17, width, device="cuda", dtype=dtype).requires_grad_()
            for width in (original.irreps_in1.dim, original.irreps_in2.dim)]
    expected = chain(original, args)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream), forbid_eqx_cuda() as calls:
        for _ in range(3):
            actual = chain(candidate, args)
    torch.cuda.current_stream().wait_stream(stream)
    assert not calls
    close(actual, expected, dtype)
    # Give capture its own static leaves and ordinary warmup. The dispatch
    # spy is an eager instrumentation check, separate from captured execution.
    captured_module = adapt_model(original)
    captured_args = [value.detach().clone().requires_grad_() for value in args]
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            chain(captured_module, captured_args)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        recorded = chain(captured_module, captured_args)
    with torch.no_grad():
        original.weight.mul_(0.81)
        captured_module.weight.copy_(original.weight)
        captured_args[0].add_(0.02)
    graph.replay()
    close(recorded, chain(original, captured_args), dtype)


def test_legacy_uuu_remains_legacy(monkeypatch):
    original = o3.TensorProduct("7x0e", "7x0e", "7x0e", [(0, 0, 0, "uuu", True)]).cuda().double()
    candidate = adapt_model(original)
    assert not candidate._fasteq_plan.scalar_y_small
    monkeypatch.setattr(product, "segmented_product", lambda *args: pytest.fail("uuu was rerouted"))
    # No new legacy JIT compilation is needed merely to verify the route.
    empty = [torch.empty(0, 7, device="cuda", dtype=torch.float64).requires_grad_() for _ in range(2)]
    close(chain(candidate, empty), chain(original, empty), torch.float64)


def test_o3_small_scalar_y_keeps_faster_uniform_route(monkeypatch):
    original = make_module("o3", torch.float64)
    candidate = adapt_model(original)
    assert candidate._fasteq_plan.scalar_y_small and not candidate._fasteq_plan.o2
    monkeypatch.setattr(product, "segmented_product", lambda *args: pytest.fail("O3 was rerouted"))
    empty = [torch.empty(0, width, device="cuda", dtype=torch.float64).requires_grad_()
             for width in (original.irreps_in1.dim, original.irreps_in2.dim)]
    close(chain(candidate, empty), chain(original, empty), torch.float64)
