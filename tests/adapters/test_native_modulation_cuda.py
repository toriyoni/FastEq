"""Native modulation host plan: selective higher-order AD and lifetime checks."""

import copy
import gc

import pytest
import torch
from e3nn import o3
from eqx_cuda_spy import forbid_eqx_cuda
from fasteq.ops.modulation import ModulatedResidualPlan, _NativeCompiler

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def make_plan(monkeypatch, enabled=True):
    monkeypatch.setenv("FASTEQ_EXPERIMENTAL_NATIVE_MODULATION", "1" if enabled else "0")
    return ModulatedResidualPlan(o3.Irreps("2x0e+3x1o+1x2e"))


def operands(plan, dtype=torch.float64):
    return [
        torch.randn(5, width, device="cuda", dtype=dtype, requires_grad=True)
        for width in (plan.dim, plan.channels, plan.dim)
    ]


def reference(plan, values):
    index = torch.tensor(plan._index, device=values[0].device)
    return values[2] + values[0] * values[1].index_select(-1, index)


def compare(a, b, dtype):
    if a is None or b is None:
        other = a if b is None else b
        assert other is None or torch.count_nonzero(other) == 0
        return
    torch.testing.assert_close(
        a,
        b,
        atol=3e-4 if dtype == torch.float32 else 2e-10,
        rtol=3e-3 if dtype == torch.float32 else 2e-9,
    )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("targets", [(0,), (1,), (0, 1), (2,), (0, 1, 2)])
def test_native_selective_missing_seeds(monkeypatch, dtype, targets):
    plan = make_plan(monkeypatch)
    left = operands(plan, dtype)
    right = [x.detach().clone().requires_grad_() for x in left]
    seed = torch.randn_like(left[0], requires_grad=True)
    seed_ref = seed.detach().clone().requires_grad_()
    output = plan(*left)
    assert output.grad_fn.name() == "FastEqModulationBackward"
    expected = reference(plan, right)
    gradients = []
    for value, inputs, weight in ((output, left, seed), (expected, right, seed_ref)):
        first = torch.autograd.grad(
            value, [inputs[i] for i in targets], weight, create_graph=True
        )
        if targets[0] != 2:
            assert first[0].grad_fn is not None
        loss = sum(g.square().sum() for g in first)
        second = torch.autograd.grad(loss, (*inputs, weight), allow_unused=True)
        gradients.append((first, second))
    for actual, expected in zip(*gradients, strict=True):
        for a, b in zip(actual, expected, strict=True):
            compare(a, b, dtype)
    native = plan._native[(left[0].device, dtype)]
    assert native.cached_executables >= (1 if targets == (2,) else 3)


def force_chain(plan, source, native):
    pos, projection, gate_weight, skip_weight = source
    x, gates, residual = pos @ projection, (pos @ gate_weight).sin(), pos @ skip_weight
    y = plan(x, gates, residual) if native else reference(plan, (x, gates, residual))
    energy = y.tanh().square().sum()
    force = -torch.autograd.grad(energy, pos, create_graph=True)[0]
    grads = torch.autograd.grad(energy + force.square().sum(), source)
    return (y, energy, force, *grads)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_full_force_loss_and_spy(monkeypatch, dtype):
    plan = make_plan(monkeypatch)
    source = [
        torch.randn(shape, device="cuda", dtype=dtype, requires_grad=True)
        for shape in ((7, 3), (3, plan.dim), (3, plan.channels), (3, plan.dim))
    ]
    expected = force_chain(plan, source, False)
    actual = force_chain(plan, source, True)
    assert actual[0].grad_fn.name() == "FastEqModulationBackward"
    assert next(iter(plan._native.values())).cached_executables >= 3
    for a, b in zip(actual, expected, strict=True):
        compare(a, b, dtype)
    with forbid_eqx_cuda():
        fallback = force_chain(plan, source, True)
    assert fallback[0].grad_fn.name() == "_ModulationBackward"
    for a, b in zip(fallback, expected, strict=True):
        compare(a, b, dtype)


@pytest.mark.parametrize("stage", [0, 1, 2])
def test_native_unsupported_abi_fallback(monkeypatch, stage):
    plan = make_plan(monkeypatch)
    original = _NativeCompiler.prepare

    def unsupported(self, selected, arguments, requested, seeds):
        return (
            None
            if selected == stage
            else original(self, selected, arguments, requested, seeds)
        )

    monkeypatch.setattr(_NativeCompiler, "prepare", unsupported)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float64, requires_grad=True)
        for shape in ((3, 3), (3, plan.dim), (3, plan.channels), (3, plan.dim))
    ]
    expected = force_chain(plan, source, False)
    actual = force_chain(plan, source, True)
    assert actual[0].grad_fn.name() == (
        "_ModulationBackward" if stage == 0 else "FastEqModulationBackward"
    )
    for a, b in zip(actual, expected, strict=True):
        compare(a, b, torch.float64)


@pytest.mark.parametrize("mutated", ["x", "g", "dy"])
def test_native_saved_versions(monkeypatch, mutated):
    plan = make_plan(monkeypatch)
    x, gates, residual = operands(plan)
    output = plan(x, gates, residual)
    dy = torch.randn_like(x, requires_grad=True)
    if mutated == "dy":
        dx, dg = torch.autograd.grad(output, (x, gates), dy, create_graph=True)
        with torch.no_grad():
            dy.add_(1)
        action = lambda: torch.autograd.grad(
            dx.square().sum() + dg.square().sum(), (x, gates, dy)
        )
    else:
        with torch.no_grad():
            (x if mutated == "x" else gates).add_(1)
        action = lambda: torch.autograd.grad(output, (x, gates), dy)
    with pytest.raises(RuntimeError, match="modified by an inplace operation"):
        action()


def test_native_graph_outlives_python_plan(monkeypatch):
    plan = make_plan(monkeypatch)
    x, gates, residual = operands(plan)
    y = plan(x, gates, residual)
    expected = reference(plan, (x, gates, residual))
    copied = copy.deepcopy(plan)
    assert not copied._native and not copied._launchers and not copied._cache
    del plan
    gc.collect()
    dy = torch.randn_like(x, requires_grad=True)
    actual = torch.autograd.grad(y, (x, gates), dy, create_graph=True)
    expected_first = torch.autograd.grad(expected, (x, gates), dy, create_graph=True)
    # First and double kernels compile after deletion, on an autograd worker.
    actual_second = torch.autograd.grad(
        sum(g.square().sum() for g in actual), (x, gates, dy)
    )
    expected_second = torch.autograd.grad(
        sum(g.square().sum() for g in expected_first), (x, gates, dy)
    )
    for a, b in zip(actual_second, expected_second, strict=True):
        compare(a, b, torch.float64)


def test_native_stream_capture_second_and_live_inputs(monkeypatch):
    plan = make_plan(monkeypatch)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float64, requires_grad=True)
        for shape in ((5, 3), (3, plan.dim), (3, plan.channels), (3, plan.dim))
    ]
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            force_chain(plan, source, True)
    torch.cuda.current_stream().wait_stream(stream)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = force_chain(plan, source, True)
    with torch.no_grad():
        source[1].mul_(0.7)
        source[2].add_(0.1)
    graph.replay()
    expected = force_chain(plan, source, False)
    for a, b in zip(actual, expected, strict=True):
        compare(a, b, torch.float64)


def test_native_enable_flag_is_plan_configuration(monkeypatch):
    disabled = make_plan(monkeypatch, False)
    enabled = make_plan(monkeypatch, True)
    monkeypatch.setenv("FASTEQ_EXPERIMENTAL_NATIVE_MODULATION", "0")
    values = operands(disabled)
    a, b = disabled(*values), enabled(*values)
    assert a.grad_fn.name() == "_ModulationBackward"
    assert b.grad_fn.name() == "FastEqModulationBackward"
    compare(a, b, torch.float64)


@pytest.mark.parametrize("enabled", [False, True])
def test_modulation_grad_modes_empty_and_hot_cache(monkeypatch, enabled):
    plan = make_plan(monkeypatch, enabled)
    values = operands(plan)
    for context in (torch.no_grad, torch.inference_mode):
        with context():
            actual = plan(*values)
            expected = reference(plan, values)
        assert not actual.requires_grad
        compare(actual, expected, torch.float64)
    empty = [value[:0] for value in values]
    actual = plan(*empty)
    first = torch.autograd.grad(actual.sum(), empty, create_graph=True)
    assert all(value.numel() == 0 for value in first)
    source = [
        torch.randn(shape, device="cuda", dtype=torch.float64, requires_grad=True)
        for shape in ((3, 3), (3, plan.dim), (3, plan.channels), (3, plan.dim))
    ]
    force_chain(plan, source, True)
    if enabled:
        count = next(iter(plan._native.values())).cached_executables

        def forbidden(*args, **kwargs):
            raise AssertionError("Hot native execution entered the Python compiler")

        monkeypatch.setattr(_NativeCompiler, "prepare", forbidden)
    actual = force_chain(plan, source, True)
    expected = force_chain(plan, source, False)
    for a, b in zip(actual, expected, strict=True):
        compare(a, b, torch.float64)
    if enabled:
        assert next(iter(plan._native.values())).cached_executables == count


def test_raw_native_plan_rejects_compiler_dtype_mismatch(monkeypatch):
    plan = make_plan(monkeypatch)
    values = operands(plan, torch.float64)
    compiler = _NativeCompiler(plan, values[0])
    from fasteq.cuda.jit import runtime

    raw = runtime().ModulationPlan(compiler._index, plan.dim, plan.channels, compiler)
    # The very first invocation must reject a wrong dtype, before any cached
    # executable exists. An FP64 cubin must never see FP32 operand pointers.
    wrong = [value.detach().float() for value in values]
    with pytest.raises(
        RuntimeError, match="dtype does not match its prepared compiler"
    ):
        raw.forward(*wrong)
    assert raw.cached_executables == 0
    compare(raw.forward(*values), reference(plan, values), torch.float64)
    with pytest.raises(
        RuntimeError, match="dtype does not match its prepared compiler"
    ):
        raw.forward(*wrong)
    assert raw.cached_executables == 1
