"""Cover large FP64 rematerialized GateNorm and higher-order taskmasks."""

import pytest
import torch
from eqx_cuda_spy import forbid_eqx_cuda
from test_fasteq_gate_norm_cuda import close, modules

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("targets", ["input", "gamma", "all"])
def test_direct_first_and_second_seeds(targets):
    torch.manual_seed(823)
    gate, norm, _, plan = modules(torch.float64, 64)
    x = torch.randn(
        2, 32, plan.dim, device="cuda", dtype=torch.float64, requires_grad=True
    )
    skip = torch.randn_like(x, requires_grad=True)
    xr = x.detach().clone().requires_grad_()
    sr = skip.detach().clone().requires_grad_()
    gamma = tuple(norm.parameters())
    assert plan.fused.rematerialize_adjoints(x.reshape(-1, plan.dim))
    launches = []
    original = plan.fused.launch

    def launch(fn, args, value, **kwargs):
        launches.append((fn.__name__, kwargs))
        return original(fn, args, value, **kwargs)

    plan.fused.launch = launch
    actual = plan(x, skip)
    reference = norm(gate((xr + sr).reshape(-1, plan.dim))).reshape_as(actual)
    seed = torch.randn_like(actual, requires_grad=True)
    seedr = seed.detach().clone().requires_grad_()
    wanted = (
        (x,)
        if targets == "input"
        else gamma[::2]
        if targets == "gamma"
        else (x, skip, *gamma)
    )
    wantedr = (
        (xr,)
        if targets == "input"
        else gamma[::2]
        if targets == "gamma"
        else (xr, sr, *gamma)
    )
    first = torch.autograd.grad(actual, wanted, seed, create_graph=True)
    firstr = torch.autograd.grad(reference, wantedr, seedr, create_graph=True)
    directions = tuple(torch.randn_like(value) for value in first)
    second = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(first, directions)),
        (x, skip, seed, *gamma),
        allow_unused=True,
    )
    secondr = torch.autograd.grad(
        sum((g * d).sum() for g, d in zip(firstr, directions)),
        (xr, sr, seedr, *gamma),
        allow_unused=True,
    )
    close(actual, reference, torch.float64)
    for a, b in zip(first, firstr):
        close(a, b, torch.float64, 5)
    for a, b in zip(second, secondr):
        if b is None:
            assert a is None or not torch.count_nonzero(a)
        else:
            close(a, b, torch.float64, 15)
    assert any(name == "double_backward_direct" for name, _ in launches)
    if targets == "gamma":
        assert not any(name == "backward_direct" for name, _ in launches)
        double = next(
            flags for name, flags in launches if name == "double_backward_direct"
        )
        assert not double["HAS_U"] and double["HFLAGS"] == (True, False, True)
    else:
        assert any(name == "backward_direct" for name, _ in launches)


@pytest.mark.parametrize("affine", [False, True])
def test_direct_indirect_force_loss_and_parameters(affine):
    torch.manual_seed(829)
    gate, norm, _, plan = modules(torch.float64, 64, affine=affine)
    x = torch.randn(
        64, plan.dim, device="cuda", dtype=torch.float64, requires_grad=True
    )
    weight = torch.randn_like(x, requires_grad=True)
    xr, wr = (
        x.detach().clone().requires_grad_(),
        weight.detach().clone().requires_grad_(),
    )
    gamma = tuple(norm.parameters())

    def execute(x, weight, fn):
        y = fn(x * weight, x.sin() * weight.square() / 9)
        energy = y.sin().sum() + y.square().sum() / 7
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        loss = force.square().mean() + energy.square() / 100
        gradients = torch.autograd.grad(loss, (x, weight, *gamma))
        return y, energy, force, loss, gradients

    reference = execute(xr, wr, lambda p, s: norm(gate(p + s)))
    actual = execute(x, weight, plan)
    for a, b in zip(actual[:4], reference[:4]):
        close(a, b, torch.float64, 5)
    for a, b in zip(actual[4], reference[4]):
        close(a, b, torch.float64, 30)
    assert any(
        key[0].__name__ == "double_backward_direct" for key in plan.fused._launchers
    )
    with forbid_eqx_cuda():
        guarded = execute(x, weight, plan)
    for a, b in zip(guarded[:4], reference[:4]):
        close(a, b, torch.float64, 5)
    for a, b in zip(guarded[4], reference[4]):
        close(a, b, torch.float64, 30)


def test_direct_offset_stream_and_full_double_cuda_graph():
    torch.manual_seed(839)
    gate, norm, _, plan = modules(torch.float64, 64)
    backing = torch.randn(1 + 64 * plan.dim, device="cuda", dtype=torch.float64)
    x = backing[1:].view(64, plan.dim).requires_grad_()
    assert x.data_ptr() % 16 == 8
    parameters = (x, *norm.parameters())

    def execute(fn, x, parameters):
        y = fn(x * 1.1, x.sin() / 9)
        energy = y.sin().sum() / 64 + y.square().mean()
        force = -torch.autograd.grad(energy, x, create_graph=True)[0]
        return torch.autograd.grad(
            force.square().sum() + energy.square() / 10, parameters
        )

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Keep a direct offset pointer in the fused operation too.
        direct = plan(x, x)
        expected_direct = norm(gate(x + x))
        for _ in range(3):
            actual = execute(plan, x, parameters)
    stream.synchronize()
    close(direct, expected_direct, torch.float64)
    xr = x.detach().clone().requires_grad_()
    reference = execute(lambda p, s: norm(gate(p + s)), xr, (xr, *norm.parameters()))
    for a, b in zip(actual, reference):
        close(a, b, torch.float64, 30)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = execute(plan, x, parameters)
    graph.replay()
    torch.cuda.synchronize()
    for a, b in zip(captured, reference):
        close(a, b, torch.float64, 30)
    assert any(
        key[0].__name__ == "double_backward_direct" for key in plan.fused._launchers
    )


@pytest.mark.parametrize(
    "dtype,channels,nodes",
    [
        (torch.float32, 64, 64),
        (torch.float64, 3, 64),
        (torch.float64, 64, 6),
    ],
)
def test_short_and_fp32_keep_original_fusion(dtype, channels, nodes):
    gate, norm, _, plan = modules(dtype, channels)
    x = torch.randn(nodes, plan.dim, device="cuda", dtype=dtype, requires_grad=True)
    assert not plan.fused.rematerialize_adjoints(x)
    y = plan(x, x)
    reference = norm(gate(x + x))
    first = torch.autograd.grad(
        y.sin().sum(), (x, *norm.parameters()), create_graph=True
    )
    torch.autograd.grad(sum(g.square().sum() for g in first), (x, *norm.parameters()))
    close(y, reference, dtype)
    assert not any(
        key[0].__name__ in ("backward_direct", "double_backward_direct")
        for key in plan.fused._launchers
    )
