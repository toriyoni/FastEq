"""Resident Norm launches keep stream ordering and a real force-loss CUDA graph."""

from copy import deepcopy

import pytest
import torch
from test_norm_selective_cuda import module_and_plan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def force_loss(module, plan, coordinates, projection, target):
    feature = (coordinates @ projection).reshape(coordinates.shape[0], *plan.shape)
    output = plan(module, feature)
    energy = output.square().mean() + 0.07 * output.sin().mean()
    forces = -torch.autograd.grad(energy, coordinates, create_graph=True)[0]
    loss = (forces - target).square().mean()
    gradients = torch.autograd.grad(
        loss, (coordinates, projection, *module.parameters()), allow_unused=True
    )
    return forces, gradients


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["tace", "prophet"])
@pytest.mark.parametrize("capture", [False, True])
def test_norm_resident_nondefault_stream_and_second_order_graph(dtype, kind, capture):
    torch.manual_seed(9091)
    torch.backends.cuda.matmul.allow_tf32 = False
    module, plan = module_and_plan(kind, dtype)
    reference = deepcopy(module)
    coordinates = torch.randn(13, 3, device="cuda", dtype=dtype, requires_grad=True)
    projection = torch.randn(3, plan.dim, device="cuda", dtype=dtype, requires_grad=True)
    coordinates_ref = coordinates.detach().clone().requires_grad_()
    projection_ref = projection.detach().clone().requires_grad_()
    assert coordinates_ref.data_ptr() != coordinates.data_ptr()
    assert projection_ref.data_ptr() != projection.data_ptr()
    assert all(a.data_ptr() != b.data_ptr() for a, b in
               zip(module.parameters(), reference.parameters(), strict=True))
    target = torch.randn_like(coordinates)
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        # Warm every first/double path before capture, using the capture stream.
        for _ in range(3):
            force_loss(module, plan, coordinates, projection, target)
    stream.synchronize()
    assert plan._launchers and all(value is not None for value in plan._launchers.values())
    if capture:
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            actual = force_loss(module, plan, coordinates, projection, target)
        graph.replay()
        torch.cuda.synchronize()
    else:
        with torch.cuda.stream(stream):
            actual = force_loss(module, plan, coordinates, projection, target)
        stream.synchronize()

    def original(owner, value):
        return owner(value)

    original.shape = plan.shape
    expected = force_loss(reference, original, coordinates_ref, projection_ref, target)
    atol, rtol = (8e-4, 8e-4) if dtype == torch.float32 else (4e-9, 4e-8)
    torch.testing.assert_close(actual[0], expected[0], atol=atol, rtol=rtol)
    for a, b in zip(actual[1], expected[1], strict=True):
        if b is None:
            assert a is None or not torch.count_nonzero(a)
        else:
            assert a is not None
            torch.testing.assert_close(a, b, atol=atol, rtol=rtol)
    # Plan deepcopy must not retain non-picklable resident executable handles.
    assert deepcopy(plan)._launchers == {}
