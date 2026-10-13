"""Review probes for alignment, row-grid reuse, indirect adjoints and capture."""

import pytest
import torch
from e3nn import o3
from fasteq.ops.modulation import ModulatedResidualPlan

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def offset_tensor(shape, dtype):
    size = torch.Size(shape).numel()
    return torch.randn(size + 1, device="cuda", dtype=dtype)[1:].view(shape)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_modulation_offset_row_reuse_indirect_seed(dtype):
    torch.manual_seed(831)
    irreps = o3.Irreps("2x0e+3x1o+1x2e")
    plan = ModulatedResidualPlan(irreps)
    index = torch.tensor(plan._index, device="cuda")
    tolerance = (
        {"atol": 3e-5, "rtol": 3e-4}
        if dtype == torch.float32
        else {"atol": 3e-11, "rtol": 3e-10}
    )
    for shape in ((), (1,), (13,), (2, 3)):
        values = [
            offset_tensor((*shape, size), dtype)
            for size in (irreps.dim, plan.channels, irreps.dim)
        ]
        assert all(value.data_ptr() % 16 for value in values)
        if len(shape) > 1:
            values = [value.transpose(0, 1) for value in values]
        values = [value.detach().requires_grad_() for value in values]
        references = [value.detach().clone().requires_grad_() for value in values]
        seeds = [
            offset_tensor(values[0].shape, dtype).requires_grad_() for _ in range(2)
        ]
        seeds[1] = seeds[0].detach().clone().requires_grad_()
        outputs = [
            plan(*values),
            references[2] + references[0] * references[1].index_select(-1, index),
        ]
        torch.testing.assert_close(*outputs, **tolerance)
        first = [
            torch.autograd.grad((output * seed).sum(), args, create_graph=True)
            for output, seed, args in zip(
                outputs, seeds, (values, references), strict=True
            )
        ]
        for a, b in zip(*first, strict=True):
            torch.testing.assert_close(a, b, **tolerance)
        directions = [offset_tensor(value.shape, dtype) for value in first[0]]
        second = [
            torch.autograd.grad(
                sum(
                    (g * direction).sum()
                    for g, direction in zip(grads, directions, strict=True)
                ),
                (*args, seed),
                allow_unused=True,
            )
            for grads, args, seed in zip(
                first, (values, references), seeds, strict=True
            )
        ]
        for a, b in zip(*second, strict=True):
            if a is None or b is None:
                other = b if a is None else a
                assert other is None or torch.count_nonzero(other) == 0
            else:
                torch.testing.assert_close(a, b, **tolerance)
    assert plan._launchers and all(
        value is not None for value in plan._launchers.values()
    )


def test_modulation_training_double_backward_graph():
    torch.manual_seed(941)
    irreps = o3.Irreps("2x0e+2x1o+2x2e")
    plan = ModulatedResidualPlan(irreps)
    index = torch.tensor(plan._index, device="cuda")
    values = [
        torch.randn(5, size, device="cuda", dtype=torch.float64, requires_grad=True)
        for size in (irreps.dim, plan.channels, irreps.dim)
    ]

    def evaluate(native, operands):
        output = (
            plan(*operands)
            if native
            else operands[2] + operands[0] * operands[1].index_select(-1, index)
        )
        first = torch.autograd.grad(output.sin().sum(), operands, create_graph=True)
        second = torch.autograd.grad(
            sum(value.square().sum() for value in first), operands
        )
        return output, first, second

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            evaluate(True, values)
    torch.cuda.current_stream().wait_stream(stream)
    references = [value.detach().clone().requires_grad_() for value in values]
    expected = evaluate(False, references)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        actual = evaluate(True, values)
    graph.replay()
    for group_a, group_b in zip(actual, expected, strict=True):
        for a, b in (
            zip(group_a, group_b, strict=True)
            if isinstance(group_a, tuple)
            else ((group_a, group_b),)
        ):
            torch.testing.assert_close(a, b, atol=3e-10, rtol=3e-9)
