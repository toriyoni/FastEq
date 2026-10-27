"""Original FastEq WGMMA reuse: tails, indexed layouts and mixed derivatives."""
import pytest
import torch
from e3nn import o3
from fasteq.ops.indexed_linear import IndexedLinear
from fasteq.ops.wgmma_linear import wgmma_linear, eligible

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() != (9, 0),
    reason="Hopper WGMMA required",
)


@pytest.mark.parametrize("nodes,banks,channels,outputs,empty_bank", [
    (129, 2, (192, 256, 256), 64, False),
    (3, 2, (17, 33, 65), 32, True),
    (257, 1, (64, 64, 64), 96, False),
    (131, 3, (64, 96, 128), 64, False),
])
def test_indexed_tails_and_mixed_derivatives(nodes, banks, channels, outputs, empty_bank):
    torch.manual_seed(3101)
    lin = o3.Linear(
        "+".join(f"{c}x{l}{'e' if l % 2 == 0 else 'o'}" for l, c in enumerate(channels)),
        f"{outputs}x0e+{outputs}x1o+{outputs}x2e", internal_weights=False, shared_weights=False,
    ).cuda()
    plan = IndexedLinear(lin)
    # Offset storage verifies that the adapter uses tensor data pointers.
    x = torch.randn(nodes * plan.input_dim + 1, device="cuda")[1:].view(nodes, -1).requires_grad_()
    w = torch.randn(banks, 1, plan.weight_numel, device="cuda", requires_grad=True)
    types = torch.arange(nodes, device="cuda") % banks
    if empty_bank:
        types.zero_()
    cotangent = torch.randn(nodes, plan.output_dim, device="cuda")
    vx, vw = torch.randn_like(x), torch.randn_like(w)

    def evaluate(fn):
        y = fn()
        gx, gw = torch.autograd.grad((y * cotangent).sum(), (x, w), create_graph=True)
        hx, hw = torch.autograd.grad((gx * vx).sum() + (gw * vw).sum(), (x, w))
        return y, gx, gw, hx, hw

    previous = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.backends.cuda.matmul.allow_tf32 = False
        expected = evaluate(lambda: lin(x, w[:, 0][types]))
        torch.backends.cuda.matmul.allow_tf32 = True
        actual = evaluate(lambda: wgmma_linear(plan.metadata, x, w, types))
        for got, ref in zip(actual, expected, strict=True):
            torch.testing.assert_close(got, ref, atol=2e-4, rtol=2e-3)
        assert torch.backends.cuda.matmul.allow_tf32
    finally:
        torch.backends.cuda.matmul.allow_tf32 = previous
    assert not eligible(plan.metadata, x, w), "Small inputs must retain the existing dispatch"
