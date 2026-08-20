import pytest
import torch

from fasteq.ops.uniform1d_jit import fast_uniform1d_jit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Uniform1D JIT requires CUDA")
def test_uniform1d_double_backward_matches_torch_reference():
    torch.manual_seed(4)
    dtype = torch.float64
    device = "cuda"
    batch, u_dim = 2, 3
    w_seg_num = x_seg_num = y_seg_num = out_seg_num = 2
    i_list = torch.tensor([0, 1], dtype=torch.int32)
    j_list = torch.tensor([1, 0], dtype=torch.int32)
    k_list = torch.tensor([0, 1], dtype=torch.int32)
    v_list = torch.tensor([1, 0], dtype=torch.int32)
    coeff_list = torch.tensor([0.7, -1.1], dtype=dtype)
    meta = {
        "i_list": i_list,
        "j_list": j_list,
        "k_list": k_list,
        "v_list": v_list,
        "coeff_list": coeff_list,
        "out_seg_num": out_seg_num,
        "w_seg_num": w_seg_num,
        "x_seg_num": x_seg_num,
        "y_seg_num": y_seg_num,
        "u_dim": u_dim,
        "size_list": [6, 6, 2, 6],
    }
    src = torch.arange(batch, device=device, dtype=torch.int32)
    w = torch.randn(batch, 6, device=device, dtype=dtype, requires_grad=True)
    x = torch.randn(batch, 6, device=device, dtype=dtype, requires_grad=True)
    y = torch.randn(batch, 2, device=device, dtype=dtype, requires_grad=True)

    out = fast_uniform1d_jit(w, x, y, {1: src}, {}, meta, False)
    grad_out = torch.randn_like(out, requires_grad=True)
    grads = torch.autograd.grad((out * grad_out).sum(), (w, x, y), create_graph=True)
    cotangents = tuple(torch.randn_like(value) for value in grads)
    actual = torch.autograd.grad(
        sum((value * cotangent).sum() for value, cotangent in zip(grads, cotangents)),
        (grad_out, w, x, y),
    )

    w3 = w.view(batch, w_seg_num, u_dim)
    x3 = x.view(batch, x_seg_num, u_dim)
    y3 = y.view(batch, y_seg_num, 1)
    reference_out = torch.zeros(
        batch, out_seg_num, u_dim, device=device, dtype=dtype
    )
    for edge in range(batch):
        for i, j, k, v, coeff in zip(
            i_list.tolist(), j_list.tolist(), k_list.tolist(),
            v_list.tolist(), coeff_list.tolist(),
        ):
            reference_out[edge, v] = (
                reference_out[edge, v]
                + coeff * w3[edge, i] * x3[edge, j] * y3[edge, k, 0]
            )
    reference_grads = torch.autograd.grad(
        (reference_out.view_as(out) * grad_out).sum(), (w, x, y), create_graph=True
    )
    expected = torch.autograd.grad(
        sum(
            (value * cotangent).sum()
            for value, cotangent in zip(reference_grads, cotangents)
        ),
        (grad_out, w, x, y),
    )

    for actual_value, expected_value in zip(actual, expected):
        torch.testing.assert_close(actual_value, expected_value, rtol=1e-10, atol=1e-10)
