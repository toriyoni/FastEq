"""Native-reference VJP checks for the split LARS double-backward kernels."""
import pytest
import torch
from fasteq.ops.uniform1d_jit import _run_bwd, _run_double_bwd

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")

@pytest.mark.parametrize("scalar_y,channels,source,scatter,shared,grad_w", [
    (True, 33, 1, True, False, True),
    (False, 64, None, False, True, False),
    (True, 17, 2, True, True, False),
])
def test_split_double_adjoint_layouts_and_channel_tails(
        monkeypatch, scalar_y, channels, source, scatter, shared, grad_w):
    monkeypatch.setenv("FASTEQ_EXPERIMENTAL_U1D_DOUBLE_SPLIT", "1")
    monkeypatch.setenv("FASTEQ_EXPERIMENTAL_U1D_SCALAR_REDUCE", "1")
    torch.manual_seed(4201)
    edges, nodes, iw, ix, ky, vo = 17, 7, 3, 3, 3, 2
    rows = [(0, 0, 0, 0), (1, 1, 0, 1), (2, 0, 2, 0),
            (0, 1, 1, 1), (1, 2, 2, 0), (2, 0, 0, 0)]
    columns = torch.tensor(rows, device="cuda", dtype=torch.int32).T.contiguous()
    coeff = torch.tensor([.3, -.4, .2, -.7, .1, .5], device="cuda")
    src = torch.randint(nodes, (edges,), device="cuda", dtype=torch.int32)
    dst = torch.randint(nodes, (edges,), device="cuda", dtype=torch.int32)
    def leaf(shape):return torch.randn(*shape, device="cuda", requires_grad=True)
    w = leaf((1 if shared else edges, iw, channels))
    x = leaf((nodes if source == 1 else edges, ix, channels))
    y = leaf((nodes if source == 2 else edges, ky, 1 if scalar_y else channels))
    go = leaf((nodes if scatter else edges, vo, channels))
    ggw, ggx, ggy = [torch.randn_like(t) for t in (w, x, y)]
    xx, yy = (x[src.long()] if source == 1 else x), (y[src.long()] if source == 2 else y)
    z = x.new_zeros(edges, vo, channels)
    for c, (i, j, k, v) in zip(coeff, rows):
        z[:, v] = z[:, v] + c * w[:, i] * xx[:, j] * yy[:, k]
    if scatter:z = z.new_zeros(nodes, vo, channels).index_add(0, dst.long(), z)
    leaves, upstream = ((w, x, y), (ggw, ggx, ggy)) if grad_w else ((x, y), (ggx, ggy))
    first = torch.autograd.grad(z, leaves, go, create_graph=True)
    expected = torch.autograd.grad(first, (go, w, x, y), upstream)
    actual_first = _run_bwd(
        grad_out=go, w=w, x=x, y=y,
        i_list=columns[0], j_list=columns[1], k_list=columns[2], v_list=columns[3],
        # First backward's public wrapper requires source indices, including
        # an identity map for the dense case. Double backward also accepts {}.
        coeff_list=coeff, input_indices=({1: torch.arange(edges, device="cuda", dtype=torch.int32)}
                                        if source is None else {source: src}),
        output_indices={0: dst} if scatter else {}, out_seg_num=vo, u_dim=channels,
        iw_dim=iw, ix_dim=ix, ky_dim=ky, v_dim=vo,
        mode="u,u,,u" if scalar_y else "u,u,u,u", grad_w=grad_w,
        use_multiwarp_candidates=False)
    for got, ref in zip(actual_first, first, strict=True):
        torch.testing.assert_close(got, ref, atol=3e-5, rtol=3e-5)
    actual = _run_double_bwd(
        grad_out=go, w=w, x=x, y=y, grad_grad_w=ggw if grad_w else None,
        grad_grad_x=ggx, grad_grad_y=ggy,
        i_list=columns[0], j_list=columns[1], k_list=columns[2], v_list=columns[3],
        coeff_list=coeff, input_indices={} if source is None else {source: src},
        output_indices={0: dst} if scatter else {}, out_seg_num=vo, u_dim=channels,
        iw_dim=iw, ix_dim=ix, ky_dim=ky, v_dim=vo,
        mode="u,u,,u" if scalar_y else "u,u,u,u", grad_w=grad_w,
        use_multiwarp_candidates=False)
    for got, ref in zip(actual, expected, strict=True):
        torch.testing.assert_close(got, ref, atol=3e-5, rtol=3e-5)
