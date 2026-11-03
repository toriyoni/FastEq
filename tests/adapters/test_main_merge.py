"""Semantic regressions for merging main with the three-model adapters."""
import itertools

import pytest
import torch

from fasteq.ops import uniform1d_jit as u1d


MASKS = [mask for mask in itertools.product((False, True), repeat=3) if any(mask)]


@pytest.mark.parametrize('setting', (None, 'auto', '0', '1'))
def test_measured_double_policy_overrides_and_guards_cpu(monkeypatch, setting):
    from types import SimpleNamespace
    if setting is None:
        monkeypatch.delenv('FASTEQ_EXPERIMENTAL_U1D_DOUBLE_SPLIT', raising=False)
    else:
        monkeypatch.setenv('FASTEQ_EXPERIMENTAL_U1D_DOUBLE_SPLIT', setting)
    props = SimpleNamespace(major=9, minor=0, name='NVIDIA H100 80GB HBM3')
    monkeypatch.setattr(torch.cuda, 'get_device_properties', lambda _: props)
    w = SimpleNamespace(dtype=torch.float32, device=torch.device('cuda'), shape=(12288, 11, 64))
    x = SimpleNamespace(shape=(1024, 9, 64))
    go = SimpleNamespace(shape=(1024, 35, 64))
    kw = dict(w=w, x=x, grad_out=go, mode='u,u,,u', u_dim=64,
              dims=(11, 9, 9, 35), paths=83, grad_flags=(True, True, True),
              index_flags=(True, False, True))
    forced = None if setting in (None, 'auto') else setting == '1'
    assert u1d._resolve_double_split_enabled(**kw) == (True if forced is None else forced)
    for change in ({'paths': 9}, {'u_dim': 33}, {'grad_flags': (False, True, True)}):
        assert u1d._resolve_double_split_enabled(**(kw | change)) == bool(forced)
    for obj, attr, value in ((x, 'shape', (1023, 9, 64)),
                             (w, 'dtype', torch.float64),
                             (w, 'device', torch.device('cpu')),
                             (props, 'name', 'NVIDIA A100'), (props, 'major', 8)):
        before = getattr(obj, attr)
        setattr(obj, attr, value)
        assert u1d._resolve_double_split_enabled(**kw) == bool(forced)
        setattr(obj, attr, before)
    kw.update(mode='u,u,u,u', dims=(1, 9, 9, 35), grad_flags=(False, True, True),
              index_flags=(False, False, False))
    w.shape = (1, 1, 64)
    assert u1d._resolve_double_split_enabled(**kw) == bool(forced)
    x.shape, go.shape = (2048, 9, 64), (2048, 35, 64)
    assert u1d._resolve_double_split_enabled(**kw) == (True if forced is None else forced)


@pytest.mark.parametrize('backend', ('triton', 'gluon'))
@pytest.mark.parametrize('mask', MASKS)
def test_main_optional_backend_double_codegen_cpu(backend, mask):
    import ast
    import importlib
    module = importlib.import_module(f'fasteq.ops.auto_schedule.{backend}_backend')
    generate = getattr(module, f'build_{backend}_uniform1d_double_bwd_module_source')
    ids = torch.tensor([0, 1, 0], dtype=torch.int32)
    source = generate(ids, ids, ids, ids, torch.tensor([.2, .3, .4]),
                      u_dim=33, mode='u,u,,u', need_grad_w=mask[0],
                      need_grad_x=mask[1], need_grad_y=mask[2])
    ast.parse(source)
    assert 'unused' not in source


def test_legacy_tuning_controls_cpu(monkeypatch):
    monkeypatch.setattr(u1d, 'ENABLE_JIT_CANDIDATES', True)
    monkeypatch.setenv('FASTEQ_UNIFORM1D_MULTI_WARP_CANDIDATES', '0')
    assert not u1d._resolve_multiwarp_candidates_enabled(None)
    assert not u1d._resolve_multiwarp_candidates_enabled(False)
    assert u1d._resolve_multiwarp_candidates_enabled(True)

    def forbidden(*args, **kwargs):
        raise AssertionError('Disabled tuning must not allocate, launch, or synchronize')

    monkeypatch.setattr(torch.cuda, 'synchronize', forbidden)
    candidates = [('first', object()), ('second', object())]
    selected = u1d._benchmark_and_select_best_jit_candidate(
        tune_key='disabled', candidates=candidates, best_cache={}, kind='FWD',
        tune_enabled=False, warmup=3, repeat=10, call_fn=forbidden,
        batch_size=16384, case_factory=forbidden, case_call_fn=forbidden)
    assert selected[:2] == candidates[0]
    monkeypatch.setattr(u1d, 'ENABLE_JIT_CANDIDATES', False)
    assert not u1d._resolve_multiwarp_candidates_enabled(True)


@pytest.mark.parametrize('mask', MASKS)
@pytest.mark.parametrize('scalar', (False, True))
def test_split_codegen_all_masks_cpu(mask, scalar):
    from fasteq.ops.uniform1d_auto_schedule import generate_code_uniform1d_double_bwd_with_scheduler
    ids = torch.tensor([0, 1, 0], dtype=torch.int32)
    source = generate_code_uniform1d_double_bwd_with_scheduler(
        ids, ids, ids, ids, torch.tensor([.2, .3, .4]),
        u_dim=33, iw_dim=2, ix_dim=2, ky_dim=2, v_dim=2,
        mode='u,u,,u' if scalar else 'u,u,u,u',
        need_grad_w=mask[0], need_grad_x=mask[1], need_grad_y=mask[2],
        split_outputs=True, out_path='')
    assert 'unused' not in source
    assert source.count('__global__ void') == 1 + sum(mask)
    for kind, enabled in zip(('dw', 'dx', 'dy'), mask):
        assert (f'__global__ void uniform1d_double_bwd_lars_splitacc_{kind}(' in source) == enabled


def _reference_derivatives(actual, expected, leaves, refs):
    forward_tolerance = dict(atol=2e-9, rtol=2e-8) if actual.dtype == torch.float64 else dict(atol=3e-5, rtol=5e-5)
    derivative_tolerance = dict(atol=2e-8, rtol=2e-7) if actual.dtype == torch.float64 else dict(atol=3e-4, rtol=3e-4)
    torch.testing.assert_close(actual, expected, **forward_tolerance)
    first = torch.autograd.grad(actual.square().sum(), leaves, create_graph=True)
    reference_first = torch.autograd.grad(expected.square().sum(), refs, create_graph=True)
    seeds = [torch.randn_like(g) for g in first]
    second = torch.autograd.grad(first, leaves, seeds, allow_unused=True)
    reference_second = torch.autograd.grad(reference_first, refs, seeds, allow_unused=True)
    for got, want in zip((*first, *second), (*reference_first, *reference_second), strict=True):
        if want is None:
            assert got is None or torch.count_nonzero(got) == 0
        else:
            assert got is not None
            torch.testing.assert_close(got, want, **derivative_tolerance)


def _case(device, scalar, shared, mask):
    torch.manual_seed(511)
    n, c = 5, 33
    shapes = [(1 if shared else n, c), (n, c), (n, 1 if scalar else c)]
    inputs = [torch.randn(s, device=device, dtype=torch.float64).requires_grad_(flag)
              for s, flag in zip(shapes, mask)]
    refs = [v.detach().clone().requires_grad_(flag) for v, flag in zip(inputs, mask)]
    index = torch.tensor([0], dtype=torch.int32, device=device)
    meta = dict(i_list=index, j_list=index, k_list=index, v_list=index,
                coeff_list=torch.ones(1, device=device, dtype=torch.float64),
                out_seg_num=1, w_seg_num=1, x_seg_num=1, y_seg_num=1, u_dim=c,
                size_list=[c, c, 1 if scalar else c, c])
    actual = u1d.fast_uniform1d_jit(*inputs, {}, {}, meta, use_multiwarp_candidates=False)
    expected = refs[0] * refs[1] * refs[2]
    _reference_derivatives(actual, expected,
                           [v for v in inputs if v.requires_grad],
                           [v for v in refs if v.requires_grad])


@pytest.mark.parametrize("mask", MASKS)
@pytest.mark.parametrize("scalar,shared", tuple(itertools.product((False, True), repeat=2)))
def test_public_input_graph_and_gradient_masks_cpu(monkeypatch, mask, scalar, shared):
    def fwd(**kw):
        return kw['w'] * kw['x'] * kw['y']

    def bwd(**kw):
        w, x, y, g = (kw[k] for k in ('w', 'x', 'y', 'grad_out'))
        values = [(g*x*y).sum_to_size(w.shape), g*w*y, (g*w*x).sum_to_size(y.shape)]
        return tuple(v for k, v in zip('wxy', values) if kw['grad_'+k])

    def bwd2(**kw):
        w, x, y, g = (kw[k] for k in ('w', 'x', 'y', 'grad_out'))
        a, b, c = [kw['grad_grad_'+k] if kw['grad_'+k] else torch.zeros_like(v)
                   for k, v in zip('wxy', (w, x, y))]
        values = [a*x*y+b*w*y+c*w*x, (b*g*y+c*g*x).sum_to_size(w.shape),
                  a*g*y+c*g*w, (a*g*x+b*g*w).sum_to_size(y.shape)]
        return (values[0], *(v for k, v in zip('wxy', values[1:]) if kw['grad_'+k]))

    monkeypatch.setattr(u1d, '_run_fwd', fwd)
    monkeypatch.setattr(u1d, '_run_bwd', bwd)
    monkeypatch.setattr(u1d, '_run_double_bwd', bwd2)
    _case('cpu', scalar, shared, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("mask", [(True, True, True), (True, False, True), (False, True, False)])
@pytest.mark.parametrize("split", (False, True))
def test_cuda_gradient_masks_and_split_accumulators(monkeypatch, mask, split):
    monkeypatch.setenv('FASTEQ_EXPERIMENTAL_U1D_DOUBLE_SPLIT', str(int(split)))
    _case('cuda', True, True, mask)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
@pytest.mark.parametrize("train_weight", (False, True))
def test_stc_renamed_entry_native_weight_gradient(train_weight):
    from fasteq.ops.stc import fast_stc_uniform1d_jit
    from fasteq.ops.stc_uniform1d_jit import fast_stc_uniform1d_jit as renamed
    assert fast_stc_uniform1d_jit is renamed
    torch.manual_seed(133)
    x1 = torch.randn(5, 2, 33, device='cuda', dtype=torch.float64).requires_grad_()
    x0 = torch.randn(3, 2, 33, device='cuda', dtype=torch.float64).requires_grad_(train_weight)
    r1 = x1.detach().clone().requires_grad_()
    r0 = x0.detach().clone().requires_grad_(train_weight)
    i0 = torch.tensor([0, 1, 0, 2, 1], device='cuda')
    paths = torch.tensor([[0, 0, 0, -1, -1], [0, 1, 1, 1, -1],
                          [1, 1, 0, 0, 1]], device='cuda', dtype=torch.int32)
    coeff = torch.tensor([.2, -.3, .4], device='cuda', dtype=torch.float64)
    actual = renamed(x1, x0, i0, coeff, paths, paths.T.contiguous(), 2)
    gathered = r0[i0]
    expected = torch.stack((.2*r1[:, 0]*gathered[:, 0],
                            -.3*r1[:, 0]*r1[:, 1]*gathered[:, 1]
                            +.4*r1[:, 1]*r1[:, 1]*r1[:, 0]*gathered[:, 0]), dim=1).flatten(1)
    _reference_derivatives(actual, expected,
                           [x1, x0] if train_weight else [x1],
                           [r1, r0] if train_weight else [r1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_fctp_weight_and_mixed_second_derivatives():
    from fasteq.ops.fctp import fast_fctp
    torch.manual_seed(134)
    w = torch.randn(2, 4, 3, 5, device='cuda').requires_grad_()
    x = torch.randn(7, 8, device='cuda').requires_grad_()
    wr, xr = [v.detach().clone().requires_grad_() for v in (w, x)]
    ids = torch.tensor([0, 1, 2, 0, 2, 1, 0], device='cuda')
    y = torch.nn.functional.one_hot(ids, 3).float()
    meta = {k: None for k in ('cg_indices', 'cg_values', 'cg_i_all', 'cg_j_all',
                             'cg_k_all', 'cg_val_all', 'nnz_per_path', 'K_per_path', 'path_offset')}
    meta.update(P=2, U=4, V=3, W=5, I_total=2, K_total=2, nnz0=2, cg_val_0=.5,
                p_for_k=torch.tensor([0, 1], device='cuda', dtype=torch.int32),
                i_for_k=torch.tensor([0, 1], device='cuda', dtype=torch.int32),
                val_for_k=torch.tensor([.3, -.7], device='cuda'), can_use_empty_grad_x=True)
    actual = fast_fctp(w, x, y, meta)
    expected = torch.stack([
        torch.einsum('bu,buw->bw', xr.view(7, 2, 4)[:, k], wr[k, :, ids, :].permute(1, 0, 2)) * scale
        for k, scale in enumerate((.15, -.35))], dim=1)
    _reference_derivatives(actual, expected.reshape_as(actual), [w, x], [wr, xr])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_main_so3_autotune_large_grid_and_cached_choice():
    from fasteq.triton import so3_linear as linear
    layout = linear.LinearLayout(((5, 2),), ((5, 2),), ((2, 2),), ((0, 0, 0),), (1.,))
    x = torch.randn(419431, 10, device='cuda')  # ceil(N*5/32) > 65535
    w = torch.randn(4, device='cuda')
    key = linear.best_impl_cache_key(layout, x.dtype)
    linear.clear_best_impl_cache()
    with torch.no_grad():
        expected = (x.view(-1, 5, 2) @ w.view(2, 2)).flatten(1)
        for cached in (None, 'group_parallel', 'path_parallel'):
            linear._BEST_IMPL_CACHE[key] = {} if cached is None else {False: cached}
            actual = linear.so3_linear_auto(w, x, layout, benchmark_warmup=0, benchmark_rep=1)
            torch.testing.assert_close(actual, expected, atol=3e-5, rtol=5e-5)
    linear.clear_best_impl_cache()
