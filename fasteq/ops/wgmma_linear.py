"""Hopper extension of FastEq's original TMA/WGMMA Linear.

Input gathering/padding and output scattering are fused into the original
equi_linear_kernels/equi_linear_fwd_f32.cu pipeline; no packed global tensors
are materialized. Derivatives reuse IndexedLinear's
existing recursive FastEq contractions, including gradients of force losses.
"""
from functools import lru_cache
from pathlib import Path
import torch


@lru_cache(maxsize=1)
def _runtime():
    from torch.utils.cpp_extension import load
    directory = Path(__file__).parents[1] / 'cuda/src'
    return load(
        name='fasteq_linear_wgmma',
        sources=[str(directory / 'equi_linear_wgmma_binding.cu')],
        extra_include_paths=[str(directory)],
        extra_cuda_cflags=['-O3', '--expt-relaxed-constexpr',
                          '-gencode=arch=compute_90a,code=sm_90a'],
        extra_ldflags=['-lcuda'],
        verbose=False,
    )


@lru_cache(maxsize=256)
def _layout(metadata):
    from .indexed_linear import _decode
    din, dout, numel, experts, layout, paths = _decode(metadata)
    if experts != 1 or layout != 'mul_ir' or not 1 <= len(paths) <= 16:
        return None
    sx = sw = sy = 0
    previous_dim = None
    co = paths[0][4]
    if co < 32 or co % 32:
        return None
    for ix, iw, iy, ci, outputs, dim, coefficient in paths:
        if (ix, iw, iy) != (sx, sw, sy) or not 1 <= ci <= 512:
            return None
        if outputs != co or not 1 <= dim <= 31 or dim == previous_dim:
            return None
        sx += ci * dim
        sw += ci * co
        sy += co * dim
        previous_dim = dim
    if (sx, sw, sy) != (din, numel, dout):
        return None
    return [list(p[:6]) for p in paths], [p[6] for p in paths], dout


def eligible(metadata, features, weight):
    return (features.is_cuda and features.dtype == torch.float32
            and features.shape[0] >= 16384 and features.shape[1] >= 1024
            and weight.shape[0] <= 4
            and torch.cuda.get_device_capability(features.device) == (9, 0)
            and not torch.cuda.is_current_stream_capturing()
            and _layout(metadata) is not None)


class _Shared(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, features, weight):
        ctx.kernel_metadata = metadata
        ctx.save_for_backward(features, weight)
        paths, coefficients, dout = _layout(metadata)
        return _runtime().linear(features.contiguous(), weight.contiguous(),
                                 paths, coefficients, dout, fusion=3)

    @staticmethod
    def backward(ctx, cotangent):
        from .indexed_linear import _contract
        features, weight = ctx.saved_tensors
        types = torch.zeros(features.shape[0], dtype=torch.int64, device=features.device)
        dx = dw = None
        if ctx.needs_input_grad[1]:
            dx = _contract(ctx.kernel_metadata, 0, types, 1, features, weight, cotangent)
        if ctx.needs_input_grad[2]:
            dw = _contract(ctx.kernel_metadata, 1, types, 1, features, weight, cotangent)
        return None, dx, dw


class _Indexed(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, features, weight, node_type):
        paths, coefficients, dout = _layout(metadata)
        result = features.new_empty((features.shape[0], dout))
        groups = [(node_type == bank).nonzero().flatten()
                  for bank in range(weight.shape[0])]
        for bank, ids in enumerate(groups):
            if ids.numel():
                _runtime().indexed_out(features.contiguous(), weight[bank:bank + 1].contiguous(),
                                       ids, result, paths, coefficients, fusion=3)
        ctx.kernel_metadata = metadata
        ctx.save_for_backward(features, weight, *groups)
        return result

    @staticmethod
    def backward(ctx, cotangent):
        from .indexed_linear import _contract
        features, weight, *groups = ctx.saved_tensors
        dx = torch.zeros_like(features) if ctx.needs_input_grad[1] else None
        dws = []
        for bank, ids in enumerate(groups):
            ww = weight[bank:bank + 1]
            if not ids.numel():
                if ctx.needs_input_grad[2]:
                    dws.append(torch.zeros_like(ww))
                continue
            xx = features.index_select(0, ids)
            gg = cotangent.index_select(0, ids)
            types = torch.zeros(ids.numel(), dtype=torch.int64, device=features.device)
            if ctx.needs_input_grad[1]:
                local = _contract(ctx.kernel_metadata, 0, types, 1, xx, ww, gg)
                dx = dx.index_add(0, ids, local)
            if ctx.needs_input_grad[2]:
                dws.append(_contract(ctx.kernel_metadata, 1, types, 1, xx, ww, gg))
        dw = torch.cat(dws, dim=0) if ctx.needs_input_grad[2] else None
        return None, dx, dw, None


def wgmma_linear(metadata, features, weight, node_type):
    """Include dynamic element grouping and the fused native-layout Linear."""
    if weight.shape[0] == 1:
        return _Shared.apply(metadata, features, weight)
    return _Indexed.apply(metadata, features, weight, node_type)
