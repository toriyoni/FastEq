"""FastEq per-copy invariant squared means using its native CUDA runtime.

Metadata and first/double-backward contexts preserve irreps_invariants.py.
CUDA reductions reuse FastEq's existing warp_reduce_sum implementation; no
Triton kernels or external CUDA executor are imported or called.
"""
from functools import lru_cache
from pathlib import Path

import torch


@lru_cache(maxsize=128)
def _source(dtype, dim, channels, angle):
    directory = Path(__file__).parents[1] / 'cuda/src'
    original = (directory / 'equi_linear_kernels/device_cuda_helper.cuh').read_text()
    marker = 'template <typename T> __device__ __forceinline__ T warp_reduce_sum'
    begin = original.index(marker)
    opening = original.index('{', begin)
    depth, end = 1, opening + 1
    while depth:
        depth += (original[end] == '{') - (original[end] == '}')
        end += 1
    reduction = original[begin:end]
    # XOR shuffles with masks smaller than lanes never cross a subgroup.
    lanes = min(32, angle)
    scalar = 'double' if dtype == torch.float64 else 'float'
    header = (directory / 'native_invariants.cuh').read_text()
    return (f'#define WARP_SIZE {lanes}\n#define FASTEQ_INVARIANT_TYPE {scalar}\n'
            f'#define FASTEQ_INVARIANT_DIM {dim}\n#define FASTEQ_INVARIANT_CHANNELS {channels}\n'
            f'#define FASTEQ_INVARIANT_LANES {lanes}\n' + reduction + '\n' + header)


def _launch(entry, x, plan, values, tables, outputs, count):
    if not count:
        return
    from fasteq.cuda.jit import kernels, runtime
    code = _source(x.dtype, plan.dim, plan.channels, plan.angle)
    kernel = kernels([code], x.device, entry=entry)[code]
    assert all(value.is_contiguous() for value in values)
    args = [value.data_ptr() for value in values]
    args += [value.data_ptr() for value in tables]
    args += [value.data_ptr() if value is not None else 0 for value in outputs]
    args.append(x.shape[0])
    runtime().launch([(kernel, args, (count+127)//128, 1, 128, 0)],
                     torch.cuda.current_stream(x.device).cuda_stream)


class _First(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gradient, plan, metadata):
        _base, widths, index = metadata
        result = x.new_empty(x.shape)
        x2, g2 = x.contiguous(), gradient.contiguous()
        _launch('native_invariant_backward', x2, plan, (x2, g2), (index, widths),
                (result,), x.numel())
        ctx.plan, ctx.tables = plan, metadata
        ctx.save_for_backward(x, gradient)
        ctx.set_materialize_grads(False)
        return result

    @staticmethod
    def backward(ctx, direction):
        if direction is None:
            return None, None, None, None
        x, gradient = ctx.saved_tensors
        plan = ctx.plan
        x2, g2, u2 = x.contiguous(), gradient.contiguous(), direction.contiguous()
        dx = x.new_empty(x.shape) if ctx.needs_input_grad[0] else None
        dg = gradient.new_empty(gradient.shape) if ctx.needs_input_grad[1] else None
        count = max(dx.numel() if dx is not None else 0,
                    dg.numel()*min(plan.angle, 32) if dg is not None else 0)
        _launch('native_invariant_double', x2, plan, (x2, g2, u2), ctx.tables,
                (dx, dg), count)
        return dx, dg, None, None


class _Norms(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan, metadata):
        base, widths, _index = metadata
        result = x.new_empty(x.shape[0], plan.channels)
        x2 = x.contiguous()
        _launch('native_invariant_forward', x2, plan, (x2,), (base, widths),
                (result,), result.numel()*min(plan.angle, 32))
        ctx.plan, ctx.tables = plan, metadata
        ctx.save_for_backward(x)
        ctx.set_materialize_grads(False)
        return result

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None or not ctx.needs_input_grad[0]:
            return None, None, None
        (x,) = ctx.saved_tensors
        return _First.apply(x, gradient, ctx.plan, ctx.tables), None, None


class InvariantPlan:
    backend = 'FastEq native CUDA invariant squared means (first/double gradients)'

    def __init__(self, irreps):
        bases, widths, index, offset = [], [], [], 0
        for channels, ir in irreps:
            for channel in range(channels):
                index.extend([len(bases)] * ir.dim)
                bases.append(offset + channel * ir.dim)
                widths.append(ir.dim)
            offset += channels * ir.dim
        self.dim, self.channels = offset, len(bases)
        self.angle = 1 << (max(widths, default=1)-1).bit_length()
        self._data = tuple(bases), tuple(widths), tuple(index)
        self._cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state['_cache'] = {}
        return state

    def __call__(self, x):
        if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
            raise TypeError('FastEq invariants require GPU FP32/FP64')
        if x.ndim != 2 or x.shape[1] != self.dim:
            raise ValueError('Invariant features must use packed mul_ir layout')
        if x.device not in self._cache:
            with torch.inference_mode(False), torch.no_grad():
                self._cache[x.device] = tuple(torch.tensor(data, device=x.device, dtype=torch.int32)
                                             for data in self._data)
        return _Norms.apply(x, self, self._cache[x.device])


__all__ = ['InvariantPlan']
