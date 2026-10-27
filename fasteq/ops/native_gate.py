"""FastEq Gate's existing metadata/AD contract with native CUDA launches.

The formulas are translated from FastEq's fused_equivariant_gate.py, preserving
its normalized activation opcodes and deterministic input-centric fan-in. This
module neither imports nor executes Triton or an external CUDA implementation.
"""
from functools import lru_cache
import math
from pathlib import Path
from typing import Callable, Optional

import torch

ACT_ID, ACT_SILU, ACT_SIGMOID, ACT_TANH = 0, 1, 2, 3


def act_code(fn: Optional[Callable]) -> int:
    """FastEq's original activation interface; reject unsupported functions."""
    if fn is None:
        return ACT_ID
    name = f"{getattr(fn, '__name__', '')} {fn.__class__.__name__} {fn}".lower()
    if 'silu' in name or 'swish' in name:
        return ACT_SILU
    if 'sigmoid' in name:
        return ACT_SIGMOID
    if 'tanh' in name:
        return ACT_TANH
    if 'identity' in name:
        return ACT_ID
    raise NotImplementedError('FastEquivariantGate only supports None/identity, SiLU, sigmoid, '
                              f'and tanh; received {fn!r}')


@lru_cache(maxsize=256)
def _source(dtype, din, dout, fan):
    scalar = 'double' if dtype == torch.float64 else 'float'
    header = (Path(__file__).parents[1] / 'cuda/src/native_gate.cuh').read_text()
    return (f'#define FASTEQ_GATE_TYPE {scalar}\n#define FASTEQ_GATE_IN {din}\n'
            f'#define FASTEQ_GATE_OUT {dout}\n#define FASTEQ_GATE_FAN {fan}\n' + header)


def _launch(entry, x, tables, values, outputs, rows, count):
    if not count:
        return
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
        raise TypeError('FastEq native Gate requires GPU float32/float64')
    from fasteq.cuda.jit import kernels, runtime
    code = _source(x.dtype, x.shape[-1], tables[0].numel(), tables[4].shape[1])
    kernel = kernels([code], x.device, entry=entry)[code]
    assert all(value.is_contiguous() for value in values)
    args = [value.data_ptr() for value in values]
    used_tables = tables[:4] if entry == 'native_gate_forward' else tables
    args += [value.data_ptr() for value in used_tables]
    args += [value.data_ptr() if value is not None else 0 for value in outputs]
    args.append(rows)
    # Callers own the packed tensors, which stay alive until launch submission.
    runtime().launch([(kernel, args, (count + 127)//128, 1, 128, 0)],
                     torch.cuda.current_stream(x.device).cuda_stream)


class FastGateBackwardFunction(torch.autograd.Function):
    """Original first-backward context; native launcher for both adjoints."""
    @staticmethod
    def forward(ctx, x, gy, out_src, out_gate, out_act, out_scale, in_out, in_role):
        rows = math.prod(x.shape[:-1])
        x2 = x.contiguous().reshape(rows, x.shape[-1])
        gy2 = gy.contiguous().reshape(rows, out_src.numel())
        dx = x.new_empty(x.shape)
        tables = (out_src, out_gate, out_act, out_scale, in_out, in_role)
        _launch('native_gate_backward', x2, tables, (x2, gy2), (dx,), rows, dx.numel())
        ctx.save_for_backward(x, gy, *tables)
        ctx.set_materialize_grads(False)
        return dx

    @staticmethod
    def backward(ctx, u):
        if u is None:
            return (None,) * 8
        x, gy, *tables = ctx.saved_tensors
        rows = math.prod(x.shape[:-1])
        x2 = x.contiguous().reshape(rows, x.shape[-1])
        gy2 = gy.contiguous().reshape(rows, tables[0].numel())
        u2 = u.contiguous().reshape_as(x2)
        dx = x.new_empty(x.shape) if ctx.needs_input_grad[0] else None
        dgy = gy.new_empty(gy.shape) if ctx.needs_input_grad[1] else None
        count = max(dx.numel() if dx is not None else 0, dgy.numel() if dgy is not None else 0)
        _launch('native_gate_double', x2, tables, (x2, gy2, u2), (dx, dgy), rows, count)
        return dx, dgy, None, None, None, None, None, None


class FastGateFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, out_src, out_gate, out_act, out_scale, in_out, in_role):
        rows = math.prod(x.shape[:-1])
        x2 = x.contiguous().reshape(rows, x.shape[-1])
        outdim = out_src.numel()
        y = x.new_empty((*x.shape[:-1], outdim))
        tables = (out_src, out_gate, out_act, out_scale, in_out, in_role)
        # Forward has no use for the fan-in tables; the ABI uses only 4 maps.
        _launch('native_gate_forward', x2, tables, (x2,), (y,), rows, y.numel())
        # Keep the public input, not a view made inside this no-grad forward.
        ctx.save_for_backward(x, *tables)
        ctx.set_materialize_grads(False)
        return y

    @staticmethod
    def backward(ctx, gy):
        if gy is None or not ctx.needs_input_grad[0]:
            return (None,) * 7
        x, *tables = ctx.saved_tensors
        return FastGateBackwardFunction.apply(x, gy, *tables), None, None, None, None, None, None


__all__ = ['FastGateFunction', 'FastGateBackwardFunction', 'act_code',
           'ACT_ID', 'ACT_SILU', 'ACT_SIGMOID', 'ACT_TANH']
