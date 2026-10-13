"""FastEq per-copy squared norms for equivariant scalar conditioning.

These are the invariant extraction step of FastEq's Gate/normalization family,
with explicit first and second derivatives for force-loss training.
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _forward(
    X,
    BASE,
    WIDTH,
    Y,
    N: tl.constexpr,
    D: tl.constexpr,
    C: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1) * B + tl.arange(0, B)
    angle = tl.arange(0, A)
    base = tl.load(BASE + channel, channel < C, 0)
    width = tl.load(WIDTH + channel, channel < C, 1)
    value = tl.load(
        X + row * D + base[:, None] + angle[None, :],
        (channel[:, None] < C) & (angle[None, :] < width[:, None]),
        0,
    )
    result = tl.sum(value * value, 1) / width
    tl.store(Y + row * C + channel, result, channel < C)


@triton.jit
def _backward(X, G, INDEX, WIDTH, Y, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    feature = tl.program_id(1) * B + tl.arange(0, B)
    channel = tl.load(INDEX + feature, feature < D, 0)
    width = tl.load(WIDTH + channel, feature < D, 1)
    x = tl.load(X + row * D + feature, feature < D, 0)
    g = tl.load(G + row * C + channel, feature < D, 0)
    tl.store(Y + row * D + feature, 2 * x * g / width, feature < D)


@triton.jit
def _second_x(G, U, INDEX, WIDTH, Y, D: tl.constexpr, C: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    feature = tl.program_id(1) * B + tl.arange(0, B)
    channel = tl.load(INDEX + feature, feature < D, 0)
    width = tl.load(WIDTH + channel, feature < D, 1)
    g = tl.load(G + row * C + channel, feature < D, 0)
    u = tl.load(U + row * D + feature, feature < D, 0)
    tl.store(Y + row * D + feature, 2 * u * g / width, feature < D)


@triton.jit
def _second_g(
    X,
    U,
    BASE,
    WIDTH,
    Y,
    D: tl.constexpr,
    C: tl.constexpr,
    A: tl.constexpr,
    B: tl.constexpr,
):
    row = tl.program_id(0)
    channel = tl.program_id(1) * B + tl.arange(0, B)
    angle = tl.arange(0, A)
    base = tl.load(BASE + channel, channel < C, 0)
    width = tl.load(WIDTH + channel, channel < C, 1)
    offsets = row * D + base[:, None] + angle[None, :]
    mask = (channel[:, None] < C) & (angle[None, :] < width[:, None])
    x = tl.load(X + offsets, mask, 0)
    u = tl.load(U + offsets, mask, 0)
    tl.store(Y + row * C + channel, 2 * tl.sum(x * u, 1) / width, channel < C)


class _First(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gradient, plan, metadata):
        _base, widths, index = metadata
        result = torch.empty_like(x)
        if x.shape[0]:
            _backward[(x.shape[0], triton.cdiv(plan.dim, 256))](
                x.contiguous(),
                gradient.contiguous(),
                index,
                widths,
                result,
                plan.dim,
                plan.channels,
                256,
            )
        ctx.plan, ctx.tables = plan, metadata
        ctx.save_for_backward(x, gradient)
        return result

    @staticmethod
    def backward(ctx, direction):
        x, gradient = ctx.saved_tensors
        plan = ctx.plan
        base, widths, index = ctx.tables
        direction = direction.contiguous()
        dx = torch.empty_like(x) if ctx.needs_input_grad[0] else None
        dg = torch.empty_like(gradient) if ctx.needs_input_grad[1] else None
        if x.shape[0]:
            if dx is not None:
                _second_x[(x.shape[0], triton.cdiv(plan.dim, 256))](
                    gradient.contiguous(),
                    direction,
                    index,
                    widths,
                    dx,
                    plan.dim,
                    plan.channels,
                    256,
                )
            if dg is not None:
                _second_g[(x.shape[0], triton.cdiv(plan.channels, 32))](
                    x.contiguous(),
                    direction,
                    base,
                    widths,
                    dg,
                    plan.dim,
                    plan.channels,
                    plan.angle,
                    32,
                )
        return dx, dg, None, None


class _Norms(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, plan, metadata):
        base, widths, _ = metadata
        result = x.new_empty(x.shape[0], plan.channels)
        if x.shape[0]:
            _forward[(x.shape[0], triton.cdiv(plan.channels, 32))](
                x.contiguous(),
                base,
                widths,
                result,
                x.shape[0],
                plan.dim,
                plan.channels,
                plan.angle,
                32,
            )
        ctx.plan, ctx.tables = plan, metadata
        ctx.save_for_backward(x)
        return result

    @staticmethod
    def backward(ctx, gradient):
        (x,) = ctx.saved_tensors
        return _First.apply(x, gradient, ctx.plan, ctx.tables), None, None


class InvariantPlan:
    def __init__(self, irreps):
        bases, widths, index, offset = [], [], [], 0
        for channels, ir in irreps:
            for channel in range(channels):
                index.extend([len(bases)] * ir.dim)
                bases.append(offset + channel * ir.dim)
                widths.append(ir.dim)
            offset += channels * ir.dim
        self.dim, self.channels = offset, len(bases)
        self.angle = triton.next_power_of_2(max(widths, default=1))
        self._data = tuple(bases), tuple(widths), tuple(index)
        self._cache = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        return state

    def __call__(self, x):
        if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
            raise TypeError("FastEq invariants require GPU FP32/FP64")
        if x.shape != (x.shape[0], self.dim):
            raise ValueError("Invariant features must use packed mul_ir layout")
        if x.device not in self._cache:
            with torch.inference_mode(False), torch.no_grad():
                self._cache[x.device] = tuple(
                    torch.tensor(data, device=x.device, dtype=torch.int32)
                    for data in self._data
                )
        return _Norms.apply(x, self, self._cache[x.device])
