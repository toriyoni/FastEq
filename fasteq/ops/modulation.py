"""FastEq Gate modulation and residual addition without packing features.

This reuses FastEq Gate activations, irrep-copy reductions and task-aware
autograd selection. The scalar MLP remains outside this equivariant operator.
"""

import math
import os

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from ..cuda.triton_launch import launch_triton, prepare_triton
from ..triton.fused_equivariant_gate import _activation
from .equivariant_norm import _part_partial, _requested_gradients


@triton.jit
def _forward(
    X,
    G,
    R,
    INDEX,
    Y,
    D: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    FP64: tl.constexpr,
    PARTS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    channel = tl.load(INDEX + i, i < D, 0)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    x = tl.load(X + row * D + i, i < D, 0).to(dtype)
    g = tl.load(G + row * C + channel, i < D, 0).to(dtype)
    residual = tl.load(R + row * D + i, i < D, 0).to(dtype)
    value = _activation(x, 0, 1.0) * g + residual
    tl.store(Y + row * D + i, value, i < D)


@triton.jit
def _gate_reduce(value, OUTPUT, row, C: tl.constexpr, PARTS: tl.constexpr):
    for k in tl.static_range(len(PARTS)):
        _part_partial(
            value,
            OUTPUT,
            row,
            C,
            PARTS[k][0],
            PARTS[k][1],
            PARTS[k][2],
            PARTS[k][3],
            False,
        )


@triton.jit
def _backward(
    X,
    G,
    DY,
    INDEX,
    DX,
    DG,
    D: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    PARTS: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_G: tl.constexpr,
    FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    dy = tl.load(DY + row * D + i, i < D, 0).to(dtype)
    if NEED_X:
        channel = tl.load(INDEX + i, i < D, 0)
        g = tl.load(G + row * C + channel, i < D, 0).to(dtype)
        tl.store(DX + row * D + i, dy * g, i < D)
    if NEED_G:
        x = tl.load(X + row * D + i, i < D, 0).to(dtype)
        _gate_reduce(x * dy, DG, row, C, PARTS)


@triton.jit
def _double(
    X,
    G,
    DY,
    U,
    V,
    INDEX,
    DDX,
    DDG,
    DDY,
    D: tl.constexpr,
    C: tl.constexpr,
    B: tl.constexpr,
    PARTS: tl.constexpr,
    HAS_U: tl.constexpr,
    HAS_V: tl.constexpr,
    NEED_X: tl.constexpr,
    NEED_G: tl.constexpr,
    NEED_DY: tl.constexpr,
    FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    i = tl.arange(0, B)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    channel = tl.load(INDEX + i, i < D, 0)
    dy = tl.load(DY + row * D + i, i < D, 0).to(dtype)
    u = tl.load(U + row * D + i, (i < D) & HAS_U, 0).to(dtype)
    v = tl.load(V + row * C + channel, (i < D) & HAS_V, 0).to(dtype)
    if NEED_X:
        tl.store(DDX + row * D + i, dy * v, i < D)
    if NEED_G:
        _gate_reduce(dy * u, DDG, row, C, PARTS)
    if NEED_DY:
        x = tl.load(X + row * D + i, i < D, 0).to(dtype)
        g = tl.load(G + row * C + channel, i < D, 0).to(dtype)
        tl.store(DDY + row * D + i, u * g + v * x, i < D)


class _Gradient(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gates, dy, plan, requested):
        ctx.plan = plan
        ctx.save_for_backward(x, gates, dy)
        ctx.set_materialize_grads(False)
        dx = (
            torch.empty_like(x, memory_format=torch.contiguous_format)
            if requested[0]
            else None
        )
        dg = (
            torch.empty_like(gates, memory_format=torch.contiguous_format)
            if requested[1]
            else None
        )
        rows = math.prod(x.shape[:-1])
        if rows and any(requested):
            with torch.cuda.device(x.device):
                flat = x.contiguous().reshape(rows, plan.dim)
                plan.launch(
                    _backward,
                    (
                        flat,
                        gates.contiguous(),
                        dy.contiguous(),
                        plan.index(x),
                        dx if dx is not None else x,
                        dg if dg is not None else gates,
                    ),
                    flat,
                    NEED_X=requested[0],
                    NEED_G=requested[1],
                )
        return dx, dg

    @staticmethod
    @once_differentiable
    def backward(ctx, u, v):
        x, gates, dy = ctx.saved_tensors
        requested = _requested_gradients(ctx.needs_input_grad[:3])
        # Direct x/g derivatives vanish when their corresponding seed is absent.
        requested = (
            requested[0] and v is not None,
            requested[1] and u is not None,
            requested[2],
        )
        outputs = tuple(
            torch.empty_like(value, memory_format=torch.contiguous_format)
            if flag
            else None
            for value, flag in zip((x, gates, dy), requested, strict=True)
        )
        rows = math.prod(x.shape[:-1])
        if rows and any(requested):
            with torch.cuda.device(x.device):
                flat = x.contiguous().reshape(rows, ctx.plan.dim)
                ctx.plan.launch(
                    _double,
                    (
                        flat,
                        gates.contiguous(),
                        dy.contiguous(),
                        u.contiguous() if u is not None else x,
                        v.contiguous() if v is not None else gates,
                        ctx.plan.index(x),
                        *(
                            value if value is not None else original
                            for value, original in zip(
                                outputs, (x, gates, dy), strict=True
                            )
                        ),
                    ),
                    flat,
                    HAS_U=u is not None,
                    HAS_V=v is not None,
                    NEED_X=requested[0],
                    NEED_G=requested[1],
                    NEED_DY=requested[2],
                )
        return *outputs, None, None


class _Modulation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, gates, residual, plan):
        ctx.plan = plan
        ctx.save_for_backward(x, gates)
        output = torch.empty_like(x, memory_format=torch.contiguous_format)
        rows = math.prod(x.shape[:-1])
        if rows:
            with torch.cuda.device(x.device):
                flat = x.contiguous().reshape(rows, plan.dim)
                plan.launch(
                    _forward,
                    (
                        flat,
                        gates.contiguous(),
                        residual.contiguous(),
                        plan.index(x),
                        output,
                    ),
                    flat,
                )
        return output

    @staticmethod
    def backward(ctx, gradient):
        x, gates = ctx.saved_tensors
        requested = _requested_gradients(ctx.needs_input_grad[:3])
        dx, dg = (
            _Gradient.apply(x, gates, gradient, ctx.plan, requested[:2])
            if any(requested[:2])
            else (None, None)
        )
        return dx, dg, gradient if requested[2] else None, None


class _NativeCompiler:
    """Independent immutable specification retained by native autograd nodes.

    It intentionally has no reference to the Python or C++ plan, so a graph can
    outlive its module without creating a reference cycle.
    """

    def __init__(self, plan, x):
        self.dim, self.channels = plan.dim, plan.channels
        self._index = plan.index(x)
        self._kwargs = plan.kwargs(x)
        self._launchers = plan._launchers

    def index(self, x):
        return self._index

    def kwargs(self, x):
        return self._kwargs.copy()

    def launch(self, fn, arguments, x, **extra):
        kwargs = self.kwargs(x)
        kwargs.update(extra)
        launch_triton(fn, arguments, x, kwargs, self._launchers)

    @staticmethod
    def _specialization(stage, requested, seeds):
        if stage == 0:
            return _forward, {}
        flags = {"NEED_X": bool(requested & 1), "NEED_G": bool(requested & 2)}
        if stage == 1:
            return _backward, flags
        flags.update(
            NEED_DY=bool(requested & 4), HAS_U=bool(seeds & 1), HAS_V=bool(seeds & 2)
        )
        return _double, flags

    def prepare(self, stage, arguments, requested, seeds):
        fn, extra = self._specialization(stage, requested, seeds)
        kwargs = self.kwargs(arguments[0])
        kwargs.update(extra)
        return prepare_triton(fn, arguments, arguments[0], kwargs, self._launchers)

    def fallback_forward(self, x, gates, residual):
        return _Modulation.apply(x, gates, residual, self)

    def fallback_first(self, x, gates, dy, requested):
        return _Gradient.apply(
            x, gates, dy, self, (bool(requested & 1), bool(requested & 2))
        )

    def fallback_double(self, arguments, requested, seeds):
        fn, extra = self._specialization(2, requested, seeds)
        self.launch(fn, arguments, arguments[0], **extra)


class ModulatedResidualPlan:
    backend = "FastEq fused Gate modulation/residual with selective analytic adjoints"

    def __init__(self, irreps):
        parts, index, offset, channel = [], [], 0, 0
        for mul, ir in irreps:
            parts.append((offset, mul, ir.dim, channel))
            index.extend(channel + u for u in range(mul) for _ in range(ir.dim))
            offset += mul * ir.dim
            channel += mul
        self.dim, self.channels = offset, channel
        self.parts, self._index, self._cache = tuple(parts), tuple(index), {}
        self._launchers = {}
        self._native = {}
        self._native_enabled = (
            os.environ.get("FASTEQ_EXPERIMENTAL_NATIVE_MODULATION") == "1"
        )

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        state["_launchers"] = {}
        state["_native"] = {}
        return state

    def index(self, x):
        cache = not torch.compiler.is_compiling() and not is_in_torch_dispatch_mode()
        if cache and x.device in self._cache:
            return self._cache[x.device]
        with torch.inference_mode(False), torch.no_grad():
            value = torch.tensor(self._index, device=x.device, dtype=torch.int32)
        if cache:
            self._cache[x.device] = value
        return value

    def kwargs(self, x):
        return {
            "D": self.dim,
            "C": self.channels,
            "B": triton.next_power_of_2(self.dim),
            "PARTS": self.parts,
            "FP64": x.dtype == torch.float64,
            "num_warps": 4,
            "enable_fp_fusion": False,
        }

    def launch(self, fn, arguments, x, **extra):
        kwargs = self.kwargs(x)
        kwargs.update(extra)
        launch_triton(fn, arguments, x, kwargs, self._launchers)

    def __call__(self, projected, gates, residual):
        if not projected.is_cuda or projected.dtype not in (
            torch.float32,
            torch.float64,
        ):
            raise TypeError("FastEq modulation requires CUDA FP32/FP64")
        if (
            projected.shape != residual.shape
            or gates.shape != (*projected.shape[:-1], self.channels)
            or projected.shape[-1] != self.dim
        ):
            raise ValueError(
                "Features/gates must have matching leading dimensions and the planned irreps"
            )
        if any(
            value.device != projected.device or value.dtype != projected.dtype
            for value in (gates, residual)
        ):
            raise ValueError("Modulation operands must share device and dtype")
        if (
            self._native_enabled
            and self.dim > 0
            and not torch.compiler.is_compiling()
            and not is_in_torch_dispatch_mode()
            and not torch._C._are_functorch_transforms_active()
        ):
            key = projected.device, projected.dtype
            if key not in self._native:
                from ..cuda.jit import runtime

                with torch.cuda.device(projected.device):
                    compiler = _NativeCompiler(self, projected)
                    self._native[key] = runtime().ModulationPlan(
                        compiler._index, self.dim, self.channels, compiler
                    )
            return self._native[key].forward(projected, gates, residual)
        return _Modulation.apply(projected, gates, residual, self)
