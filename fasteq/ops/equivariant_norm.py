"""FastEq normalization for TACE/Prophet, including analytic double backward.

The weighted-moment and cross-row reduction primitives are reused from FastEq
d37eaca. This extends those primitives to FP64, ragged irrep multiplicities,
source-specific centering/weights and a second derivative. EQX is not imported.
"""

import math

import torch
import triton
import triton.language as tl
from torch.autograd.function import once_differentiable
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from ..cuda.triton_launch import launch_triton
from ..triton.fused_equivariant_layer_norm import _moment, _reduce_rows


@triton.jit
def _project(value, ids, CENTERS: tl.constexpr):
    result = value
    for i in tl.static_range(len(CENTERS)):
        member = (ids >= CENTERS[i][0]) & (ids < CENTERS[i][0] + CENTERS[i][1])
        mean = tl.sum(tl.where(member, value, 0.0), 0) / CENTERS[i][1]
        result = tl.where(member, value - mean, result)
    return result


@triton.jit
def _update_parameter(result, PTR, ids, START: tl.constexpr, CHANNELS: tl.constexpr,
                      DIMENSION: tl.constexpr, BASE: tl.constexpr,
                      IR_MUL: tl.constexpr, FP64: tl.constexpr):
    channel = ((ids - START) % CHANNELS if IR_MUL else (ids - START) // DIMENSION)
    member = (ids >= START) & (ids < START + CHANNELS * DIMENSION)
    value = tl.load(PTR + BASE + channel, member, 0.0)
    return tl.where(member, value.to(tl.float64 if FP64 else tl.float32), result)


@triton.jit
def _parameters(PTRS, ids, PARTS: tl.constexpr, IR_MUL: tl.constexpr,
                BIAS: tl.constexpr, ENABLED: tl.constexpr, DEFAULT: tl.constexpr,
                FP64: tl.constexpr):
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    result = tl.full(ids.shape, DEFAULT, dtype)
    for i in tl.static_range(len(PARTS)):
        if PARTS[i][1] > 0:
            if BIAS:
                if PARTS[i][5] >= 0:  # noqa: SIM102 - constexpr pruning protects an empty ENABLED tuple.
                    if ENABLED[PARTS[i][5]]:
                        result = _update_parameter(result, PTRS[PARTS[i][5]], ids,
                            PARTS[i][0], PARTS[i][1], PARTS[i][2], PARTS[i][6], IR_MUL, FP64)
            else:
                if PARTS[i][3] >= 0:  # noqa: SIM102 - constexpr pruning protects an empty ENABLED tuple.
                    if ENABLED[PARTS[i][3]]:
                        result = _update_parameter(result, PTRS[PARTS[i][3]], ids,
                            PARTS[i][0], PARTS[i][1], PARTS[i][2], PARTS[i][4], IR_MUL, FP64)
    return result


@triton.jit
def _part_partial(value, OUTPUT, row, WIDTH: tl.constexpr,
                  START: tl.constexpr, CHANNELS: tl.constexpr, DIMENSION: tl.constexpr,
                  OFFSET: tl.constexpr, IR_MUL: tl.constexpr):
    c = tl.arange(0, triton.next_power_of_2(CHANNELS))
    m = tl.arange(0, triton.next_power_of_2(DIMENSION))
    member = (c[:, None] < CHANNELS) & (m[None, :] < DIMENSION)
    index = START + (m[None, :] * CHANNELS + c[:, None] if IR_MUL
                     else c[:, None] * DIMENSION + m[None, :])
    index = tl.where(member, index, 0)
    selected = tl.gather(value, index.reshape((c.shape[0] * m.shape[0],)), 0)
    selected = selected.reshape((c.shape[0], m.shape[0]))
    partial = tl.sum(tl.where(member, selected, 0.0), 1)
    tl.store(OUTPUT + row * WIDTH + OFFSET + c, partial, c < CHANNELS)


@triton.jit
def _gamma_partials(value, ids, OUTPUT, row, WIDTH: tl.constexpr,
                    PARTS: tl.constexpr, OFFSETS: tl.constexpr, IR_MUL: tl.constexpr):
    for i in tl.static_range(len(PARTS)):
        if PARTS[i][1] > 0 and PARTS[i][3] >= 0:
            _part_partial(value, OUTPUT, row, WIDTH, PARTS[i][0], PARTS[i][1],
                          PARTS[i][2], OFFSETS[PARTS[i][3]] + PARTS[i][4], IR_MUL)


@triton.jit
def _selected_gamma_partials(value, OUTPUT, row, WIDTH: tl.constexpr,
                             PARTS: tl.constexpr, OFFSETS: tl.constexpr,
                             IR_MUL: tl.constexpr, FLAGS: tl.constexpr):
    for i in tl.static_range(len(PARTS)):
        if PARTS[i][1] > 0 and PARTS[i][3] >= 0:
            if FLAGS[PARTS[i][3]]:
                _part_partial(value, OUTPUT, row, WIDTH, PARTS[i][0], PARTS[i][1],
                              PARTS[i][2], OFFSETS[PARTS[i][3]] + PARTS[i][4], IR_MUL)


@triton.jit
def _forward_kernel(X, COEF, GAMMA, BETA, Y, D: tl.constexpr, BLOCK: tl.constexpr,
                    PARTS: tl.constexpr, CENTERS: tl.constexpr, IR_MUL: tl.constexpr,
                    GFLAGS: tl.constexpr, BFLAGS: tl.constexpr, EPS: tl.constexpr,
                    FP64: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ids = tl.arange(0, BLOCK)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    x = tl.load(X + row * D + ids, ids < D, 0.0).to(dtype)
    b = tl.load(COEF + ids, ids < D, 0.0).to(dtype)
    v = _project(x, ids, CENTERS)
    # Reuse FastEq's weighted-moment implementation: one ragged component
    # per row of the conceptual [D,1] tile, with its exact source coefficient.
    variance = _moment(v[:, None] * v[:, None], b, 1.0,
                       1, 'components_first', False)
    r = tl.rsqrt(variance + EPS)
    gamma = _parameters(GAMMA, ids, PARTS, IR_MUL, False, GFLAGS, 1.0, FP64)
    beta = _parameters(BETA, ids, PARTS, IR_MUL, True, BFLAGS, 0.0, FP64)
    tl.store(Y + row * D + ids, v * r * gamma + beta, ids < D)


@triton.jit
def _backward_kernel(X, DY, COEF, GAMMA, DX, PARTIALS, D: tl.constexpr,
                     BLOCK: tl.constexpr, PARTS: tl.constexpr, CENTERS: tl.constexpr,
                     IR_MUL: tl.constexpr, GFLAGS: tl.constexpr,
                     GOFFSETS: tl.constexpr, BOFFSETS: tl.constexpr,
                     WIDTH: tl.constexpr, EPS: tl.constexpr, FP64: tl.constexpr,
                     DXFLAG: tl.constexpr, GGRADFLAGS: tl.constexpr,
                     BGRADFLAGS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ids = tl.arange(0, BLOCK)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    x = tl.load(X + row * D + ids, ids < D, 0.0).to(dtype)
    dy = tl.load(DY + row * D + ids, ids < D, 0.0).to(dtype)
    b = tl.load(COEF + ids, ids < D, 0.0).to(dtype)
    v = _project(x, ids, CENTERS)
    variance = _moment(v[:, None] * v[:, None], b, 1.0, 1, 'components_first', False)
    r = tl.rsqrt(variance + EPS)
    if DXFLAG:
        gamma = _parameters(GAMMA, ids, PARTS, IR_MUL, False, GFLAGS, 1.0, FP64)
        g = dy * gamma
        a = tl.sum(g * v, 0)
        dx = _project(r * g - r * r * r * b * v * a, ids, CENTERS)
        tl.store(DX + row * D + ids, dx, ids < D)
    _selected_gamma_partials(dy * v * r, PARTIALS, row, WIDTH, PARTS,
                             GOFFSETS, IR_MUL, GGRADFLAGS)
    for i in tl.static_range(len(PARTS)):
        if PARTS[i][1] > 0 and PARTS[i][5] >= 0:
            if BGRADFLAGS[PARTS[i][5]]:
                c = tl.arange(0, triton.next_power_of_2(PARTS[i][1]))
                gradient = tl.load(DY + row * D + PARTS[i][0] + c, c < PARTS[i][1], 0.0)
                tl.store(PARTIALS + row * WIDTH + BOFFSETS[PARTS[i][5]] + PARTS[i][6] + c,
                         gradient, c < PARTS[i][1])


@triton.jit
def _double_kernel(X, DY, COEF, GAMMA, U, H, K, DDX, DDY, PARTIALS,
                   D: tl.constexpr, BLOCK: tl.constexpr, PARTS: tl.constexpr,
                   CENTERS: tl.constexpr, IR_MUL: tl.constexpr,
                   GFLAGS: tl.constexpr, HFLAGS: tl.constexpr, KFLAGS: tl.constexpr,
                   HAS_U: tl.constexpr, GOFFSETS: tl.constexpr, WIDTH: tl.constexpr,
                   EPS: tl.constexpr, FP64: tl.constexpr,
                   DDXFLAG: tl.constexpr, DDYFLAG: tl.constexpr,
                   GGRADFLAGS: tl.constexpr):
    row = tl.program_id(0).to(tl.int64)
    ids = tl.arange(0, BLOCK)
    dtype: tl.constexpr = tl.float64 if FP64 else tl.float32
    x = tl.load(X + row * D + ids, ids < D, 0.0).to(dtype)
    dy = tl.load(DY + row * D + ids, ids < D, 0.0).to(dtype)
    b = tl.load(COEF + ids, ids < D, 0.0).to(dtype)
    u = tl.load(U + row * D + ids, (ids < D) & HAS_U, 0.0).to(dtype)
    v, t = _project(x, ids, CENTERS), _project(u, ids, CENTERS)
    variance = _moment(v[:, None] * v[:, None], b, 1.0, 1, 'components_first', False)
    r = tl.rsqrt(variance + EPS)
    gamma = _parameters(GAMMA, ids, PARTS, IR_MUL, False, GFLAGS, 1.0, FP64)
    h = _parameters(H, ids, PARTS, IR_MUL, False, HFLAGS, 0.0, FP64)
    k = _parameters(K, ids, PARTS, IR_MUL, True, KFLAGS, 0.0, FP64)
    g = dy * gamma
    a, s = tl.sum(g * v, 0), tl.sum(t * b * v, 0)
    c = tl.sum(t * g + dy * h * v, 0)
    r3 = r * r * r
    ddx = _project(r * dy * h - b * v * r3 * c
                   + 3.0 * b * v * r3 * r * r * a * s
                   - r3 * (s * g + a * b * t), ids, CENTERS)
    ddy = r * (t * gamma + h * v) - r3 * s * v * gamma + k
    if DDXFLAG:
        tl.store(DDX + row * D + ids, ddx, ids < D)
    if DDYFLAG:
        tl.store(DDY + row * D + ids, ddy, ids < D)
    _selected_gamma_partials(dy * (r * t - r3 * s * v), PARTIALS, row,
                             WIDTH, PARTS, GOFFSETS, IR_MUL, GGRADFLAGS)


def _requested_gradients(fallback):
    """Respect the current autograd task, rather than requires_grad alone."""
    if (
        torch.compiler.is_compiling()
        or is_in_torch_dispatch_mode()
        or torch._C._are_functorch_transforms_active()
    ):
        return tuple(fallback)
    from fasteq.cuda.jit import runtime
    requested = runtime().norm_gradient_mask(len(fallback))
    return tuple(requested) if len(requested) == len(fallback) else tuple(fallback)


def _partial_layout(widths, requested):
    offsets, active_widths, width = [], [], 0
    for size, needed in zip(widths, requested, strict=True):
        offsets.append(width)
        if needed:
            active_widths.append(size)
            width += size
    return tuple(offsets), tuple(active_widths), width


def _parameter_gradients(partials, parameters, requested, widths, width):
    if not width:
        return [None] * len(parameters)
    combined = _reduce_rows(partials, width,
                            'fp64' if partials.dtype == torch.float64 else 'fp32')
    pieces = iter(combined.split(widths))
    return [next(pieces).reshape(parameter.shape) if needed else None
            for parameter, needed in zip(parameters, requested, strict=True)]


class _Norm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, coefficient, plan, *parameters):
        ctx.plan = plan
        ctx.save_for_backward(x, coefficient, *parameters)
        ctx.set_materialize_grads(False)
        y = torch.empty_like(x)
        if x.shape[0]:
            with torch.cuda.device(x.device):
                plan.launch(_forward_kernel, (x, coefficient, parameters[:plan.ng],
                    parameters[plan.ng:], y), x, plan.kwargs(x))
        return y

    @staticmethod
    def backward(ctx, dy):
        x, coefficient, *parameters = ctx.saved_tensors
        if dy is None:
            return (None,) * (3 + len(parameters))
        requested = _requested_gradients(
            (ctx.needs_input_grad[0], False, *ctx.needs_input_grad[3:])
        )
        gradients = _NormBackward.apply(
            x, dy, coefficient, ctx.plan, (requested[0], *requested[2:]), *parameters
        )
        return gradients[0], None, None, *gradients[1:]


class _NormBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, dy, coefficient, plan, requested, *parameters):
        ctx.plan = plan
        ctx.save_for_backward(x, dy, coefficient, *parameters)
        ctx.set_materialize_grads(False)
        dx = torch.empty_like(x) if requested[0] else x
        parameter_flags = tuple(requested[1:])
        offsets, widths, width = _partial_layout(plan.widths, parameter_flags)
        partials = x.new_empty((x.shape[0], width)) if width else x
        with torch.cuda.device(x.device):
            if x.shape[0]:
                kw = plan.kwargs(x)
                kw.pop("BFLAGS")
                plan.launch(_backward_kernel, (x, dy.contiguous(), coefficient,
                    parameters[:plan.ng], dx, partials), x, dict(GOFFSETS=offsets[:plan.ng],
                    BOFFSETS=offsets[plan.ng:], WIDTH=width, DXFLAG=requested[0],
                    GGRADFLAGS=parameter_flags[:plan.ng], BGRADFLAGS=parameter_flags[plan.ng:], **kw))
            grads = _parameter_gradients(partials, parameters, parameter_flags, widths, width)
        return (dx if requested[0] else None, *grads)

    @staticmethod
    @once_differentiable
    def backward(ctx, u, *seeds):
        x, dy, coefficient, *parameters = ctx.saved_tensors
        parameters = tuple(parameters)
        plan = ctx.plan
        requested = _requested_gradients(
            (ctx.needs_input_grad[0], ctx.needs_input_grad[1], False,
             *ctx.needs_input_grad[5:])
        )
        hflags = tuple(seed is not None for seed in seeds[:plan.ng])
        kflags = tuple(seed is not None for seed in seeds[plan.ng:])
        h = tuple(seed.contiguous() if seed is not None else parameter
                  for seed, parameter in zip(seeds[:plan.ng], parameters[:plan.ng]))
        k = tuple(seed.contiguous() if seed is not None else parameter
                  for seed, parameter in zip(seeds[plan.ng:], parameters[plan.ng:]))
        ddx = torch.empty_like(x) if requested[0] else x
        ddy = torch.empty_like(dy) if requested[1] else dy
        gamma_flags = requested[3:3 + plan.ng]
        offsets, widths, width = _partial_layout(plan.widths[:plan.ng], gamma_flags)
        partials = x.new_empty((x.shape[0], width)) if width else x
        with torch.cuda.device(x.device):
            if x.shape[0]:
                kw = plan.kwargs(x)
                kw.pop("BFLAGS")
                plan.launch(_double_kernel, (x, dy.contiguous(), coefficient,
                    parameters[:plan.ng], u.contiguous() if u is not None else x,
                    h, k, ddx, ddy, partials), x, dict(HFLAGS=hflags, KFLAGS=kflags,
                    HAS_U=u is not None, GOFFSETS=offsets, WIDTH=width,
                    DDXFLAG=requested[0], DDYFLAG=requested[1], GGRADFLAGS=gamma_flags, **kw))
            gamma_grads = _parameter_gradients(
                partials, parameters[:plan.ng], gamma_flags, widths, width
            )
        return (ddx if requested[0] else None, ddy if requested[1] else None,
                None, None, None, *gamma_grads, *([None] * plan.nb))


def _entry(container, index):
    value = container._parameters.get(str(index))
    return value if value is not None else getattr(container, str(index))


class NormPlan:
    """Original TACE/Prophet normalization with live source parameters."""

    backend = "FastEq Triton Norm (d37eaca moment/reduction, FP64/ragged/double extension)"

    def __init__(self, module, tace=False):
        self.tace = tace
        parts = ([(module.num_channels, 2*degree+1) for degree in module.ls] if tace
                 else [(ir.mul, ir.ir.dim) for ir in module.irreps])
        self.shape = (sum(d for c, d in parts), module.num_channels) if tace else (sum(c*d for c, d in parts),)
        self.dim = math.prod(self.shape)
        if not self.dim:
            raise ValueError("Normalization requires a nonempty feature representation")
        self.ng = int(module.affine) if tace else len(parts) if module.affine else 0
        self.bias_indices = () if tace or module.affine_bias is None else tuple(
            i for i, value in enumerate(module.affine_bias) if value is not None)
        self.nb = int(module.affine_bias is not None) if tace else len(self.bias_indices)
        gwidths = ([sum(c for c,d in parts)] if tace else [c for c,d in parts]) if module.affine else []
        bwidths = [module.num_channels] if tace and self.nb else [parts[i][0] for i in self.bias_indices]
        self.widths = tuple(gwidths + bwidths)
        self.gwidth, self.width = sum(gwidths), sum(self.widths)
        self.goffsets = tuple(sum(gwidths[:i]) for i in range(self.ng))
        self.boffsets = tuple(self.gwidth + sum(bwidths[:i]) for i in range(self.nb))
        descriptor, centers, offset, angle, goffset = [], [], 0, 0, 0
        for i, (c, d) in enumerate(parts):
            center = module.centering and (angle == module.scalar_index if tace else d == 1)
            if center and c:
                centers.append((offset, c))
            bs = (0 if center and self.nb else -1) if tace else (
                self.bias_indices.index(i) if i in self.bias_indices else -1)
            descriptor.append((offset, c, d, 0 if module.affine and tace else i if module.affine else -1,
                               goffset if tace else 0, bs, 0))
            offset += c*d
            angle += d
            goffset += c
        self.parts, self.centers = tuple(descriptor), tuple(centers)
        self.eps = module.eps
        self._cache = {}
        self._launchers = {}

    def __getstate__(self):
        state = self.__dict__.copy()
        state["_cache"] = {}
        state["_launchers"] = {}
        return state

    def launch(self, fn, arguments, x, kwargs):
        launch_triton(fn, arguments, x, kwargs, self._launchers)

    def kwargs(self, x):
        return {"D": self.dim, "BLOCK": triton.next_power_of_2(self.dim), "PARTS": self.parts,
                "CENTERS": self.centers, "IR_MUL": self.tace, "GFLAGS": (True,)*self.ng,
                "BFLAGS": (True,)*self.nb, "EPS": self.eps, "FP64": x.dtype == torch.float64,
                "num_warps": 4 if self.dim <= 2048 else 8, "enable_fp_fusion": False}

    def __call__(self, module, x):
        if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
            raise TypeError("FastEq normalization requires GPU float32/float64")
        if x.ndim < len(self.shape) or tuple(x.shape[-len(self.shape):]) != self.shape:
            raise ValueError(f"Normalization must end in shape {self.shape}")
        state = module.balance_degree_weight if self.tace and module.std_balance_degrees else None
        version = state._version if state is not None and not state.is_inference() else 0
        key = (x.device, x.dtype, id(state), version,
               module.normalization if self.tace else None, module.std_balance_degrees)
        cache = not torch.compiler.is_compiling() and not is_in_torch_dispatch_mode()
        cache &= state is None or not state.is_inference()
        if not cache or key not in self._cache:
            if self.tace:
                if module.normalization == 'norm':
                    coefficients = [1/module.num_channels] * self.dim
                elif module.std_balance_degrees:
                    coefficients = [value/module.num_channels for value in state.detach().flatten().cpu().tolist()
                                    for _ in range(module.num_channels)]
                else:
                    coefficients = [1/self.dim] * self.dim
            else:
                if not module.std_balance_degrees:
                    raise NotImplementedError("Prophet unbalanced norm is unsupported upstream")
                coefficients = [1/(d*d*len(self.parts)) for _, c, d, *_ in self.parts for _ in range(c*d)]
            with torch.inference_mode(False):
                coefficient = torch.tensor(coefficients, device=x.device, dtype=x.dtype)
            if cache:
                self._cache[key] = coefficient
        else:
            coefficient = self._cache[key]
        gamma = ([module.affine_weight] if self.ng else []) if self.tace else [
            _entry(module.affine_weight, i) for i in range(self.ng)]
        beta = ([module.affine_bias] if self.nb else []) if self.tace else [
            _entry(module.affine_bias, i) for i in self.bias_indices]
        parameters = tuple(parameter.contiguous() for parameter in gamma + beta)
        if any(parameter.dtype != x.dtype or parameter.device != x.device for parameter in parameters):
            raise ValueError("Normalization parameters must share the input dtype/device")
        flat = x.contiguous().reshape(-1, self.dim)
        return _Norm.apply(flat, coefficient, self, *parameters).reshape(x.shape)


__all__ = ["NormPlan"]
