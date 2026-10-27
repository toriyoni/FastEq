"""FastEq residual/Gate/RMS fusion with full first and second derivatives."""

import math

import torch
import triton
from torch.autograd.function import once_differentiable
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

from .native_gate_norm import launch as launch_native
from ..triton import gate_norm as kernels
from .equivariant_norm import (
    NormPlan,
    _parameter_gradients,
    _partial_layout,
    _requested_gradients,
)


class _GateNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, skip, plan, metadata, *gamma):
        ctx.plan, ctx.tables = plan, metadata
        ctx.save_for_backward(projected, skip, *gamma)
        ctx.set_materialize_grads(False)
        result = projected.new_empty(projected.shape[0], plan.outdim)
        if projected.shape[0]:
            with torch.cuda.device(projected.device):
                plan.launch(
                    kernels.forward,
                    (projected, skip, *metadata[:4], metadata[6], gamma, result),
                    projected,
                )
        return result

    @staticmethod
    def backward(ctx, gradient):
        projected, skip, *gamma = ctx.saved_tensors
        if gradient is None:
            return (None,) * (4 + len(gamma))
        requested = _requested_gradients(
            (
                ctx.needs_input_grad[0],
                ctx.needs_input_grad[1],
                *ctx.needs_input_grad[4:],
            )
        )
        gradients = _GateNormBackward.apply(
            projected,
            skip,
            gradient,
            ctx.plan,
            ctx.tables,
            (requested[0] or requested[1], *requested[2:]),
            *gamma,
        )
        dz = gradients[0]
        return (
            dz if requested[0] else None,
            dz if requested[1] else None,
            None,
            None,
            *gradients[1:],
        )


class _GateNormBackward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, projected, skip, gradient, plan, metadata, requested, *gamma):
        ctx.plan, ctx.tables = plan, metadata
        ctx.save_for_backward(projected, skip, gradient, *gamma)
        ctx.set_materialize_grads(False)
        dz = torch.empty_like(projected) if requested[0] else projected
        gamma_flags = tuple(requested[1:])
        offsets, widths, width = _partial_layout(plan.norm.widths, gamma_flags)
        partials = (
            projected.new_empty(projected.shape[0], width) if width else projected
        )
        with torch.cuda.device(projected.device):
            if projected.shape[0]:
                if requested[0] and plan.rematerialize_adjoints(projected):
                    plan.launch(
                        kernels.backward_direct,
                        (
                            projected,
                            skip,
                            gradient.contiguous(),
                            *metadata,
                            gamma,
                            dz,
                            partials,
                        ),
                        projected,
                        MAX_FAN=plan.max_fan,
                        GOFFSETS=offsets,
                        WIDTH=width,
                        DZFLAG=requested[0],
                        GGRADFLAGS=gamma_flags,
                    )
                else:
                    plan.launch(
                        kernels.backward,
                        (
                            projected,
                            skip,
                            gradient.contiguous(),
                            *metadata,
                            gamma,
                            dz,
                            partials,
                        ),
                        projected,
                        MAX_FAN=plan.max_fan,
                        GOFFSETS=offsets,
                        WIDTH=width,
                        DZFLAG=requested[0],
                        GGRADFLAGS=gamma_flags,
                    )
            grads = _parameter_gradients(partials, gamma, gamma_flags, widths, width)
        return (dz if requested[0] else None, *grads)

    @staticmethod
    @once_differentiable
    def backward(ctx, direction, *seeds):
        projected, skip, gradient, *gamma = ctx.saved_tensors
        gamma = tuple(gamma)
        plan, metadata = ctx.plan, ctx.tables
        requested = _requested_gradients(
            (
                ctx.needs_input_grad[0],
                ctx.needs_input_grad[1],
                ctx.needs_input_grad[2],
                *ctx.needs_input_grad[6:],
            )
        )
        flags = tuple(seed is not None for seed in seeds)
        h = tuple(
            seed.contiguous() if seed is not None else parameter
            for seed, parameter in zip(seeds, gamma)
        )
        dzflag = requested[0] or requested[1]
        ddz = torch.empty_like(projected) if dzflag else projected
        ddy = torch.empty_like(gradient) if requested[2] else gradient
        gamma_flags = requested[3:]
        offsets, widths, width = _partial_layout(plan.norm.widths, gamma_flags)
        partials = (
            projected.new_empty(projected.shape[0], width) if width else projected
        )
        with torch.cuda.device(projected.device):
            if projected.shape[0]:
                u = direction.contiguous() if direction is not None else projected
                if dzflag and plan.rematerialize_adjoints(projected):
                    plan.launch(
                        kernels.double_backward_direct,
                        (
                            projected,
                            skip,
                            gradient.contiguous(),
                            u,
                            h,
                            *metadata,
                            gamma,
                            ddz,
                            ddy,
                            partials,
                        ),
                        projected,
                        MAX_FAN=plan.max_fan,
                        HFLAGS=flags,
                        HAS_U=direction is not None,
                        GOFFSETS=offsets,
                        WIDTH=width,
                        DDZFLAG=dzflag,
                        DDYFLAG=requested[2],
                        GGRADFLAGS=gamma_flags,
                    )
                else:
                    plan.launch(
                        kernels.double_backward,
                        (
                            projected,
                            skip,
                            gradient.contiguous(),
                            u,
                            h,
                            *metadata,
                            gamma,
                            ddz,
                            ddy,
                            partials,
                        ),
                        projected,
                        MAX_FAN=plan.max_fan,
                        HFLAGS=flags,
                        HAS_U=direction is not None,
                        GOFFSETS=offsets,
                        WIDTH=width,
                        DDZFLAG=dzflag,
                        DDYFLAG=requested[2],
                        GGRADFLAGS=gamma_flags,
                    )
            grads = _parameter_gradients(partials, gamma, gamma_flags, widths, width)
        return (
            ddz if requested[0] else None,
            ddz if requested[1] else None,
            ddy if requested[2] else None,
            None,
            None,
            None,
            *grads,
        )


class GateNormPlan:
    """Noncentering Prophet RMS after Gate; parameters remain owned by the norm."""

    backend = "FastEq native CUDA fused residual/Gate/RMS (first and second gradients)"

    @staticmethod
    def supports(norm):
        return (
            not norm.centering
            and norm.std_balance_degrees
            and norm.affine_bias is None
            and norm.irreps.dim > 0
        )

    def __init__(self, gate, norm):
        if not self.supports(norm):
            raise ValueError(
                "Fused Gate/RMS requires noncentering balanced normalization without bias"
            )
        self.gate, self.norm = gate, NormPlan(norm)
        self.dim, self.outdim = gate.dim, gate.outdim
        if self.outdim != norm.irreps.dim:
            raise ValueError("Gate output must match normalization irreps")
        self.max_fan = gate.metadata[4].shape[1]
        self._metadata = (
            *gate.metadata,
            torch.tensor(
                [
                    1 / (ir.dim * ir.dim * len(norm.irreps))
                    for mul, ir in norm.irreps
                    for _ in range(mul * ir.dim)
                ],
                dtype=torch.float64,
            ),
        )
        self._cache = {}
        self._launchers = {}

    def __getstate__(self):
        result = self.__dict__.copy()
        result["_cache"] = {}
        result["_launchers"] = {}
        return result

    def kwargs(self, x):
        return {
            "DIN": self.dim,
            "DOUT": self.outdim,
            "BLOCK": triton.next_power_of_2(max(self.dim, self.outdim)),
            "PARTS": self.norm.parts,
            "GFLAGS": (True,) * self.norm.ng,
            "EPS": self.norm.eps,
            "FP64": x.dtype == torch.float64,
            "num_warps": 4,
            "enable_fp_fusion": False,
        }

    def launch(self, fn, arguments, x, **extra):
        """Reuse FastEq Gate/Norm native primitives in one fused row kernel."""
        kwargs = self.kwargs(x)
        if fn is kernels.backward_direct or fn is kernels.double_backward_direct:
            kwargs["num_warps"] = 16
        kwargs.update(extra)
        launch_native(self, fn, arguments, x, kwargs)

    def rematerialize_adjoints(self, x):
        """Avoid large FP64 cross-warp gathers; short rows keep fused gathers."""
        return x.dtype == torch.float64 and self.dim >= 512 and x.shape[0] >= 32

    def __call__(self, norm, projected, skip):
        if not self.supports(norm):
            raise ValueError(
                "Normalization configuration changed after fusion installation"
            )
        if (
            not projected.is_cuda
            or projected.dtype not in (torch.float32, torch.float64)
            or skip.device != projected.device
            or skip.dtype != projected.dtype
        ):
            raise TypeError("Fused Gate/RMS requires matching CUDA FP32/FP64 inputs")
        if (
            projected.ndim < 1
            or projected.shape[-1] != self.dim
            or skip.shape != projected.shape
        ):
            raise ValueError(
                "Projected and skip tensors must share the Gate input shape"
            )
        key = projected.device, projected.dtype
        cache = not torch.compiler.is_compiling() and not is_in_torch_dispatch_mode()
        if not cache or key not in self._cache:
            with torch.inference_mode(False), torch.no_grad():
                metadata = tuple(
                    value.to(
                        device=projected.device,
                        dtype=projected.dtype if i in (3, 6) else value.dtype,
                    )
                    for i, value in enumerate(self._metadata)
                )
            if cache:
                self._cache[key] = metadata
        else:
            metadata = self._cache[key]
        gamma = tuple(
            (
                norm.affine_weight._parameters.get(str(i))
                if norm.affine_weight._parameters.get(str(i)) is not None
                else getattr(norm.affine_weight, str(i))
            ).contiguous()
            for i in range(self.norm.ng)
        )
        if any(
            g.dtype != projected.dtype or g.device != projected.device for g in gamma
        ):
            raise ValueError(
                "Normalization parameters must match input dtype and device"
            )
        rows = math.prod(projected.shape[:-1])
        result = _GateNorm.apply(
            projected.contiguous().reshape(rows, self.dim),
            skip.contiguous().reshape(rows, self.dim),
            self,
            metadata,
            *gamma,
        )
        return result.reshape(*projected.shape[:-1], self.outdim)


__all__ = ["GateNormPlan"]
