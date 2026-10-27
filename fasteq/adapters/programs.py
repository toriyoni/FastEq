"""Interface translation to FastEq-owned equivariant kernels."""

import math
from types import SimpleNamespace

import torch

from fasteq.ops.indexed_linear import IndexedLinear
from fasteq.ops.modulation import ModulatedResidualPlan
from fasteq.ops.native_invariants import InvariantPlan
from fasteq.ops.native_bias import native_bias_add

from .cache import TensorCache, indices


class LinearPlan:
    backend = "FastEq indexed CUDA Linear (general paths and full gradients)"

    def __init__(self, module, o2=False):
        self.o2 = o2
        self.din, self.dout = module.irreps_in.dim, module.irreps_out.dim
        self.shared = module.shared_weights
        instructions = tuple(i for i in module.instructions if i.i_in >= 0)
        descriptor = SimpleNamespace(
            irreps_in=module.irreps_in,
            irreps_out=module.irreps_out,
            instructions=instructions,
            weight_numel=module.weight_numel,
        )
        self.element = IndexedLinear(descriptor, layout="ir_mul" if o2 else "mul_ir")
        self.bias_map = [-1] * self.dout
        offset = 0
        for instruction in module.instructions:
            if instruction.i_in >= 0:
                continue
            sl = module.irreps_out.slices()[instruction.i_out]
            size = math.prod(instruction.path_shape)
            self.bias_map[sl.start : sl.stop] = list(range(offset, offset + size))
            offset += size
        self.bias_numel = offset
        self.constants = TensorCache()

    def __call__(self, module, x, weight=None, bias=None):
        if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
            raise TypeError("FastEq linear requires GPU FP32/FP64")
        if x.ndim < 1 or x.shape[-1] != self.din:
            raise ValueError(f"Linear features must end in {self.din}")
        if self.o2:
            weight = module._get_weights(weight)
        elif weight is None:
            if module.weight_numel and not module.internal_weights:
                raise ValueError("External linear weights are required")
            weight = module.weight
        if bias is None:
            if self.bias_numel and not module.internal_weights:
                raise RuntimeError(
                    "Biases must be provided for external linear weights"
                )
            bias = module.bias
        if bias is not None and (bias.ndim < 1 or bias.shape[-1] != self.bias_numel):
            raise ValueError(f"Bias trailing dimension must be {self.bias_numel}")
        shared = math.prod(weight.shape[:-1]) == 1 if self.o2 else self.shared
        bias_shared = (
            True if not self.o2 else bias is None or math.prod(bias.shape[:-1]) == 1
        )
        shape = torch.broadcast_shapes(
            x.shape[:-1],
            () if shared else weight.shape[:-1],
            () if bias_shared else bias.shape[:-1],
        )
        n = math.prod(shape)
        features = x.expand(*shape, self.din).reshape(n, self.din)
        weights = (
            weight.reshape(1, module.weight_numel)
            if shared
            else weight.expand(*shape, module.weight_numel).reshape(
                n, module.weight_numel
            )
        )
        output = self.element(features, weights, indices(x, n, sequence=not shared))
        if self.bias_numel:
            output = native_bias_add(output, bias, self.bias_map, shape,
                                     shared=bias_shared)
        return output.reshape(*shape, self.dout)


class NormPlan:
    """Delay loading extended FastEq norm until a norm is encountered."""

    def __new__(cls, module, tace=False):
        from fasteq.ops.equivariant_norm import NormPlan as NativeNormPlan

        return NativeNormPlan(module, tace=tace)


class FFNPlan:
    backend = "FastEq invariant kernels + fused residual modulation (Gate/Norm primitives)"

    def __init__(self, module):
        self.norm = InvariantPlan(module.irreps)
        self.modulate = ModulatedResidualPlan(module.irreps)

    def __call__(self, module, scalar_features, irreps_features):
        projected = module.linear1(irreps_features)
        invariant = self.norm(projected)
        context = module.mlp(torch.cat([invariant, scalar_features], -1))
        scalars, gates = context.split([module.scalar_dim, module.num_ch], -1)
        return scalar_features + scalars, self.modulate(
            projected, gates, irreps_features
        )
