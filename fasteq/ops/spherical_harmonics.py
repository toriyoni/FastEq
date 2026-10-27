"""Extend FastEq's existing SH recurrence with native CUDA analytic adjoints."""

import ast
import math
from functools import lru_cache

import torch
from e3nn import o3

from ..triton.spherical_harmonics import (
    SphericalHarmonics as _MetadataSH,
    _required_assignments,
)
from .scalar_expression import ScalarProgram, _adjoint, _execute_raw, _parse, scalar_expression


class _SHExpression(torch.autograd.Function):
    """Use the existing CUDA DAG/transpose without custom-op redispatch."""

    @staticmethod
    def forward(ctx, metadata, *values):
        ctx.kernel_metadata = metadata
        ctx.save_for_backward(*values)
        ctx.set_materialize_grads(False)
        return tuple(_execute_raw(metadata, list(values)))

    @staticmethod
    def backward(ctx, *gradients):
        values = ctx.saved_tensors
        active = tuple(i for i, needed in enumerate(ctx.needs_input_grad[1:]) if needed)
        results = [None] * len(values)
        if active:
            _, groups, _ = _parse(ctx.kernel_metadata)
            seeds = [grad if grad is not None else values[0].new_zeros(values[0].shape[0], len(group))
                     for grad, group in zip(gradients, groups, strict=True)]
            adjoints = _SHExpression.apply(_adjoint(ctx.kernel_metadata, active), *values, *seeds)
            for i, grad in zip(active, adjoints, strict=True):
                results[i] = grad
        return (None, *results)


@lru_cache(maxsize=128)
def _program(degrees, normalize, normalization):
    """Reuse the FastEq polynomial DAG; the CUDA emitter/AD is shared with Wigner."""
    program=ScalarProgram()
    env={name:program.add("input",0,i) for i,name in enumerate(("x","y","z"))}
    if normalize:
        radius2=program.constant(0)
        for value in env.values():
            radius2=program.add("add",radius2,program.add("mul",value,value))
        epsilon2=program.constant(1e-24)
        # Clamp BEFORE sqrt so inactive derivatives never divide by sqrt(0).
        radius=program.add("sqrt",program.add("select",
            program.add("lt",radius2,epsilon2),epsilon2,radius2))
        env={name:program.add("div",value,radius) for name,value in env.items()}

    def expression(node):
        if isinstance(node,ast.Name): return env[node.id]
        if isinstance(node,ast.Constant): return program.constant(node.value)
        if isinstance(node,ast.UnaryOp) and isinstance(node.op,ast.USub):
            return program.add("neg",expression(node.operand))
        if isinstance(node,ast.BinOp):
            left,right=expression(node.left),expression(node.right)
            if isinstance(node.op,ast.Sub):
                return program.add("add",left,program.add("neg",right))
            op={ast.Add:"add",ast.Mult:"mul",ast.Div:"div"}.get(type(node.op))
            if op is not None:return program.add(op,left,right)
        if isinstance(node,ast.Call) and isinstance(node.func,ast.Attribute):
            name=node.func.attr
            if name=="ones_like": return program.constant(1)
            if name=="sqrt" and isinstance(node.func.value,ast.Name) and node.func.value.id=="math":
                value=program.nodes[expression(node.args[0])]
                if value[0]!="constant":raise ValueError("SH recurrence sqrt must be constant")
                return program.constant(math.sqrt(value[1]))
            if name=="pow" and isinstance(node.args[0],ast.Constant):
                power=node.args[0].value
                if not isinstance(power,int) or power<0:raise ValueError("SH polynomial power")
                value=expression(node.func.value);result=program.constant(1)
                for _ in range(power):result=program.add("mul",result,value)
                return result
        raise ValueError("Unsupported FastEq SH recurrence: "+ast.dump(node))

    for name,node in _required_assignments(degrees):env[name]=expression(node)
    outputs=[]
    for degree in degrees:
        divisor=(math.sqrt(4*math.pi) if normalization=="integral"
                 else math.sqrt(2*degree+1) if normalization=="norm" else 1)
        for i in range(2*degree+1):
            outputs.append(program.add("div",env[f"sh_{degree}_{i}"],program.constant(divisor)))
    return repr((tuple(program.nodes),(tuple(outputs),),(3,)))


class SphericalHarmonics(_MetadataSH):
    """Original FastEq API/recurrence with hand-written CUDA execution."""
    def forward(self,x):
        if not x.is_cuda:return self._reference_forward(x)
        if x.ndim<1 or x.shape[-1]!=3:raise ValueError("SH expects (...,3)")
        if x.dtype not in (torch.float32,torch.float64):
            raise TypeError("Native CUDA SH requires FP32/FP64")
        degrees=tuple(self._ls_list)
        metadata=_program(degrees,self.normalize,self.normalization)
        values=x.reshape(-1,3)
        # The established custom-op registration supplies fake/compiled and
        # transform dispatch. Eager plain tensors use the identical raw DAG.
        if (torch.compiler.is_compiling() or type(x) is not torch.Tensor
            or torch._C._are_functorch_transforms_active()
            or torch._C._len_torch_dispatch_stack()):
            result=scalar_expression(metadata,[values])[0]
        else:
            result=_SHExpression.apply(metadata,values)[0]
        return result.reshape(*x.shape[:-1],sum(2*l+1 for l in degrees))


@lru_cache(maxsize=128)
def _operator(degrees, normalize, normalization):
    return SphericalHarmonics(list(degrees), normalize, normalization)


def spherical_harmonics(l, x, normalize, normalization="integral"):
    """Evaluate FastEq SH in FP32/FP64 with first and second CUDA derivatives."""
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
        raise TypeError("FastEq spherical harmonics require GPU float32/float64")
    if isinstance(l, (str, o3.Irreps)):
        irreps = o3.Irreps(l)
        degrees = tuple(ir.l for mul, ir in irreps for _ in range(mul))
    elif isinstance(l, int):
        degrees = (l,)
    else:
        degrees = tuple(l)
    return _operator(degrees, bool(normalize), normalization)(x)


__all__ = ["SphericalHarmonics", "spherical_harmonics"]
