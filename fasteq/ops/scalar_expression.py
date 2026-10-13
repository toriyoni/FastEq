"""FastEq scalar CUDA expressions with differentiable analytic transposes.

The CUDA emitter reuses FastEq's spherical-harmonic adapter implementation.
This module supplies its own small expression builder and derivative rules so
geometry operators do not require an external execution/codegen backend.
"""

import ast
import math
import os
from functools import lru_cache

import torch

from fasteq.cuda.jit import kernels


class ScalarProgram:
    def __init__(self, nodes=()):
        self.nodes = [tuple(node) for node in nodes]
        self._ids = {node: i for i, node in enumerate(self.nodes)}

    def constant(self, value):
        return self.add("constant", float(value))

    def add(self, op, *args):
        if op == "constant":
            args = (float(args[0]),)
        elif op == "input":
            args = tuple(int(a) for a in args)
        elif op == "neg":
            if self.nodes[args[0]][0] == "constant":
                return self.constant(-self.nodes[args[0]][1])
            if self.nodes[args[0]][0] == "neg":
                return self.nodes[args[0]][1]
        elif op in ("add", "mul", "div"):
            a, b = (self.nodes[i] for i in args)
            if a[0] == b[0] == "constant":
                if op == "add":
                    return self.constant(a[1] + b[1])
                if op == "mul":
                    return self.constant(a[1] * b[1])
                if b[1] != 0:
                    return self.constant(a[1] / b[1])
            if op == "add":
                if a == ("constant", 0.0):
                    return args[1]
                if b == ("constant", 0.0):
                    return args[0]
            elif op == "mul":
                if a == ("constant", 0.0) or b == ("constant", 0.0):
                    return self.constant(0)
                if a == ("constant", 1.0):
                    return args[1]
                if b == ("constant", 1.0):
                    return args[0]
            elif b == ("constant", 1.0):
                return args[0]
        elif op == "select" and args[1] == args[2]:
            return args[1]
        node = (op, *args)
        if node not in self._ids:
            self._ids[node] = len(self.nodes)
            self.nodes.append(node)
        return self._ids[node]

    def transpose(self, outputs, widths, active):
        """Seed output vectors and return adjoints for selected input vectors."""
        count = len(self.nodes)
        adjoints = {}

        def accumulate(node, seed):
            adjoints[node] = (
                self.add("add", adjoints[node], seed) if node in adjoints else seed
            )

        for slot, group in enumerate(outputs, len(widths)):
            for j, node in enumerate(group):
                accumulate(node, self.add("input", slot, j))
        half, zero = self.constant(0.5), self.constant(0)
        for node in reversed(range(count)):
            if node not in adjoints:
                continue
            seed = adjoints[node]
            op, *args = self.nodes[node]
            if op in ("constant", "input", "lt", "le"):
                continue
            if op == "add":
                for operand in args:
                    accumulate(operand, seed)
            elif op == "mul":
                a, b = args
                accumulate(a, self.add("mul", seed, b))
                accumulate(b, self.add("mul", seed, a))
            elif op == "div":
                a, b = args
                accumulate(a, self.add("div", seed, b))
                accumulate(
                    b, self.add("neg", self.add("div", self.add("mul", seed, node), b))
                )
            elif op == "neg":
                accumulate(args[0], self.add("neg", seed))
            elif op == "sqrt":
                accumulate(args[0], self.add("div", self.add("mul", seed, half), node))
            elif op == "exp":
                accumulate(args[0], self.add("mul", seed, node))
            elif op == "sin":
                accumulate(args[0], self.add("mul", seed, self.add("cos", args[0])))
            elif op == "cos":
                accumulate(
                    args[0],
                    self.add("neg", self.add("mul", seed, self.add("sin", args[0]))),
                )
            elif op == "select":
                condition, yes, no = args
                accumulate(yes, self.add("select", condition, seed, zero))
                accumulate(no, self.add("select", condition, zero, seed))
            else:
                raise NotImplementedError(f"No FastEq scalar derivative for {op}")
        gradients = []
        for slot in active:
            gradients.append(
                tuple(
                    adjoints.get(self.add("input", slot, j), zero)
                    for j in range(widths[slot])
                )
            )
        return tuple(gradients), tuple(widths) + tuple(len(group) for group in outputs)


@lru_cache(maxsize=256)
def _parse(metadata):
    return ast.literal_eval(metadata)


@lru_cache(maxsize=256)
def _store_tile_configuration(metadata, dtype):
    _, groups, _ = _parse(metadata)
    widths = tuple(map(len, groups))
    if max(widths, default=0) < 8:
        return None
    total = sum(widths)
    pitch = total + (total % 2 == 0)
    block = 128
    size = 8 if dtype == "double" else 4
    while block >= 32 and block * pitch * size > 32768:
        block //= 2
    return (block, pitch) if block >= 32 else None


@lru_cache(maxsize=256)
def _source(metadata, dtype, strategy="direct", block=128):
    nodes, outputs, widths = _parse(metadata)
    needed = {node for group in outputs for node in group}
    for node in reversed(range(len(nodes))):
        if node in needed and nodes[node][0] not in ("input", "constant"):
            needed.update(nodes[node][1:])
    args = [f"const T* x{i}" for i in range(len(widths))]
    args += [f"T* y{i}" for i in range(len(outputs))] + ["long long n"]
    code = [
        f"using T={dtype};",
        'extern "C" __global__ void run(' + ",".join(args) + "){",
    ]
    if strategy == "tile":
        choice = _store_tile_configuration(metadata, dtype)
        if choice is None or choice[0] != block:
            raise ValueError("Scalar store tile does not match the shared-memory budget")
        _, pitch = choice
        code.extend((f"__shared__ T tile[{block}][{pitch}];",
                     "long long base=(long long)blockIdx.x*blockDim.x;",
                     "long long r=base+threadIdx.x; if(r<n){"))
    elif strategy == "direct" and block == 128:
        code.append("long long r=(long long)blockIdx.x*blockDim.x+threadIdx.x; if(r>=n)return;")
    else:
        raise ValueError("Unknown ScalarExpression store strategy or block size")
    for node, (op, *data) in enumerate(nodes):
        if node not in needed:
            continue
        values = [f"v{i}" for i in data]
        if op == "input":
            expression = f"x{data[0]}[r*{widths[data[0]]}+{data[1]}]"
        elif op == "constant":
            if not math.isfinite(data[0]):
                raise ValueError("FastEq scalar constants must be finite")
            expression = f"T({data[0]:.17g})"
        elif op == "select":
            expression = f"{values[0]} ? {values[1]} : {values[2]}"
        elif op in ("sqrt", "exp", "sin", "cos"):
            expression = f"{op}({values[0]})"
        elif op == "neg":
            expression = f"-{values[0]}"
        else:
            symbol = {"add": "+", "mul": "*", "div": "/", "lt": "<", "le": "<="}[op]
            expression = f"{values[0]} {symbol} {values[1]}"
        code.append(f"T v{node}={expression};")
    offset = 0
    for slot, group in enumerate(outputs):
        code.extend(
            (f"tile[threadIdx.x][{offset+j}]=v{node};" if strategy == "tile"
             else f"y{slot}[r*{len(group)}+{j}]=v{node};")
            for j, node in enumerate(group)
        )
        offset += len(group)
    if strategy == "tile":
        # Every thread reaches the barrier, including invalid tail rows.
        # Each output group keeps its own global stride and shared prefix.
        code.extend(("}", "__syncthreads();"))
        offset = 0
        for slot, group in enumerate(outputs):
            width = len(group)
            if width:
                code.append(
                    f"for(int i=threadIdx.x;i<{block*width};i+=blockDim.x){{"
                    f"int row=i/{width},column=i%{width};"
                    f"if(base+row<n)y{slot}[(base+row)*{width}+column]="
                    f"tile[row][{offset}+column];}}"
                )
            offset += width
    return "\n".join([*code, "}"])


@torch.library.custom_op(
    "fasteq::scalar_expression", mutates_args=(), device_types="cuda"
)
def scalar_expression(metadata: str, inputs: list[torch.Tensor]) -> list[torch.Tensor]:
    _, outputs, widths = _parse(metadata)
    if (
        not inputs
        or not inputs[0].is_cuda
        or inputs[0].dtype not in (torch.float32, torch.float64)
    ):
        raise TypeError("FastEq scalar CUDA supports FP32/FP64")
    n = inputs[0].shape[0]
    for value, width in zip(inputs, widths, strict=True):
        if (
            value.shape != (n, width)
            or value.device != inputs[0].device
            or value.dtype != inputs[0].dtype
        ):
            raise ValueError("Scalar input shapes/dtypes/devices do not match metadata")
    values = [value.contiguous() for value in inputs]
    results = [values[0].new_empty(n, len(group)) for group in outputs]
    if n and results:
        dtype = "double" if values[0].dtype == torch.float64 else "float"
        choice = (_store_tile_configuration(metadata, dtype)
                  if n >= 16384 and os.environ.get("FASTEQ_EXPERIMENTAL_WIGNER_STORE_TILE") == "1"
                  else None)
        strategy, block = ("tile", choice[0]) if choice else ("direct", 128)
        source = _source(metadata, dtype, strategy, block)
        kernel = kernels([source], values[0].device)[source]
        kernel.launch(
            [value.data_ptr() for value in (*values, *results)] + [n],
            (n + block - 1) // block,
            1,
            block,
            torch.cuda.current_stream(values[0].device).cuda_stream,
        )
    return results


@scalar_expression.register_fake
def _fake(metadata, inputs):
    return [
        inputs[0].new_empty(inputs[0].shape[0], len(group))
        for group in _parse(metadata)[1]
    ]


def _setup(ctx, inputs, output):
    ctx.kernel_metadata, values = inputs
    ctx.output_widths = tuple(value.shape[1] for value in output)
    ctx.save_for_backward(*values)
    ctx.set_materialize_grads(False)


@lru_cache(maxsize=256)
def _adjoint(metadata, active):
    nodes, outputs, widths = _parse(metadata)
    program = ScalarProgram(nodes)
    for slot, width in enumerate(widths):
        for j in range(width):
            program.add("input", slot, j)
    result, sizes = program.transpose(outputs, widths, active)
    return repr((tuple(program.nodes), result, sizes))


def _backward(ctx, gradients):
    values = list(ctx.saved_tensors)
    active = tuple(i for i, required in enumerate(ctx.needs_input_grad[1]) if required)
    result = [None] * len(values)
    if active:
        seeds = [
            grad if grad is not None else values[0].new_zeros(values[0].shape[0], width)
            for grad, width in zip(gradients, ctx.output_widths, strict=True)
        ]
        adjoints = scalar_expression(
            _adjoint(ctx.kernel_metadata, active), [*values, *seeds]
        )
        for i, grad in zip(active, adjoints, strict=True):
            result[i] = grad
    return None, result


scalar_expression.register_autograd(_backward, setup_context=_setup)
