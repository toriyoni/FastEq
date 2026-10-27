"""FastEq scalar-irrep bias map, its ordered transpose, and native addition.

The bias node receives the original parameter, not an expanded tensor. Its
transpose groups target rows in logical order and extends NativeNorm's exact
256-row CUDA tree with a mapped/strided reader. Broadcast and transpose are
recursive autograd transposes, including higher force-loss derivatives.
"""

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import torch

from .native_equivariant_norm import reduce_rows, source_for
from .equivariant_norm import _requested_gradients


def _validate(*values):
    first = values[0]
    if not first.is_cuda or first.dtype not in (torch.float32, torch.float64):
        raise TypeError("Native elementwise operators require CUDA FP32/FP64")
    if any(v.device != first.device or v.dtype != first.dtype for v in values):
        raise TypeError("Native elementwise operands must share device and dtype")


def _divisors(shape):
    return tuple(math.prod(shape[i + 1:]) for i in range(len(shape)))


def _offset(shape, strides, index="index"):
    terms = [f"(({index}/{div}LL)%{size}LL)*{stride}LL"
             for size, div, stride in zip(shape, _divisors(shape), strides)
             if size > 1 and stride]
    return "+".join(terms) or "0LL"


def _map_function(mapping):
    """Compress ordinary scalar segments rather than divergent table loads."""
    lines = ["__device__ __forceinline__ int fasteq_bias_coordinate(int c) {"]
    start = 0
    while start < len(mapping):
        if mapping[start] < 0:
            start += 1
            continue
        stop = start + 1
        while stop < len(mapping) and mapping[stop] == mapping[start] + stop - start:
            stop += 1
        lines.append(f"if(c>={start} && c<{stop}) return c+({mapping[start]-start});")
        start = stop
    lines += ["return -1;", "}"]
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class _BiasPlan:
    target_shape: tuple
    bias_shape: tuple
    mapping: tuple
    shared: bool

    @property
    def rows(self):
        return math.prod(self.target_shape)

    @property
    def channels(self):
        return self.bias_shape[-1]

    @property
    def bias_rows(self):
        return math.prod(self.bias_shape[:-1])

    @property
    def leading(self):
        if self.shared:
            return (1,) * len(self.target_shape)
        return (1,) * (len(self.target_shape) - len(self.bias_shape) + 1) + self.bias_shape[:-1]

    @property
    def repeats(self):
        return math.prod(s for s, b in zip(self.target_shape, self.leading) if b == 1)

    def row_offset(self, strides):
        if self.shared:
            return "0LL"
        padded = (0,) * (len(self.target_shape) - len(self.bias_shape) + 1) + strides[:-1]
        terms = [f"((row/{div}LL)%{size}LL)*{stride}LL"
                 for size, b, div, stride in zip(self.target_shape, self.leading,
                                                _divisors(self.target_shape), padded)
                 if b > 1 and stride]
        return "+".join(terms) or "0LL"

    def target_row(self):
        # Bias row and repetition are each decoded in original target-axis
        # order. This groups precisely the rows sharing one live parameter.
        matched = tuple(s for s, b in zip(self.target_shape, self.leading) if b > 1)
        repeated = tuple(s for s, b in zip(self.target_shape, self.leading) if b == 1)
        matched_divs, repeated_divs = iter(_divisors(matched)), iter(_divisors(repeated))
        terms = []
        for size, b, div in zip(self.target_shape, self.leading, _divisors(self.target_shape)):
            inner_div = next(matched_divs) if b > 1 else next(repeated_divs)
            if size > 1:
                index = "biasrow" if b > 1 else "repeat"
                terms.append(f"(({index}/{inner_div}LL)%{size}LL)*{div}LL")
        return "+".join(terms) or "0LL"


@lru_cache(maxsize=256)
def _plan(target_shape, bias_shape, mapping, shared):
    plan = _BiasPlan(target_shape, bias_shape, mapping, shared)
    if not bias_shape or any(i < -1 or i >= plan.channels for i in mapping):
        raise ValueError("Invalid scalar-irrep bias map")
    if shared:
        if plan.bias_rows != 1:
            raise ValueError("Shared bias must contain one parameter row")
    elif len(bias_shape) - 1 > len(target_shape) or any(
            b != 1 and b != s for s, b in zip(target_shape, plan.leading)):
        raise ValueError("Bias leading shape does not broadcast to the output")
    return plan


def _reader(scalar, name, expression):
    return (f"__device__ __forceinline__ {scalar} {name}(const {scalar}* input, long long index) {{"
            f"return input[{expression}];}}\n")


@lru_cache(maxsize=512)
def _element_source(role, shape, strides_a, strides_b, plan, dtype):
    scalar = "double" if dtype == torch.float64 else "float"
    entry = "fasteq_element_" + role
    prefix = (f"#define FASTEQ_ELEMENT_TYPE {scalar}\n"
              f"#define FASTEQ_ELEMENT_FP64 {int(dtype == torch.float64)}\n"
              f"#define FASTEQ_ELEMENT_ENTRY {entry}\n"
              f"#define FASTEQ_ELEMENT_HAS_LEFT {int(role in ('add', 'bias_add', 'gather'))}\n"
              f"#define FASTEQ_ELEMENT_HAS_RIGHT {int(role in ('add', 'bias_add', 'broadcast'))}\n")
    contiguous = tuple(_divisors(shape))
    left_contiguous = role in ('add', 'bias_add') and strides_a == contiguous
    right_contiguous = role == 'add' and strides_b == contiguous
    prefix += (f"#define FASTEQ_ELEMENT_LEFT_CONTIGUOUS {int(left_contiguous)}\n"
               f"#define FASTEQ_ELEMENT_RIGHT_CONTIGUOUS {int(right_contiguous)}\n")
    functions = ""
    if role in ('add', 'bias_add'):
        functions += _reader(scalar, 'fasteq_element_left', _offset(shape, strides_a))
    if role == 'add':
        functions += _reader(scalar, 'fasteq_element_right', _offset(shape, strides_b))
    if role in ('bias_add', 'broadcast'):
        functions += _map_function(plan.mapping)
        functions += (f"__device__ __forceinline__ {scalar} fasteq_element_right("
                      f"const {scalar}* input, long long index) {{"
                      f"const long long row=index/{len(plan.mapping)}LL;"
                      f"const int c=fasteq_bias_coordinate(index%{len(plan.mapping)}LL);"
                      f"return c<0?{scalar}(0):input[({plan.row_offset(strides_b)})+c*{strides_b[-1]}LL];}}\n")
    if role == 'gather':
        functions += _transpose_reader(plan, strides_a, dtype)
        functions += (f"__device__ __forceinline__ {scalar} fasteq_element_left("
                      f"const {scalar}* input,long long index) {{"
                      "return fasteq_bias_grad_read(input,0,index);}\n")
    header = (Path(__file__).parents[1] / 'cuda/src/native_elementwise.cuh').read_text()
    return prefix + functions + header, entry


def _transpose_reader(plan, strides, dtype):
    scalar = "double" if dtype == torch.float64 else "float"
    add = "__dadd_rn" if dtype == torch.float64 else "__fadd_rn"
    inverse = [[] for _ in range(plan.channels)]
    for output, bias in enumerate(plan.mapping):
        if bias >= 0:
            inverse[bias].append(output)
    lines = [f"__device__ __forceinline__ {scalar} fasteq_bias_grad_read("
             f"const {scalar}* input,long long repeat,long long column) {{",
             f"const long long biasrow=column/{plan.channels}LL;",
             f"const int bias=column%{plan.channels}LL;",
             f"const long long row={plan.target_row()};",
             f"{scalar} value={scalar}(0);"]
    # Consecutive one-to-one maps become one branch plus one coalesced load.
    b = 0
    while b < plan.channels:
        locations = inverse[b]
        if len(locations) == 1:
            stop = b + 1
            while stop < plan.channels and inverse[stop] == [locations[0] + stop - b]:
                stop += 1
            lines.append(f"if(bias>={b} && bias<{stop}) value="
                         f"input[row*{strides[0]}LL+(bias+({locations[0]-b}))*{strides[1]}LL];")
            b = stop
        else:
            if locations:
                lines.append(f"if(bias=={b}) {{")
                for coordinate in locations:
                    lines.append(f"value={add}(value,input[row*{strides[0]}LL+{coordinate}LL*{strides[1]}LL]);")
                lines.append("}")
            b += 1
    lines += ["return value;", "}"]
    return "\n".join(lines) + "\n"


@lru_cache(maxsize=256)
def _transpose_source(plan, strides, dtype):
    # Only the reader and flattened CTA indexing extend the existing Norm
    # kernel. All arithmetic, synchronization, padding, and tree order remain.
    source = source_for(3, {}, dtype)
    old = "const NormIndex row_base = NormIndex(blockIdx.x) * 256;\n    const int column_base = blockIdx.y * 32;"
    new = ("const NormIndex column_tiles = (width + 31LL) / 32;\n"
           "    const NormIndex group = NormIndex(blockIdx.x) / column_tiles;\n"
           "    const NormIndex row_base = group * 256;\n"
           "    const int column_base = (NormIndex(blockIdx.x) % column_tiles) * 32;")
    assert source.count(old) == 1
    source = source.replace(old, new)
    for expression, replacement in [
        ("input[(row_base + r) * width + column]", "fasteq_bias_grad_read(input,row_base+r,column)"),
        ("input[(row_base + r + 128) * width + column]", "fasteq_bias_grad_read(input,row_base+r+128,column)"),
        ("output[NormIndex(blockIdx.x) * width + column_base + threadIdx.x]",
         "output[group * width + column_base + threadIdx.x]")]:
        assert source.count(expression) == 1
        source = source.replace(expression, replacement)
    source = source.replace("#define FASTEQ_NORM_ENTRY fasteq_norm_reduce_rows",
                            "#define FASTEQ_NORM_ENTRY fasteq_bias_reduce_rows")
    return _transpose_reader(plan, strides, dtype) + source


def _element(role, first, second, shape, plan=None):
    from fasteq.cuda.jit import kernels
    output = first.new_empty(shape)
    if output.numel() == 0:
        return output
    source, entry = _element_source(role, tuple(shape), tuple(first.stride()),
                                    tuple(second.stride()) if second is not None else (),
                                    plan, first.dtype)
    kernel = kernels([source], first.device, entry=entry)[source]
    items = 2 if first.dtype == torch.float64 else 4
    packets = (output.numel() + items - 1) // items
    kernel.launch([first.data_ptr(), second.data_ptr() if second is not None else 0,
                   output.data_ptr(), output.numel()], min((packets + 127) // 128, 65535),
                  1, 128, torch.cuda.current_stream(first.device).cuda_stream)
    return output


def _transpose(plan, gradient):
    if plan.bias_rows * plan.channels == 0:
        return gradient.new_empty(plan.bias_shape)
    if plan.repeats == 0:
        return _element('fill', gradient, None, plan.bias_shape)
    if plan.repeats == 1:
        return _element('gather', gradient, None, plan.bias_shape, plan)
    from fasteq.cuda.jit import kernels
    width = plan.bias_rows * plan.channels
    groups = (plan.repeats + 255) // 256
    partials = gradient.new_empty((groups, width))
    source = _transpose_source(plan, tuple(gradient.stride()), gradient.dtype)
    kernel = kernels([source], gradient.device, entry='fasteq_bias_reduce_rows')[source]
    grid = groups * ((width + 31) // 32)
    if grid > 2147483647:
        raise ValueError("Native bias transpose grid exceeds the CUDA x-dimension")
    kernel.launch([gradient.data_ptr(), partials.data_ptr(), plan.repeats, width],
                  grid, 1, 128, torch.cuda.current_stream(gradient.device).cuda_stream)
    if width <= 65535 * 32:
        return reduce_rows(partials, width).reshape(plan.bias_shape)
    # Large per-row parameter domains need column chunks only for Norm's
    # unmodified second-stage y-grid. These are copies/views, not library math.
    reduced = gradient.new_empty((width,))
    for start in range(0, width, 65535 * 32):
        stop = min(width, start + 65535 * 32)
        reduced[start:stop].copy_(reduce_rows(partials[:, start:stop].contiguous(), stop-start))
    return reduced.reshape(plan.bias_shape)


class _BiasTranspose(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, gradient):
        ctx.plan = plan
        ctx.set_materialize_grads(False)
        return _transpose(plan, gradient)

    @staticmethod
    def backward(ctx, seed):
        return None, None if seed is None else _BiasBroadcast.apply(ctx.plan, seed)


class _BiasBroadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, bias):
        ctx.plan = plan
        ctx.set_materialize_grads(False)
        return _element('broadcast', bias, bias, (plan.rows, len(plan.mapping)), plan)

    @staticmethod
    def backward(ctx, gradient):
        return None, None if gradient is None else _BiasTranspose.apply(ctx.plan, gradient)


class _BiasAdd(torch.autograd.Function):
    @staticmethod
    def forward(ctx, plan, output, bias):
        ctx.plan = plan
        ctx.set_materialize_grads(False)
        return _element('bias_add', output, bias, tuple(output.shape), plan)

    @staticmethod
    def backward(ctx, gradient):
        if gradient is None:
            return None, None, None
        # The descriptor is not a Tensor edge. Reuse Norm's live graph-task
        # selection to avoid computing a bias transpose during coordinate AD.
        requested = _requested_gradients(ctx.needs_input_grad[1:])
        return (None, gradient if requested[0] else None,
                _BiasTranspose.apply(ctx.plan, gradient) if requested[1] else None)


class _Add(torch.autograd.Function):
    @staticmethod
    def forward(ctx, a, b):
        ctx.set_materialize_grads(False)
        return _element('add', a, b, tuple(a.shape))

    @staticmethod
    def backward(ctx, gradient):
        return (gradient if ctx.needs_input_grad[0] else None,
                gradient if ctx.needs_input_grad[1] else None)


def native_bias_add(output, bias, mapping, target_shape, *, shared=True):
    """Add only scalar-irrep coordinates, preserving the live bias parameter."""
    _validate(output, bias)
    plan = _plan(tuple(target_shape), tuple(bias.shape), tuple(mapping), bool(shared))
    if tuple(output.shape) != (plan.rows, len(mapping)):
        raise ValueError("Bias output must be the flattened target leading shape")
    return _BiasAdd.apply(plan, output, bias)


def native_add(a, b):
    """Same-shape native CUDA addition with identity recursive derivatives."""
    _validate(a, b)
    if a.shape != b.shape:
        raise ValueError("Native addition requires equal logical shapes")
    return _Add.apply(a, b)


__all__ = ['native_bias_add', 'native_add']
