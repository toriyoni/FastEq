"""FastEq CUDA sparse channel products and packed-feature split adjoints."""

from ast import literal_eval
from collections import defaultdict
from functools import lru_cache
import os

import torch


@lru_cache(maxsize=256)
def _parse(metadata):
    return literal_eval(metadata)


@lru_cache(maxsize=256)
def _source(metadata, dtype, role, required):
    width, dims, entries = _parse(metadata)
    groups = defaultdict(list)
    for indices, coefficient in entries:
        if indices[role] < 0 or any(
            index < 0 for i, index in enumerate(indices) if required & (1 << i)
        ):
            continue
        factors = [
            f"a{i}[(r{i}==1?0:e)*{dims[i] * width}+{index * width}+channel]"
            for i, index in enumerate(indices)
            if i != role and index >= 0
        ]
        groups[indices[role]].append(
            f"value+=T({coefficient:.17g})" + "".join(f"*{x}" for x in factors) + ";"
        )
    cases = "".join(
        f"case {slot}: {{ {''.join(terms)} break; }}"
        for slot, terms in sorted(groups.items())
    )
    pointers = ",".join(f"const T* a{i}" for i in range(len(dims)))
    counts = ",".join(f"long long r{i}" for i in range(len(dims)))
    return f"""
      using T={"double" if dtype == torch.float64 else "float"};
      extern "C" __global__ void run({pointers},T* result,{counts},long long rows,long long count) {{
        long long index=(long long)blockIdx.x*blockDim.x+threadIdx.x;
        if(index>=count*{dims[role] * width}) return;
        long long row=index/{dims[role] * width};
        int coordinate=index%{dims[role] * width}, segment=coordinate/{width},channel=coordinate%{width};
        T total=0;
        for(long long e=count==1?0:row;e<(count==1?rows:row+1);++e) {{
          T value=0; switch(segment) {{{cases}}} total+=value;
        }}
        result[index]=total;
      }}
    """


def _execute(metadata, role, required, rows, values):
    from fasteq.cuda.jit import kernels, runtime

    width, dimensions, _ = _parse(metadata)
    x = values[0]
    if not x.is_cuda or x.dtype not in (torch.float32, torch.float64):
        raise TypeError("Local products require CUDA FP32/FP64")
    if any(v.dtype != x.dtype or v.device != x.device for v in values):
        raise ValueError("Local product operands must share device and dtype")
    values = [v.contiguous() for v in values]
    counts = [v.shape[0] for v in values]
    count = rows if role == len(dimensions) - 1 else counts[role]
    result = x.new_empty(count, dimensions[role] * width)
    if not rows or not result.numel():
        return result.zero_()
    code = _source(metadata, x.dtype, role, required)
    kernel = kernels([code], x.device)[code]
    args = [v.data_ptr() for v in values] + [result.data_ptr(), *counts, rows, count]
    runtime().launch(
        [(kernel, args, (result.numel() + 127) // 128, 1, 128, 0)],
        torch.cuda.current_stream(x.device).cuda_stream,
    )
    return result


class _Product(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, role, required, rows, *values):
        ctx.kernel_metadata, ctx.role, ctx.required, ctx.rows = (
            metadata,
            role,
            required,
            rows,
        )
        ctx.save_for_backward(*values)
        return _execute(metadata, role, required, rows, values)

    @staticmethod
    def backward(ctx, gradient):
        values = list(ctx.saved_tensors)
        values[ctx.role] = gradient
        result = [
            _Product.apply(
                ctx.kernel_metadata,
                i,
                ctx.required | (1 << ctx.role),
                ctx.rows,
                *values,
            )
            if i != ctx.role and need
            else None
            for i, need in enumerate(ctx.needs_input_grad[4:])
        ]
        return None, None, None, None, *result


def local_product(metadata, *features):
    """Contract optional multilinear factors without an EQX runtime dependency."""
    width, dimensions, _ = _parse(metadata)
    values = [v.flatten(1) for v in features]
    counts = [v.shape[0] for v in values]
    rows = 0 if 0 in counts else max(counts)
    if any(count not in (1, rows) for count in counts):
        raise ValueError("Local product leading rows do not broadcast")
    values.append(features[0].new_empty(0, dimensions[-1] * width))
    return _Product.apply(metadata, len(dimensions) - 1, 0, rows, *values)


class LocalSplit(torch.autograd.Function):
    """Return packed views; run native sparse CUDA for their shared adjoint."""

    @classmethod
    def apply(cls, features, ranges, metadata):
        if (
            os.environ.get("FASTEQ_EXPERIMENTAL_LOCAL_SPLIT_VIEW") == "1"
            and not torch.is_grad_enabled()
        ):
            return tuple(features.narrow(1, start, size) for start, size in ranges)
        return super().apply(features, ranges, metadata)

    @staticmethod
    def forward(ctx, features, ranges, metadata):
        ctx.kernel_metadata, ctx.shape = metadata, features.shape
        ctx.set_materialize_grads(False)
        return tuple(features.narrow(1, start, size) for start, size in ranges)

    @staticmethod
    def backward(ctx, *gradients):
        reference = next((g for g in gradients if g is not None), None)
        if reference is None:
            return None, None, None
        width, dimensions, _ = _parse(ctx.kernel_metadata)
        values = [reference.new_empty(reference.shape[0], 0)] + [
            g
            if g is not None
            else reference.new_empty(reference.shape[0], dimensions[i + 1] * width)
            for i, g in enumerate(gradients)
        ]
        if all(g is not None for g in gradients):
            return (
                _Product.apply(ctx.kernel_metadata, 0, 0, reference.shape[0], *values),
                None,
                None,
            )
        # Each split path contains precisely one output factor. Summing the
        # requested transposes preserves omitted outputs and arbitrary orders.
        result = None
        for i, g in enumerate(gradients, 1):
            if g is not None:
                value = _Product.apply(
                    ctx.kernel_metadata, 0, 1 << i, reference.shape[0], *values
                )
                result = value if result is None else result + value
        return result, None, None
