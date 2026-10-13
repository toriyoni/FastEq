"""Reuse Triton CUDA binaries through FastEq's resident driver launcher.

Only pointer arguments and a row grid are supported. Unrecognized compiler
ABIs retain Triton's launcher. Graph capture requires warmup of every path.
"""

import re

import torch
from torch.utils._python_dispatch import is_in_torch_dispatch_mode

_RESIDENT_KERNELS = {}


def _pointers(values):
    result = []
    for value in values:
        if isinstance(value, tuple):
            result.extend(_pointers(value))
        else:
            result.append(value.data_ptr())
    return result


def prepare_triton(fn, arguments, x, kwargs, cache, *, pointers=None):
    """Prepare a resident executable without launching the operation.

    Native autograd plans use this only on a cold specialization. The tuple
    keeps the CUDA module alive and retains the same alignment/ABI checks as
    the Python launcher. Unsupported compiler ABIs return ``None``.
    """
    if torch.compiler.is_compiling() or is_in_torch_dispatch_mode():
        return None
    if pointers is None:
        pointers = _pointers(arguments)
    key = (
        fn,
        x.device,
        x.dtype,
        tuple(kwargs.items()),
        tuple(pointer % 16 == 0 for pointer in pointers),
    )
    if key not in cache:
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "Warm FastEq fused operations and derivatives before graph capture"
            )
        compiled = fn.warmup(*arguments, grid=(x.shape[0],), **kwargs)
        entry = re.search(
            r"\.entry\s+" + re.escape(compiled.name) + r"\s*\((.*?)\)",
            compiled.asm["ptx"],
            re.DOTALL,
        )
        meta = compiled.metadata
        safe = (
            entry is not None
            and entry.group(1).count(".param") == len(pointers) + 1
            and getattr(meta, "global_scratch_size", -1) == 0
            and getattr(meta, "num_ctas", 1) == 1
            and meta.shared <= 49152
        )
        if safe:
            from .jit import runtime

            resident_key = compiled.hash, x.device
            if resident_key not in _RESIDENT_KERNELS:
                _RESIDENT_KERNELS[resident_key] = runtime().Kernel(
                    compiled.asm["cubin"], compiled.name
                )
            cache[key] = (
                _RESIDENT_KERNELS[resident_key],
                meta.num_warps * 32,
                meta.shared,
            )
        else:
            cache[key] = None
    return cache[key]


def launch_triton(fn, arguments, x, kwargs, cache):
    if torch.compiler.is_compiling() or is_in_torch_dispatch_mode():
        fn[(x.shape[0],)](*arguments, **kwargs)
        return
    pointers = _pointers(arguments)
    executable = prepare_triton(fn, arguments, x, kwargs, cache, pointers=pointers)
    if executable is None:
        fn[(x.shape[0],)](*arguments, **kwargs)
    else:
        kernel, threads, shared = executable
        kernel.launch(
            [*pointers, 0],
            x.shape[0],
            1,
            threads,
            torch.cuda.current_stream(x.device).cuda_stream,
            shared,
        )
