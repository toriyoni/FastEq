"""Cached e3nn-compatible entry point for FastEq's own Triton SH kernels."""

from functools import lru_cache

import torch
from e3nn import o3

from ..triton.spherical_harmonics import SphericalHarmonics


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
