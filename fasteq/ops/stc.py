"""Compatibility imports for the STC implementation renamed by FastEq main."""
from .stc_uniform1d_jit import (
    FastSTCBackwardFunction,
    FastSymmetricTensorContractionUniform1dFunction,
    fast_stc_uniform1d_jit,
)

def __getattr__(name):
    from . import stc_uniform1d_jit
    return getattr(stc_uniform1d_jit, name)
