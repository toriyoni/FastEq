"""Optional FastEq Triton operators, loaded only when requested."""

__all__ = ["EquivariantNormSpec", "NormResult", "TritonEquivariantNorm", "from_reference"]

def __getattr__(name):
    if name in __all__:
        from . import fused_equivariant_layer_norm
        value = getattr(fused_equivariant_layer_norm, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
