"""H100 FP32 shared Linear using FastEq's serial SO(3) kernels.

The native mul_ir addressing is fused into the upstream FastEq kernel. Only
the measured C64 domain is dispatched here. The kernels are installed into
the existing C++ Linear plan, preserving selective and recursive autograd.
Element-indexed weights, other dtypes and devices retain the original kernels.
"""

from functools import lru_cache

import torch


@lru_cache(maxsize=256)
def _layout(metadata):
    from .indexed_linear import _decode

    din, dout, numel, experts, storage, paths = _decode(metadata)
    if (
        experts != 1
        or storage != "mul_ir"
        or not 1 <= len(paths) <= 6
        or not any(p[5] > 1 for p in paths)
        or any(p[3] != 64 or p[4] not in (64, 192) or p[5] not in (1, 3, 5)
               for p in paths)
    ):
        return None

    # Preserve offsets and path order, including multiple inputs contributing
    # to one output. Reject gaps/overlap instead of silently changing storage.
    inputs = sorted({(sx, dim, ci) for sx, _, _, ci, _, dim, _ in paths})
    outputs = sorted({(sy, dim, co) for _, _, sy, _, co, dim, _ in paths})
    weights = sorted({(sw, ci, co) for _, sw, _, ci, co, _, _ in paths})
    for segments, total in ((inputs, din), (outputs, dout), (weights, numel)):
        stop = 0
        for offset, a, b in segments:
            if offset != stop:
                return None
            stop += a * b
        if stop != total:
            return None

    from fasteq.triton.so3_linear import LinearLayout

    return LinearLayout(
        tuple((a, b) for _, a, b in inputs),
        tuple((a, b) for _, a, b in outputs),
        tuple((a, b) for _, a, b in weights),
        tuple((weights.index((sw, ci, co)), inputs.index((sx, dim, ci)),
               outputs.index((sy, dim, co)))
              for sx, sw, sy, ci, co, dim, _ in paths),
        tuple(p[6] for p in paths),
    )


@lru_cache(maxsize=16)
def _is_hopper(device):
    return torch.cuda.get_device_capability(device) == (9, 0)


@lru_cache(maxsize=128)
def _serial_plan(metadata, device, nodes):
    from fasteq.cuda.triton_launch import prepare_triton
    from fasteq.triton import so3_linear
    from .indexed_linear import _plan

    layout = _layout(metadata)
    _, din, dout = layout.sizes
    # No tensor payload is cached or read during compilation. The C++ launcher
    # checks pointer alignment and retains original kernels for offset views.
    prototype = torch.empty(1, dtype=torch.float32, device=device)
    arguments = (prototype, prototype, prototype)
    common = dict(Z=nodes, SW=1, BM=32, BN=64, BK=32, PRECISION="ieee",
                  ANGULAR_FIRST=False, num_warps=4, num_stages=2)
    fwd_kw = dict(common, SX0=din, SX1=1, DY=dout, BZ=16)
    fwd_kw.update(zip(("GI", "GV", "GO", "GS", "GC", "PX", "PU", "PW", "PC"),
                      layout.flat_metadata))
    dx_kw = dict(Z=nodes, DIN=din, DOUT=dout, BZ=32, BM=32, BN=64, BK=32,
                 num_warps=4, num_stages=2)
    dx_kw.update(zip(("GI", "GU", "GXO", "GS", "GC", "PY", "PV", "PW", "PC"),
                     layout.backward_x_metadata))
    forward = prepare_triton(so3_linear._so3_linear_serial_cta_kernel,
                             arguments, prototype, fwd_kw, {})
    transpose = prepare_triton(so3_linear._so3_linear_packed_backward_x_kernel,
                               arguments, prototype, dx_kw, {})
    pack_kw = dict(OFFSETS=so3_linear._offsets(layout.weight_segments)[0],
                   U=tuple(u for u, _ in layout.weight_segments),
                   V=tuple(v for _, v in layout.weight_segments), BLOCK=256,
                   num_warps=4, num_stages=1)
    pack = prepare_triton(so3_linear._so3_linear_transpose_weights_kernel,
                          arguments[:2], prototype, pack_kw, {})
    if forward is None or transpose is None or pack is None:
        return None  # Unknown compiler ABI: retain the complete original plan.
    # A separate plan is essential: switching off this path must not mutate
    # cached baseline plans, including those used by compile/functorch fallbacks.
    plan = _plan.__wrapped__(metadata, torch.float32, device)
    max_v = max(v for _, v in layout.output_segments)
    plan.configure_serial(
        [forward[0], transpose[0], pack[0]],
        [[(nodes + 15) // 16, (max_v + 63) // 64, forward[1], forward[2]],
         [(nodes + 31) // 32, 1, transpose[1], transpose[2]],
         [(layout.sizes[0] + 255) // 256, 1, pack[1], pack[2]]],
        nodes,
    )
    return plan


def try_inference_linear(metadata, features, weight, node_type):
    """Return a fused, differentiable output or None for the legacy dispatcher.

    Forward and dX reuse main's serial tile, with live weight transposition
    before dX. dW reuses FastEq's existing CUDA reduction. Higher derivatives
    recurse through the same native plan and prune unused derivative branches.
    No weights are cached. CUDA graph capture requires a warmed specialization.
    """
    if (
        not features.is_cuda
        or features.dtype != torch.float32
        or features.ndim != 2
        or features.shape[0] < 16384
        or weight.ndim != 3
        or weight.shape[:2] != (1, 1)
        or weight.device != features.device
        or weight.dtype != features.dtype
        or node_type.device != features.device
        or node_type.dtype != torch.int64
        or node_type.shape != features.shape[:1]
        or any(s <= 0 for s in features.stride())
        or weight.stride(-1) <= 0
        or not _is_hopper(features.device)
    ):
        return None
    layout = _layout(metadata)
    if layout is None:
        return None
    dw, dx, _ = layout.sizes
    if weight.shape[-1] != dw or features.shape[1] != dx:
        return None

    from fasteq.triton import so3_linear

    if so3_linear.triton is None:
        return None  # Keep pure-CUDA installations usable without Triton.

    plan = _serial_plan(metadata, features.device, features.shape[0])
    return None if plan is None else plan.forward(features, weight, node_type)
