"""General FastEq CUDA Linear with indexed element and expert weights.

The legacy ``fast_equi_linear`` entry remains unchanged. This API accepts
arbitrary dense irrep paths and uses static tiled kernels, rather than the
legacy kernel's fixed angular output and MACE path-count specializations.
"""

import ast
import os
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import torch
from torch.utils._python_dispatch import is_in_torch_dispatch_mode


@lru_cache(maxsize=256)
def _decode(metadata):
    return ast.literal_eval(metadata)


def _groups(paths, role, experts):
    grouped = defaultdict(list)
    for i, (sx, sw, sy, ci, co, dim, _) in enumerate(paths):
        key = (
            (sx, ci, dim) if role == 0 else (sw, ci, co) if role == 1 else (sy, co, dim)
        )
        grouped[key].append(i)
    records, ids, tiles, intervals = [], [], [], []
    for group, ((offset, channels, dim), entries) in enumerate(grouped.items()):
        records.append((offset, channels, dim, len(ids), len(entries)))
        ids.extend(entries)
        width = channels * dim * (1 if role == 1 else experts)
        intervals.append((offset, offset + width))
        for expert in range(experts):
            if role == 1:
                tiles.extend(
                    (group, expert, start, 0) for start in range(0, channels * dim, 128)
                )
            else:
                tiles.extend(
                    (group, expert, channel, angle)
                    for channel in range(0, channels, 32)
                    for angle in range(0, dim, 4)
                )
    intervals.sort()
    stop = 0
    for first, last in intervals:
        if first < stop:
            raise ValueError("Distinct Linear groups must not overlap")
        if first != stop:
            return records, ids, tiles, False
        stop = last
    return records, ids, tiles, stop


@lru_cache(maxsize=256)
def _plan(metadata, dtype, device):
    from fasteq.cuda.jit import kernels, runtime

    din, dout, numel, experts, layout, paths = _decode(metadata)
    descriptions = [_groups(paths, role, experts) for role in range(3)]
    integers = torch.tensor(
        [p[:6] for p in paths], dtype=torch.int32, device=device
    ).reshape(-1, 6)
    coefficients = torch.tensor([p[6] for p in paths], dtype=dtype, device=device)
    group_tensors = [
        torch.tensor(g, dtype=torch.int32, device=device).reshape(-1, 5)
        for g, _, _, _ in descriptions
    ]
    ids = [
        torch.tensor(i, dtype=torch.int32, device=device) for _, i, _, _ in descriptions
    ]
    tiles = [
        torch.tensor(t, dtype=torch.int32, device=device).reshape(-1, 4)
        for _, _, t, _ in descriptions
    ]
    coverage = [
        description[3] == size
        for description, size in zip(descriptions, (din, numel, dout), strict=True)
    ]
    body = (Path(__file__).parents[1] / "cuda/src/indexed_linear.cu").read_text()
    scalar = "double" if dtype == torch.float64 else "float"
    sources = [
        f"#define FASTEQ_LINEAR_SCALAR {scalar}\n#define FASTEQ_LINEAR_ROLE {role}\n"
        + body
        for role in range(5)
    ]
    tensor_core = (
        dtype == torch.float64
        and bool(paths)
        and all(p[3] >= 64 and p[4] >= 64 for p in paths)
        and torch.cuda.get_device_capability(device) in ((8, 0), (9, 0))
    )
    simt_tiles = (
        dtype == torch.float32
        # Experimental until an exclusive-GPU paired run validates speed.
        # Set before the first plan creation; plans are process-local cached.
        and os.environ.get("FASTEQ_EXPERIMENTAL_LINEAR_SIMT") == "1"
        and bool(paths)
        and all(p[3] >= 64 and p[4] >= 64 for p in paths)
    )
    if tensor_core:
        # Reuse the existing FastEq FP64 MMA instruction definition verbatim.
        ptx = (
            Path(__file__).parents[1] / "cuda/src/equi_linear_kernels/ptx_inst.cuh"
        ).read_text()
        first = ptx.index("#define asm_mma_m16n8k4_f64_f64_f64_f64")
        macro = ptx[first : ptx.index("\n\n", first)]
        sources.extend(
            macro
            + f"\n#define FASTEQ_LINEAR_SCALAR double\n#define FASTEQ_LINEAR_ROLE {role}\n"
            + body
            for role in (5, 6, 7)
        )
    elif simt_tiles:
        sources.extend(
            f"#define FASTEQ_LINEAR_SCALAR float\n#define FASTEQ_LINEAR_ROLE {role}\n"
            + body
            for role in (8, 9, 10)
        )
    compiled = kernels(sources, device)
    plan = runtime().IndexedLinearPlan(
        [compiled[s] for s in sources],
        integers,
        coefficients,
        group_tensors,
        ids,
        tiles,
        din,
        dout,
        numel,
        experts,
        layout == "ir_mul",
        coverage,
    )
    if tensor_core:
        mma_tiles = [
            (group, expert, c, 0)
            for group, (_, co, _, _, _) in enumerate(descriptions[2][0])
            for expert in range(experts)
            for c in range(0, co, 8)
        ]
        plan.configure_tensor_core(
            torch.tensor(mma_tiles, dtype=torch.int32, device=device).reshape(-1, 4),
            max(p[5] for p in paths),
        )
        dx_tiles = [
            (group, expert, c, 0)
            for group, (_, ci, _, _, _) in enumerate(descriptions[0][0])
            for expert in range(experts)
            for c in range(0, ci, 8)
        ]
        dw_tiles = [
            (group, expert, ci, co)
            for group, (_, channels, outputs, _, _) in enumerate(descriptions[1][0])
            for expert in range(experts)
            for ci in range(0, channels, 64)
            for co in range(0, outputs, 8)
        ]
        plan.configure_derivative_tensor_core(
            torch.tensor(dx_tiles, dtype=torch.int32, device=device).reshape(-1, 4),
            torch.tensor(dw_tiles, dtype=torch.int32, device=device).reshape(-1, 4),
        )
    elif simt_tiles:
        grouped_tiles = [
            [(group, expert, c, 0)
             for group, (_, channels, _, _, _) in enumerate(descriptions[role][0])
             for expert in range(experts)
             for c in range(0, channels, 32)]
            for role in (0, 2)
        ]
        weight_tiles = [
            (group, expert, ci, co)
            for group, (_, channels, outputs, _, _) in enumerate(descriptions[1][0])
            for expert in range(experts)
            for ci in range(0, channels, 32)
            for co in range(0, outputs, 32)
        ]
        plan.configure_simt_tiles(
            *[torch.tensor(table, dtype=torch.int32, device=device).reshape(-1, 4)
              for table in (grouped_tiles[0], weight_tiles, grouped_tiles[1])],
            max(p[5] for p in paths),
        )
    return plan


@torch.library.custom_op("fasteq::indexed_linear", mutates_args=(), device_types="cuda")
def _contract(
    metadata: str,
    role: int,
    node_type: torch.Tensor,
    elements: int,
    x: torch.Tensor,
    weight: torch.Tensor,
    y: torch.Tensor,
) -> torch.Tensor:
    return _plan(metadata, x.dtype, x.device).contract(role, node_type, x, weight, y)


@_contract.register_fake
def _fake(metadata, role, node_type, elements, x, weight, y):
    din, dout, numel, experts, _, _ = _decode(metadata)
    return x.new_empty(
        (elements, experts, numel)
        if role == 1
        else (node_type.shape[0], din if role == 0 else dout)
    )


def _setup_context(ctx, inputs, output):
    ctx.kernel_metadata, ctx.role, types, ctx.elements, *operands = inputs
    ctx.save_for_backward(types, *operands)


def _backward(ctx, cotangent):
    types, *operands = ctx.saved_tensors
    operands[ctx.role] = cotangent
    gradients = [None, None, None]
    for role in range(3):
        if role != ctx.role and ctx.needs_input_grad[4 + role]:
            gradients[role] = _contract(
                ctx.kernel_metadata, role, types, ctx.elements, *operands
            )
    return None, None, None, None, *gradients


_contract.register_autograd(_backward, setup_context=_setup_context)


def indexed_linear(metadata, features, weight, node_type):
    """Apply a serialized plan to CUDA features and three-dimensional weights."""
    if features.dtype not in (torch.float32, torch.float64):
        raise TypeError("FastEq indexed Linear requires FP32 or FP64")
    if (
        torch.compiler.is_compiling()
        or is_in_torch_dispatch_mode()
        or torch._C._are_functorch_transforms_active()
    ):
        return _contract(
            metadata,
            2,
            node_type,
            weight.shape[0],
            features,
            weight,
            features.new_empty((0,)),
        )
    return _plan(metadata, features.dtype, features.device).forward(
        features, weight, node_type
    )


class IndexedLinear(torch.nn.Module):
    """An unbiased dense irrep map with external ``[elements, experts, W]`` weights.

    ``linear`` is an e3nn-compatible descriptor; its parameters are not copied.
    Both ``mul_ir`` and ``ir_mul`` layouts are supported. Input and output
    channels may vary between paths. Transposes differentiate recursively,
    including the parameter gradients of force losses.
    """

    def __init__(self, linear, num_experts=1, layout="mul_ir"):
        super().__init__()
        if num_experts < 1 or layout not in ("mul_ir", "ir_mul"):
            raise ValueError("Invalid expert count or feature layout")
        self.num_experts = num_experts
        self.input_dim = linear.irreps_in.dim * num_experts
        self.output_dim = linear.irreps_out.dim * num_experts
        self.weight_numel = linear.weight_numel
        self.irreps_in, self.irreps_out = linear.irreps_in, linear.irreps_out
        self.layout = layout
        paths, sw = [], 0
        for ins in linear.instructions:
            if ins.i_in < 0:
                raise ValueError("Pass scalar biases separately from IndexedLinear")
            ci = linear.irreps_in[ins.i_in].mul
            ir = linear.irreps_in[ins.i_in].ir
            co = linear.irreps_out[ins.i_out].mul
            paths.append(
                (
                    linear.irreps_in.slices()[ins.i_in].start * num_experts,
                    sw,
                    linear.irreps_out.slices()[ins.i_out].start * num_experts,
                    ci,
                    co,
                    ir.dim,
                    float(ins.path_weight),
                )
            )
            sw += ci * co
        self.paths = tuple(paths)
        self.metadata = repr(
            (
                self.input_dim,
                self.output_dim,
                self.weight_numel,
                num_experts,
                layout,
                self.paths,
            )
        )
        # CUDA metadata uses signed 32-bit fields; tensor row addressing is 64-bit.
        if any(value < 0 or value >= 2**31 for p in paths for value in p[:6]):
            raise ValueError("Path dimensions exceed the int32 metadata range")

    def forward(self, features, weight, node_type):
        if features.ndim != 2 or features.shape[1] != self.input_dim:
            raise ValueError(
                "Features must have shape (nodes, irreps_in.dim * experts)"
            )
        if node_type.dtype != torch.int64 or node_type.shape != features.shape[:1]:
            raise ValueError("Node types must be an int64 vector of length nodes")
        expected = (
            (self.weight_numel,)
            if weight.ndim == 2 and self.num_experts == 1
            else (self.num_experts, self.weight_numel)
        )
        if tuple(weight.shape[1:]) != expected:
            raise ValueError("Weights must have element, expert and path dimensions")
        if (
            not features.is_cuda
            or weight.device != features.device
            or node_type.device != features.device
            or weight.dtype != features.dtype
        ):
            raise ValueError(
                "Features, weights and indices must share a CUDA device; floating dtypes must match"
            )
        return indexed_linear(
            self.metadata,
            features,
            weight.reshape(weight.shape[0], self.num_experts, self.weight_numel),
            node_type,
        )


class MoEIndexedLinear(IndexedLinear):
    """Independent expert channels with the same external-weight interface."""

    def __init__(self, linear, num_experts, layout="mul_ir"):
        super().__init__(linear, num_experts, layout)
