"""FastEq CUDA frame rotations and rotary contractions.

Frame/rotary arithmetic extends FastEq's FCTP shared transpose, vector loads
and native warp reductions. Wigner's API and mathematical reference follows
EQX (Zemin Xu), CC-BY-4.0. FastEq independently implements the integer Sym^n
recurrence and owned scalar CUDA emitter/AD; no EQX CUDA implementation is
executed or copied into these kernels.
"""

import math
from pathlib import Path
from numbers import Integral
from ast import literal_eval
from collections import defaultdict
from functools import cache, lru_cache
from itertools import pairwise

import torch


@lru_cache(maxsize=1)
def _fasteq_rotation_primitives():
    """Read FastEq's original warp sum and shared FCTP vector loader."""
    directory = Path(__file__).parents[1] / "cuda/src"
    utilities = (directory / "cuda_utils.hpp").read_text()
    def extract(signature):
        start = utilities.index(signature)
        return utilities[start:utilities.index("\ntemplate <typename T>", start + 5)]
    shuffle = extract("template <typename T>\n__device__ __forceinline__ T gpu_shfl_down")
    reduction = extract("template <typename T>\n__device__ __forceinline__ T warp_reduce_sum(T v)")
    return ("#define GPU_BACKEND_HIP 0\n#define FULL_MASK 0xffffffffu\n"
            + shuffle + reduction
            + (directory / "fctp_shared.cuh").read_text())


def _rotation_table(name, values):
    payload = ",".join(map(str, values)) if values else "0"
    return f"__device__ __constant__ int {name}[]={{{payload}}};\n"


def _frame_configuration(dtype, metadata, role):
    degrees_in, degrees_out = literal_eval(metadata)
    ni, no = len(degrees_in), len(degrees_out)
    pitch_in, pitch_out = ni + (ni % 2 == 0), no + (no % 2 == 0)
    shared_columns = pitch_in + pitch_out if role == 1 else (
        pitch_in if role == 2 else pitch_out)
    tile = 128
    while tile > 4 and tile * shared_columns * (8 if dtype == torch.float64 else 4) > 32768:
        tile //= 2
    if tile * shared_columns * (8 if dtype == torch.float64 else 4) > 32768:
        raise ValueError("Frame angular metadata exceeds the bounded FastEq shared tile")
    return degrees_in, degrees_out, tile


@lru_cache(maxsize=128)
def _frame_source(dtype, channels, metadata, role, strategy="scalar"):
    """Specialize shared FastEq CUDA using degree tables, without external templates."""
    if role not in (0, 1, 2) or strategy not in ("scalar", "vector"):
        raise ValueError("Unknown frame role or strategy")
    degrees_in, degrees_out, tile = _frame_configuration(dtype, metadata, role)
    pairs = [(output, source) for output, a in enumerate(degrees_out)
             for source, b in enumerate(degrees_in) if a == b]
    defines = (
        f"#define FASTEQ_FRAME_ROLE {role}\n"
        f"#define FASTEQ_ROTATION_TYPE {'double' if dtype == torch.float64 else 'float'}\n"
        f"#define FASTEQ_ROTATION_CHANNELS {channels}\n"
        f"#define FASTEQ_ROTATION_NI {len(degrees_in)}\n"
        f"#define FASTEQ_ROTATION_NO {len(degrees_out)}\n"
        f"#define FASTEQ_ROTATION_PAIRS {len(pairs)}\n"
        f"#define FASTEQ_FRAME_TILE_CHANNELS {tile}\n"
        f"#define FASTEQ_ROTATION_ENTRY {_frame_entry(role)}\n"
    )
    tables = (_rotation_table("fasteq_frame_degree_in", degrees_in)
              + _rotation_table("fasteq_frame_degree_out", degrees_out)
              + _rotation_table("fasteq_frame_pair_out", [p[0] for p in pairs])
              + _rotation_table("fasteq_frame_pair_in", [p[1] for p in pairs]))
    body = (Path(__file__).parents[1] / "cuda/src/fctp_rotation.cuh").read_text()
    source = _fasteq_rotation_primitives() + defines + tables + body
    groups = (1 if pairs else 0) if role == 1 else (channels + tile - 1) // tile
    return source, groups


def _frame_entry(role):
    return ("fasteq_frame_dx", "fasteq_frame_dmatrix", "fasteq_frame_forward")[role]


@lru_cache(maxsize=128)
def _rotary_layout(lmax, mmax):
    if not 0 <= mmax <= lmax:
        raise ValueError("Rotary orders require 0 <= mmax <= lmax")
    order, real, imaginary = [], [], []
    cursor = 0
    for absolute_order in range(mmax + 1):
        width = lmax - absolute_order + 1
        for degree in range(width):
            order.append(absolute_order)
            real.append(cursor + degree)
            imaginary.append(cursor + width + degree if absolute_order else -1)
        cursor += width * (2 if absolute_order else 1)
    return tuple(order), tuple(real), tuple(imaginary), cursor


@lru_cache(maxsize=128)
def _rotary_source(dtype, lmax, mmax, channels, heads, role):
    """Pair real/imaginary channels in FastEq's native contraction/reduction family."""
    if role not in (0, 1, 2, 3) or heads <= 0 or channels % heads:
        raise ValueError("Invalid rotary role or head/channel count")
    order, real, imaginary, angular = _rotary_layout(lmax, mmax)
    defines = (
        f"#define FASTEQ_ROTARY_ROLE {role}\n"
        f"#define FASTEQ_ROTATION_TYPE {'double' if dtype == torch.float64 else 'float'}\n"
        f"#define FASTEQ_ROTATION_CHANNELS {channels}\n"
        f"#define FASTEQ_ROTATION_HEADS {heads}\n"
        f"#define FASTEQ_ROTATION_ANGULAR {angular}\n"
        f"#define FASTEQ_ROTATION_MMAX {mmax}\n"
        f"#define FASTEQ_ROTATION_PAIRS {len(order)}\n"
        f"#define FASTEQ_ROTATION_ENTRY {_rotary_entry(role)}\n"
    )
    tables = (_rotation_table("fasteq_rotary_order", order)
              + _rotation_table("fasteq_rotary_real", real)
              + _rotation_table("fasteq_rotary_imag", imaginary))
    body = (Path(__file__).parents[1] / "cuda/src/fctp_rotation.cuh").read_text()
    packed = (lmax, mmax, channels, heads) == (2, 1, 64, 4)
    prefix = "#define FASTEQ_ROTARY_PACKED_WARPS 1\n" if packed else ""
    return prefix + _fasteq_rotation_primitives() + defines + tables + body


def _rotary_entry(role):
    return ("fasteq_rotary_dquery", "fasteq_rotary_dkey",
            "fasteq_rotary_dphase", "fasteq_rotary_forward")[role]


@lru_cache(maxsize=128)
def _packed_rotary_kernel(dtype, role, device):
    """Keep fixed TECE C64/H4 executables resident across eager/Graph calls."""
    from fasteq.cuda.jit import kernels
    code = _rotary_source(dtype, 2, 1, 64, 4, role)
    return kernels([code], device, entry=_rotary_entry(role))[code]


def _launch(code, values, output, groups, *, entry="run"):
    from fasteq.cuda.jit import kernels, runtime

    if not values[0].is_cuda or values[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Frame operators require CUDA FP32/FP64")
    if any(v.device != values[0].device or v.dtype != values[0].dtype for v in values):
        raise ValueError("Frame operands must share device and dtype")
    values = [v.contiguous() for v in values]
    if not values[0].shape[0] or not output.numel() or not groups:
        return output.zero_()
    kernel = kernels([code], output.device, entry=entry)[code]
    args = [v.data_ptr() for v in (*values, output)]
    runtime().launch(
        [(kernel, args, values[0].shape[0], groups, 128, 0)],
        torch.cuda.current_stream(output.device).cuda_stream,
    )
    return output


class _Frame(torch.autograd.Function):
    @staticmethod
    def forward(ctx, metadata, role, *values):
        features, matrix, _edge = values
        ctx.kernel_metadata, ctx.role = metadata, role
        ctx.save_for_backward(*values)
        if role == 0:
            result = features.new_empty(features.shape)
        elif role == 1:
            result = matrix.new_zeros(matrix.shape)
        else:
            result = features.new_empty(
                features.shape[0], matrix.shape[1], features.shape[2]
            )
        # The native CUDA loader checks each actual vector address and tail;
        # odd storage offsets and non-multiple channels use scalar loads.
        code, groups = _frame_source(features.dtype, features.shape[2], metadata, role)
        return _launch(code, values, result, groups, entry=_frame_entry(role))

    @staticmethod
    def backward(ctx, gradient):
        values = list(ctx.saved_tensors)
        values[ctx.role] = gradient
        return (
            None,
            None,
            *[
                _Frame.apply(ctx.kernel_metadata, role, *values)
                if role != ctx.role and need
                else None
                for role, need in enumerate(ctx.needs_input_grad[2:])
            ],
        )


def rotate(features, matrix, metadata):
    """Rotate matching degree blocks with complete matrix/feature adjoints."""
    empty = features.new_empty(0, matrix.shape[1], features.shape[2])
    return _Frame.apply(metadata, 2, features, matrix, empty)


class _Rotary(torch.autograd.Function):
    @staticmethod
    def forward(ctx, lmax, mmax, heads, role, *values):
        query, _key, phase, _score = values
        ctx.lmax, ctx.mmax, ctx.heads, ctx.role = lmax, mmax, heads, role
        ctx.save_for_backward(*values)
        if role < 2:
            result = query.new_empty(query.shape)
        elif role == 2:
            result = phase.new_empty(phase.shape)
        else:
            result = query.new_empty(query.shape[0], heads)
        if (lmax, mmax, query.shape[2], heads) == (2, 1, 64, 4):
            if not query.is_cuda or query.dtype not in (torch.float32, torch.float64):
                raise TypeError("Frame operators require CUDA FP32/FP64")
            if any(v.device != query.device or v.dtype != query.dtype for v in values):
                raise ValueError("Frame operands must share device and dtype")
            if not query.shape[0] or not result.numel():
                return result.zero_()
            packed_values = [v.contiguous() for v in values]
            kernel = _packed_rotary_kernel(query.dtype, role, query.device)
            # HC16: two heads fill one warp; four warps own independent tasks.
            groups = 3 if role < 2 else 2 if role == 2 else 1
            kernel.launch(
                [v.data_ptr() for v in (*packed_values, result)],
                query.shape[0], groups, 128,
                torch.cuda.current_stream(query.device).cuda_stream, 0,
            )
            return result
        angular, channels = query.shape[1:]
        pairs = len(_rotary_layout(lmax, mmax)[0])
        groups = (heads * ((pairs + 3) // 4) * ((channels // heads + 31) // 32)
                  if role < 2 else (mmax + 1) * heads if role == 2 else heads)
        code = _rotary_source(query.dtype, lmax, mmax, channels, heads, role)
        return _launch(code, values, result, groups, entry=_rotary_entry(role))

    @staticmethod
    def backward(ctx, gradient):
        values = list(ctx.saved_tensors)
        values[ctx.role] = gradient
        return (
            None,
            None,
            None,
            None,
            *[
                _Rotary.apply(ctx.lmax, ctx.mmax, ctx.heads, role, *values)
                if role != ctx.role and need
                else None
                for role, need in enumerate(ctx.needs_input_grad[4:])
            ],
        )


def rotary_product(query, key, phase, lmax, mmax, heads):
    """Contract packed Q/K and phases; flattened phase views are supported."""
    counts = (lmax, mmax, heads)
    if any(not isinstance(value, Integral) or isinstance(value, bool) for value in counts):
        raise TypeError("Rotary lmax, mmax and heads must be integers")
    if not 0 <= mmax <= lmax or heads <= 0:
        raise ValueError("Rotary requires 0 <= mmax <= lmax and heads > 0")
    if not all(isinstance(value, torch.Tensor) for value in (query, key, phase)):
        raise TypeError("Rotary inputs must be tensors")
    if query.ndim != 3 or key.shape != query.shape:
        raise ValueError("Rotary query/key must have the same (N,A,C) shape")
    rows, angular, channels = query.shape
    if angular != _rotary_layout(lmax, mmax)[3] or channels % heads:
        raise ValueError("Rotary angular width or channel/head division does not match metadata")
    if phase.numel() != rows * (mmax + 1) * heads * 2:
        raise ValueError("Rotary phase must contain N*(mmax+1)*heads*2 packed values")
    if not all(value.is_cuda for value in (query, key, phase)):
        raise TypeError("Rotary inputs require CUDA")
    if query.dtype not in (torch.float32, torch.float64):
        raise TypeError("Rotary inputs require FP32/FP64")
    if any(value.device != query.device or value.dtype != query.dtype for value in (key, phase)):
        raise ValueError("Rotary inputs must share device and dtype")
    placeholder = query.new_empty(0, heads)
    return _Rotary.apply(lmax, mmax, heads, 3, query, key, phase, placeholder)


@lru_cache(maxsize=32)
def _quaternion_coefficients(lmax):
    """Build real harmonic coefficients by an integer polynomial recurrence.

    FastEq's polynomial DAG uses multiplication by linear forms. Apply the
    same construction to Sym^n of the two SU(2) columns, retaining integer
    coefficients and the implicit phase i**(power_y+power_z). Multiplying one
    column at a time replaces combinatorial expansion by a triangular degree
    recurrence. Real basis changes use Gaussian integers before normalization.
    """
    lmax = int(lmax)
    if lmax < 0:
        raise ValueError("Quaternion harmonic degree must be nonnegative")
    pointers, exponents, coefficients = [0], [], []
    # Each column contains coefficients in the auxiliary monomial basis
    # X**(n-row) Y**row. Physical monomials are (w,x,y,z) exponent tuples.
    columns = [[{(0, 0, 0, 0): 1}]]
    column_zero = (((0, 1), (2, -1)), ((1, -1), (3, 1)))
    column_one = (((1, 1), (3, 1)), ((0, 1), (2, 1)))
    for order in range(2 * lmax + 1):
        if order:
            extended = []
            for column in range(order + 1):
                previous = columns[column if column < order else column - 1]
                linear = column_zero if column < order else column_one
                entries = []
                for row in range(order + 1):
                    polynomial = {}
                    for old_row, terms in ((row, linear[0]), (row - 1, linear[1])):
                        if not 0 <= old_row < order:
                            continue
                        for powers, value in previous[old_row].items():
                            for axis, sign in terms:
                                shifted = list(powers); shifted[axis] += 1
                                key = tuple(shifted)
                                polynomial[key] = polynomial.get(key, 0) + sign * value
                    entries.append({key: value for key, value in polynomial.items() if value})
                extended.append(entries)
            columns = extended
        if order % 2:
            continue
        degree = order // 2
        basis = []
        for magnetic in range(-degree, degree + 1):
            parity = -1 if magnetic % 2 else 1
            if magnetic < 0:
                basis.append(((degree + magnetic, 0, -1), (degree - magnetic, 0, parity)))
            elif magnetic:
                basis.append(((degree - magnetic, 1, 0), (degree + magnetic, parity, 0)))
            else:
                basis.append(((degree, 1, 0),))
        for row, row_basis in enumerate(basis):
            for column, column_basis in enumerate(basis):
                real, imaginary = {}, {}
                for i, rr, ri in row_basis:
                    for j, cr, ci in column_basis:
                        factor_real, factor_imaginary = rr * cr + ri * ci, rr * ci - ri * cr
                        phases = ((factor_real, factor_imaginary), (-factor_imaginary, factor_real),
                                  (-factor_real, -factor_imaginary), (factor_imaginary, -factor_real))
                        for powers, value in columns[j][i].items():
                            re, im = phases[(powers[2] + powers[3]) % 4]
                            if re:
                                real[powers] = real.get(powers, 0) + re * value
                            if im:
                                imaginary[powers] = imaginary.get(powers, 0) + im * value
                if any(imaginary.values()):
                    raise RuntimeError("Real harmonic change of basis produced an imaginary coefficient")
                normalization = math.sqrt(math.comb(order, column) / math.comb(order, row))
                normalization /= math.sqrt(len(row_basis) * len(column_basis))
                for powers, value in sorted(real.items()):
                    if value:
                        exponents.append(powers)
                        coefficients.append(value * normalization)
                pointers.append(len(coefficients))
    return (
        torch.tensor(pointers, dtype=torch.int32, device="cpu"),
        torch.tensor(exponents, dtype=torch.int32, device="cpu"),
        torch.tensor(coefficients, dtype=torch.float64, device="cpu"),
    )


@lru_cache(maxsize=4)
def _alignment_quaternion(dtype):
    from fasteq.ops.scalar_expression import ScalarProgram

    p = ScalarProgram()
    zero, one, _two, half = map(p.constant, (0, 1, 2, 0.5))
    eps2 = p.constant(1e-14)
    machine_eps = p.constant(2 ** (-23 if dtype == "float" else -52))

    def sum_values(values):
        result = zero
        for value in values:
            result = p.add("add", result, value)
        return result

    def normalize(values):
        norm = p.add(
            "sqrt", p.add("add", sum_values([p.add("mul", v, v) for v in values]), eps2)
        )
        return [p.add("div", v, norm) for v in values]

    x, y, z = normalize([p.add("input", 0, i) for i in range(3)])
    positive = normalize([p.add("add", one, y), p.add("neg", z), zero, x])
    negative = normalize([p.add("neg", z), p.add("add", one, p.add("neg", y)), x, zero])
    t = p.add("mul", half, p.add("add", y, one))
    t = p.add(
        "select",
        p.add("lt", t, zero),
        zero,
        p.add("select", p.add("lt", one, t), one, t),
    )

    def smooth_exp(value):
        clamped = p.add("select", p.add("lt", value, machine_eps), machine_eps, value)
        return p.add("exp", p.add("neg", p.add("div", one, clamped)))

    left, right = smooth_exp(t), smooth_exp(p.add("add", one, p.add("neg", t)))
    blend = p.add("div", left, p.add("add", left, right))
    blend = p.add(
        "select",
        p.add("le", t, zero),
        zero,
        p.add("select", p.add("le", one, t), one, blend),
    )
    dot = sum_values([p.add("mul", a, b) for a, b in zip(positive, negative)])
    positive = [
        p.add("select", p.add("lt", dot, zero), p.add("neg", q), q) for q in positive
    ]
    w, x, y, z = normalize(
        [
            p.add(
                "add",
                p.add("mul", p.add("add", one, p.add("neg", blend)), a),
                p.add("mul", blend, b),
            )
            for a, b in zip(negative, positive)
        ]
    )

    return tuple(p.nodes), (w, x, y, z)


@lru_cache(maxsize=64)
def _wigner_metadata(lmax, dtype):
    from fasteq.ops.scalar_expression import ScalarProgram

    nodes, quaternion = _alignment_quaternion(dtype)
    p = ScalarProgram(nodes)
    pointers, exponents, coefficients = _quaternion_coefficients(lmax)
    pointers, exponents, coefficients = (
        pointers.tolist(),
        exponents.tolist(),
        coefficients.tolist(),
    )
    powers = []
    for component in quaternion:
        row = [p.constant(1)]
        for degree in range(1, 2 * lmax + 1):
            row.append(p.add("mul", row[-1], component))
        powers.append(row)
    outputs = []
    for start, stop in pairwise(pointers):
        value = p.constant(0)
        for index in range(start, stop):
            term = p.constant(coefficients[index])
            for axis, power in enumerate(exponents[index]):
                term = p.add("mul", term, powers[axis][power])
            value = p.add("add", value, term)
        outputs.append(value)
    return repr((tuple(p.nodes), (tuple(outputs),), (3,)))


def wigner_packed(vectors, lmax):
    """Evaluate real quaternion Wigner polynomials in FastEq's scalar CUDA IR."""
    from fasteq.ops.scalar_expression import scalar_expression

    if vectors.ndim != 2 or vectors.shape[1] != 3:
        raise ValueError("Wigner vectors must have shape (edges, 3)")
    if not vectors.is_cuda or vectors.dtype not in (torch.float32, torch.float64):
        raise TypeError("FastEq Wigner polynomials require CUDA FP32/FP64")
    dtype = "float" if vectors.dtype == torch.float32 else "double"
    return scalar_expression(_wigner_metadata(lmax, dtype), [vectors])[0]
