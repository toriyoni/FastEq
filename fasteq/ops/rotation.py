"""FastEq CUDA frame rotations and rotary contractions.

CUDA expression algorithms adapted from EQX (Zemin Xu), CC-BY-4.0.
This module owns its compiled kernels, launches and recursive differentiation;
it never calls an EQX CUDA operator or launcher.
"""

import math
import os
from ast import literal_eval
from collections import defaultdict
from functools import cache, lru_cache
from itertools import pairwise

import torch


@lru_cache(maxsize=128)
def _frame_source(dtype, channels, metadata, role, strategy="scalar"):
    if strategy == "vector":
        if dtype != torch.float64 or channels != 128 or role not in (0, 2):
            raise ValueError("Frame vector source requires FP64 C128 and feature roles")
        return _frame_vector_source(metadata, role)
    if strategy != "scalar":
        raise ValueError("Unknown frame source strategy")
    degrees_in, degrees_out = literal_eval(metadata)
    ni, no = len(degrees_in), len(degrees_out)
    scalar = "double" if dtype == torch.float64 else "float"
    groups = []
    if role == 1:
        pairs = [
            (i, j)
            for i, a in enumerate(degrees_out)
            for j, b in enumerate(degrees_in)
            if a == b
        ]
        rows = ",".join(str(i) for i, _ in pairs)
        columns = ",".join(str(j) for _, j in pairs)
        body = f"""
        long long e=blockIdx.x; int p=blockIdx.y*4+threadIdx.x/32,lane=threadIdx.x%32;
        if(p>={len(pairs)}) return;
        int i=rows[p],j=columns[p]; scalar sum=0;
        for(int c=lane;c<{channels};c+=32)
            sum+=features[(e*{ni}+j)*{channels}+c]*edge[(e*{no}+i)*{channels}+c];
        for(int offset=16;offset;offset>>=1) sum+=__shfl_down_sync(0xffffffff,sum,offset);
        if(!lane) result[(e*{no}+i)*{ni}+j]=sum;
        """
        constants = (
            f"__device__ __constant__ int rows[]={{{rows}}},columns[]={{{columns}}};"
        )
        grid = (len(pairs) + 3) // 4
    else:
        outputs, inputs = (
            (degrees_out, degrees_in) if role == 2 else (degrees_in, degrees_out)
        )
        for i, degree in enumerate(outputs):
            terms = []
            for j, other in enumerate(inputs):
                if degree != other:
                    continue
                if role == 2:
                    terms.append(
                        f"sum+=matrix[(e*{no}+{i})*{ni}+{j}]*features[(e*{ni}+{j})*{channels}+c];"
                    )
                else:
                    terms.append(
                        f"sum+=matrix[(e*{no}+{j})*{ni}+{i}]*edge[(e*{no}+{j})*{channels}+c];"
                    )
            groups.append(f"case {i}: {{ {''.join(terms)} break; }}")
        constants = ""
        body = f"""
        long long e=blockIdx.x; int ac=blockIdx.y*128+threadIdx.x;
        if(ac>={len(outputs) * channels}) return;
        int a=ac/{channels},c=ac%{channels}; scalar sum=0;
        switch(a) {{ {"".join(groups)} }}
        result[e*{len(outputs) * channels}+ac]=sum;
        """
        grid = (len(outputs) * channels + 127) // 128
    return (
        f"""
    using scalar={scalar};
    {constants}
    extern "C" __global__ void run(const scalar* features,const scalar* matrix,
        const scalar* edge,scalar* result) {{ {body} }}
    """,
        grid,
    )


@lru_cache(maxsize=128)
def _frame_vector_source(metadata, role):
    assert role in (0, 2)
    degrees_in, degrees_out = literal_eval(metadata)
    ni, no = len(degrees_in), len(degrees_out)
    outputs, inputs = ((degrees_out, degrees_in) if role == 2
                       else (degrees_in, degrees_out))
    groups = []
    for i, degree in enumerate(outputs):
        terms = []
        for j, other in enumerate(inputs):
            if degree != other:
                continue
            coefficient = (f"matrix[(e*{no}+{i})*{ni}+{j}]" if role == 2
                           else f"matrix[(e*{no}+{j})*{ni}+{i}]")
            operand = (f"features+(e*{ni}+{j})*128+c" if role == 2
                       else f"edge+(e*{no}+{j})*128+c")
            terms.append("{"
                f"double weight={coefficient}; const double* row={operand};"
                "double2 p0=*reinterpret_cast<const double2*>(row);"
                "double2 p1=*reinterpret_cast<const double2*>(row+64);"
                "s0.x+=weight*p0.x;s0.y+=weight*p0.y;"
                "s1.x+=weight*p1.x;s1.y+=weight*p1.y;}")
        groups.append(f"case {i}: {{{''.join(terms)} break;}}")
    return f"""
    extern "C" __global__ void run(const double* features,const double* matrix,
        const double* edge,double* result) {{
        long long e=blockIdx.x;
        int a=blockIdx.y*4+threadIdx.x/32,c=2*(threadIdx.x%32);
        if(a>={len(outputs)}) return;
        double2 s0=make_double2(0,0),s1=make_double2(0,0);
        switch(a) {{{''.join(groups)}}}
        double* row=result+(e*{len(outputs)}+a)*128+c;
        *reinterpret_cast<double2*>(row)=s0;
        *reinterpret_cast<double2*>(row+64)=s1;
    }}
    """, (len(outputs)+3)//4


@lru_cache(maxsize=128)
def _rotary_source(dtype, lmax, mmax, channels, heads, role):
    orders, partners, signs = [], [], []
    starts, offset = [], 0
    for m in range(mmax + 1):
        n = lmax + 1 - m
        starts.append(offset)
        for real in range(1 if m == 0 else 2):
            orders.extend([m] * n)
            partners.extend(
                range(offset + (1 - real) * n, offset + (2 - real) * n)
                if m
                else range(n)
            )
            signs.extend([(-1 if real == 0 else 1) if m else 0] * n)
        offset += n * (1 if m == 0 else 2)
    angular, per_head = offset, channels // heads
    constants = "\n".join(
        f"__device__ __constant__ int {name}[]={{{','.join(map(str, values))}}};"
        for name, values in (
            ("order", orders),
            ("partner", partners),
            ("sign", signs),
            ("start", starts),
        )
    )
    if role in (0, 1):
        other = "key" if role == 0 else "query"
        cross_sign = "sign[a]" if role == 0 else "-sign[a]"
        body = f"""
        long long e=blockIdx.x; int ac=blockIdx.y*128+threadIdx.x;
        if(ac>={angular * channels}) return;
        int a=ac/{channels},c=ac%{channels},h=c/{per_head};
        long long p=((e*{mmax + 1}+order[a])*{heads}+h)*2;
        scalar value=phase[p]*{other}[e*{angular * channels}+ac];
        if(sign[a]) value+=scalar({cross_sign})*phase[p+1]*{other}[(e*{angular}+partner[a])*{channels}+c];
        result[e*{angular * channels}+ac]=value*score[e*{heads}+h];
        """
    elif role == 2:
        body = f"""
        long long e=blockIdx.x; int m=blockIdx.y/{heads},h=blockIdx.y%{heads};
        int n={lmax + 1}-m; scalar real=0,imag=0;
        for(int i=threadIdx.x;i<n*{per_head};i+=128) {{
            int a=start[m]+i/{per_head},c=h*{per_head}+i%{per_head};
            long long pos=(e*{angular}+a)*{channels}+c;
            scalar qr=query[pos],kr=key[pos]; real+=qr*kr;
            if(m) {{
                scalar qi=query[pos+n*{channels}],ki=key[pos+n*{channels}];
                real+=qi*ki; imag+=qi*kr-qr*ki;
            }}
        }}
        __shared__ scalar re[128],im[128]; re[threadIdx.x]=real; im[threadIdx.x]=imag;
        __syncthreads();
        for(int s=64;s;s>>=1) {{
            if(threadIdx.x<s) {{ re[threadIdx.x]+=re[threadIdx.x+s]; im[threadIdx.x]+=im[threadIdx.x+s]; }}
            __syncthreads();
        }}
        if(!threadIdx.x) {{
            long long p=((e*{mmax + 1}+m)*{heads}+h)*2;
            result[p]=re[0]*score[e*{heads}+h]; result[p+1]=im[0]*score[e*{heads}+h];
        }}
        """
    else:
        body = f"""
        long long e=blockIdx.x; int h=blockIdx.y; scalar sum=0;
        for(int i=threadIdx.x;i<{angular * per_head};i+=128) {{
            int a=i/{per_head},c=h*{per_head}+i%{per_head};
            long long p=((e*{mmax + 1}+order[a])*{heads}+h)*2;
            scalar k=phase[p]*key[(e*{angular}+a)*{channels}+c];
            if(sign[a]) k+=scalar(sign[a])*phase[p+1]*key[(e*{angular}+partner[a])*{channels}+c];
            sum+=query[(e*{angular}+a)*{channels}+c]*k;
        }}
        __shared__ scalar values[128]; values[threadIdx.x]=sum; __syncthreads();
        for(int s=64;s;s>>=1) {{
            if(threadIdx.x<s) values[threadIdx.x]+=values[threadIdx.x+s];
            __syncthreads();
        }}
        if(!threadIdx.x) result[e*{heads}+h]=values[0];
        """
    scalar = "double" if dtype == torch.float64 else "float"
    return f"""
    using scalar={scalar};
    {constants}
    extern "C" __global__ void run(const scalar* query,const scalar* key,
        const scalar* phase,const scalar* score,scalar* result) {{ {body} }}
    """


def _launch(code, values, output, groups):
    from fasteq.cuda.jit import kernels, runtime

    if not values[0].is_cuda or values[0].dtype not in (torch.float32, torch.float64):
        raise TypeError("Frame operators require CUDA FP32/FP64")
    if any(v.device != values[0].device or v.dtype != values[0].dtype for v in values):
        raise ValueError("Frame operands must share device and dtype")
    values = [v.contiguous() for v in values]
    if not values[0].shape[0] or not output.numel() or not groups:
        return output.zero_()
    kernel = kernels([code], output.device)[code]
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
        # The read operand may be a contiguous view with an odd offset.
        # Keep scalar CUDA unless both actual vector pointers are aligned.
        vector_operand = features if role == 2 else values[2]
        strategy = (
            "vector"
            if features.shape[0] >= 16384
            and features.dtype == torch.float64 and features.shape[2] == 128
            and role in (0, 2)
            and os.environ.get("FASTEQ_EXPERIMENTAL_ROTATE_VECTOR") == "1"
            and vector_operand.is_contiguous() and vector_operand.data_ptr() % 16 == 0
            and result.is_contiguous() and result.data_ptr() % 16 == 0
            else "scalar"
        )
        code, groups = _frame_source(features.dtype, features.shape[2], metadata, role, strategy)
        return _launch(code, values, result, groups)

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
            result = torch.empty_like(query)
        elif role == 2:
            result = torch.empty_like(phase)
        else:
            result = query.new_empty(query.shape[0], heads)
        angular, channels = query.shape[1:]
        groups = (
            (angular * channels + 127) // 128
            if role < 2
            else (mmax + 1) * heads
            if role == 2
            else heads
        )
        code = _rotary_source(query.dtype, lmax, mmax, channels, heads, role)
        return _launch(code, values, result, groups)

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
    """Compute the head-wise real/imaginary frame contraction in FastEq CUDA."""
    placeholder = query.new_empty(0, heads)
    return _Rotary.apply(lmax, mmax, heads, 3, query, key, phase, placeholder)


@lru_cache(maxsize=32)
def _quaternion_coefficients(lmax):
    """Expand the symmetric-power representation in the real harmonic basis.

    Polynomial coefficients are accumulated as integers before applying the
    harmonic normalization. Structural zeros therefore need no threshold or
    numerical fit. The spin-half matrix is
    ``[[w - iy, x + iz], [-x + iz, w + iy]]``.
    """
    pointers, exponents, coefficients = [0], [], []
    for l in range(lmax + 1):
        n = 2 * l

        @cache
        def complex_polynomial(row, column, n=n):
            result = defaultdict(int)
            for r in range(max(0, row - column), min(n - column, row) + 1):
                a, b, c, d = n - column - r, column - row + r, r, row - r
                wy, xz = [0] * (a + d + 1), [0] * (b + c + 1)
                for i in range(a + 1):
                    for j in range(d + 1):
                        wy[i + j] += (-1) ** i * math.comb(a, i) * math.comb(d, j)
                for i in range(b + 1):
                    for j in range(c + 1):
                        xz[i + j] += (-1) ** (c - j) * math.comb(b, i) * math.comb(c, j)
                scale = math.comb(n - column, r) * math.comb(column, row - r)
                for y, cy in enumerate(wy):
                    for z, cz in enumerate(xz):
                        if cy and cz:
                            result[a + d - y, b + c - z, y, z] += scale * cy * cz
            return result

        # Each entry is (complex index, power of i). The common degree phase
        # cancels between the real-to-complex matrix and its adjoint.
        basis = []
        for m in range(-l, l + 1):
            if m < 0:
                basis.append(((l + m, 3), (l - m, (1 - 2 * m) % 4)))
            elif m > 0:
                basis.append(((l - m, 0), (l + m, (2 * m) % 4)))
            else:
                basis.append(((l, 0),))
        for row in range(n + 1):
            for column in range(n + 1):
                real, imaginary = defaultdict(int), defaultdict(int)
                for i, pi in basis[row]:
                    for j, pj in basis[column]:
                        for powers, value in complex_polynomial(i, j).items():
                            phase = (powers[2] + powers[3] + pj - pi) % 4
                            (real if phase % 2 == 0 else imaginary)[powers] += (
                                value if phase < 2 else -value
                            )
                if any(imaginary.values()):
                    raise RuntimeError("The quaternion polynomial must be real.")
                scale = math.sqrt(math.comb(n, column) / math.comb(n, row))
                scale /= math.sqrt(len(basis[row]) * len(basis[column]))
                for powers, value in sorted(real.items()):
                    if value:
                        exponents.append(powers)
                        coefficients.append(value * scale)
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
