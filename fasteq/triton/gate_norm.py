"""Residual/Gate/RMS fusion using the existing FastEq mathematical primitives.

Gate activation derivatives and weighted moments/reduction are reused from the
MIT FastEq d37eaca kernels. The composition and chain rule below are new.
"""

import triton
import triton.language as tl

from ..ops.equivariant_norm import _parameters, _selected_gamma_partials
from .fused_equivariant_gate import _activation, _activation_d1, _activation_d2
from .fused_equivariant_layer_norm import _moment


@triton.jit
def _read_sum(P, S, row, coordinate, valid, DIN: tl.constexpr):
    return tl.load(P + row * DIN + coordinate, valid, 0.0) + tl.load(
        S + row * DIN + coordinate, valid, 0.0
    )


@triton.jit
def _read_direction(U, row, coordinate, valid, DIN: tl.constexpr, HAS_U: tl.constexpr):
    return tl.load(U + row * DIN + coordinate, valid & HAS_U, 0.0)


@triton.jit
def _gate(P, S, SRC, GATE, ACT, SCALE, row, ids, DIN: tl.constexpr, DOUT: tl.constexpr):
    valid = ids < DOUT
    source = tl.load(SRC + ids, valid, 0)
    gate = tl.load(GATE + ids, valid, -1)
    code = tl.load(ACT + ids, valid, 0)
    scale = tl.load(SCALE + ids, valid, 0.0)
    xs = _read_sum(P, S, row, source, valid, DIN)
    xg = _read_sum(P, S, row, tl.maximum(gate, 0), valid & (gate >= 0), DIN)
    activation = _activation(tl.where(gate >= 0, xg, xs), code, scale)
    return tl.where(gate >= 0, xs * activation, activation)


@triton.jit
def _gate_tangent(
    P,
    S,
    U,
    SRC,
    GATE,
    ACT,
    SCALE,
    row,
    ids,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    HAS_U: tl.constexpr,
):
    valid = ids < DOUT
    source = tl.load(SRC + ids, valid, 0)
    gate = tl.load(GATE + ids, valid, -1)
    code = tl.load(ACT + ids, valid, 0)
    scale = tl.load(SCALE + ids, valid, 0.0)
    xs = _read_sum(P, S, row, source, valid, DIN)
    xg = _read_sum(P, S, row, tl.maximum(gate, 0), valid & (gate >= 0), DIN)
    us = _read_direction(U, row, source, valid, DIN, HAS_U)
    ug = _read_direction(U, row, tl.maximum(gate, 0), valid & (gate >= 0), DIN, HAS_U)
    return tl.where(
        gate >= 0,
        us * _activation(xg, code, scale) + xs * _activation_d1(xg, code, scale) * ug,
        _activation_d1(xs, code, scale) * us,
    )


@triton.jit
def _gate_transpose(
    P,
    S,
    U,
    G,
    H,
    SRC,
    GATE,
    ACT,
    SCALE,
    IN_OUT,
    IN_ROLE,
    row,
    ids,
    DIN: tl.constexpr,
    MAX_FAN: tl.constexpr,
    HAS_U: tl.constexpr,
    SECOND: tl.constexpr,
):
    result = tl.full(ids.shape, 0.0, G.dtype)
    for fan in tl.static_range(MAX_FAN):
        output = tl.load(IN_OUT + ids * MAX_FAN + fan, ids < DIN, -1)
        role = tl.load(IN_ROLE + ids * MAX_FAN + fan, ids < DIN, -1)
        valid = (ids < DIN) & (output >= 0)
        output = tl.maximum(output, 0)
        source = tl.load(SRC + output, valid, 0)
        gate = tl.load(GATE + output, valid, -1)
        code = tl.load(ACT + output, valid, 0)
        scale = tl.load(SCALE + output, valid, 0.0)
        xs = _read_sum(P, S, row, source, valid, DIN)
        xg = _read_sum(P, S, row, tl.maximum(gate, 0), valid & (gate >= 0), DIN)
        a = _activation(xg, code, scale)
        d1 = _activation_d1(xg, code, scale)
        jacobian = tl.where(
            role == 0, _activation_d1(xs, code, scale), tl.where(role == 1, a, xs * d1)
        )
        contribution = tl.gather(G, output, 0) * jacobian
        if SECOND:
            us = _read_direction(U, row, source, valid, DIN, HAS_U)
            ug = _read_direction(
                U, row, tl.maximum(gate, 0), valid & (gate >= 0), DIN, HAS_U
            )
            curvature = tl.where(
                role == 0,
                _activation_d2(xs, code, scale) * us,
                tl.where(
                    role == 1,
                    d1 * ug,
                    d1 * us + xs * _activation_d2(xg, code, scale) * ug,
                ),
            )
            contribution += tl.gather(H, output, 0) * curvature
        result += tl.where(valid, contribution, 0.0)
    return result


@triton.jit
def forward(
    P,
    S,
    SRC,
    GATE,
    ACT,
    SCALE,
    COEF,
    GAMMA,
    Y,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    BLOCK: tl.constexpr,
    PARTS: tl.constexpr,
    GFLAGS: tl.constexpr,
    EPS: tl.constexpr,
    FP64: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ids = tl.arange(0, BLOCK)
    value = _gate(P, S, SRC, GATE, ACT, SCALE, row, ids, DIN, DOUT)
    coefficient = tl.load(COEF + ids, ids < DOUT, 0.0)
    variance = _moment(
        value[:, None] * value[:, None], coefficient, 1.0, 1, "components_first", False
    )
    r = tl.rsqrt(variance + EPS)
    gamma = _parameters(GAMMA, ids, PARTS, False, False, GFLAGS, 1.0, FP64)
    tl.store(Y + row * DOUT + ids, value * r * gamma, ids < DOUT)


@triton.jit
def backward(
    P,
    S,
    DY,
    SRC,
    GATE,
    ACT,
    SCALE,
    IN_OUT,
    IN_ROLE,
    COEF,
    GAMMA,
    DZ,
    PARTIALS,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_FAN: tl.constexpr,
    PARTS: tl.constexpr,
    GFLAGS: tl.constexpr,
    GOFFSETS: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    FP64: tl.constexpr,
    DZFLAG: tl.constexpr,
    GGRADFLAGS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ids = tl.arange(0, BLOCK)
    value = _gate(P, S, SRC, GATE, ACT, SCALE, row, ids, DIN, DOUT)
    dy = tl.load(DY + row * DOUT + ids, ids < DOUT, 0.0)
    coefficient = tl.load(COEF + ids, ids < DOUT, 0.0)
    variance = _moment(
        value[:, None] * value[:, None], coefficient, 1.0, 1, "components_first", False
    )
    r = tl.rsqrt(variance + EPS)
    if DZFLAG:
        gamma = _parameters(GAMMA, ids, PARTS, False, False, GFLAGS, 1.0, FP64)
        g = dy * gamma
        a = tl.sum(g * value, 0)
        norm_gradient = r * g - r * r * r * coefficient * value * a
        dz = _gate_transpose(
            P,
            S,
            P,
            norm_gradient,
            norm_gradient,
            SRC,
            GATE,
            ACT,
            SCALE,
            IN_OUT,
            IN_ROLE,
            row,
            ids,
            DIN,
            MAX_FAN,
            False,
            False,
        )
        tl.store(DZ + row * DIN + ids, dz, ids < DIN)
    _selected_gamma_partials(
        dy * value * r, PARTIALS, row, WIDTH, PARTS, GOFFSETS, False, GGRADFLAGS
    )


@triton.jit
def double_backward(
    P,
    S,
    DY,
    U,
    H,
    SRC,
    GATE,
    ACT,
    SCALE,
    IN_OUT,
    IN_ROLE,
    COEF,
    GAMMA,
    DDZ,
    DDY,
    PARTIALS,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_FAN: tl.constexpr,
    PARTS: tl.constexpr,
    GFLAGS: tl.constexpr,
    HFLAGS: tl.constexpr,
    HAS_U: tl.constexpr,
    GOFFSETS: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    FP64: tl.constexpr,
    DDZFLAG: tl.constexpr,
    DDYFLAG: tl.constexpr,
    GGRADFLAGS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ids = tl.arange(0, BLOCK)
    value = _gate(P, S, SRC, GATE, ACT, SCALE, row, ids, DIN, DOUT)
    t = _gate_tangent(P, S, U, SRC, GATE, ACT, SCALE, row, ids, DIN, DOUT, HAS_U)
    dy = tl.load(DY + row * DOUT + ids, ids < DOUT, 0.0)
    coefficient = tl.load(COEF + ids, ids < DOUT, 0.0)
    variance = _moment(
        value[:, None] * value[:, None], coefficient, 1.0, 1, "components_first", False
    )
    r = tl.rsqrt(variance + EPS)
    gamma = _parameters(GAMMA, ids, PARTS, False, False, GFLAGS, 1.0, FP64)
    h = _parameters(H, ids, PARTS, False, False, HFLAGS, 0.0, FP64)
    g = dy * gamma
    a = tl.sum(g * value, 0)
    s = tl.sum(t * coefficient * value, 0)
    c = tl.sum(t * g + dy * h * value, 0)
    r3 = r * r * r
    if DDZFLAG:
        norm_gradient = r * g - r3 * coefficient * value * a
        norm_double = (
            r * dy * h
            - coefficient * value * r3 * c
            + 3.0 * coefficient * value * r3 * r * r * a * s
            - r3 * (s * g + a * coefficient * t)
        )
        ddz = _gate_transpose(
            P,
            S,
            U,
            norm_double,
            norm_gradient,
            SRC,
            GATE,
            ACT,
            SCALE,
            IN_OUT,
            IN_ROLE,
            row,
            ids,
            DIN,
            MAX_FAN,
            HAS_U,
            True,
        )
        tl.store(DDZ + row * DIN + ids, ddz, ids < DIN)
    if DDYFLAG:
        ddy = r * (t * gamma + h * value) - r3 * s * value * gamma
        tl.store(DDY + row * DOUT + ids, ddy, ids < DOUT)
    _selected_gamma_partials(
        dy * (r * t - r3 * s * value),
        PARTIALS,
        row,
        WIDTH,
        PARTS,
        GOFFSETS,
        False,
        GGRADFLAGS,
    )


@triton.jit
def _gate_transpose_direct(
    P,
    S,
    DY,
    U,
    H,
    SRC,
    GATE,
    ACT,
    SCALE,
    IN_OUT,
    IN_ROLE,
    COEF,
    GAMMA,
    row,
    ids,
    r,
    a,
    s,
    c,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    PARTS: tl.constexpr,
    GFLAGS: tl.constexpr,
    HFLAGS: tl.constexpr,
    FP64: tl.constexpr,
    MAX_FAN: tl.constexpr,
    HAS_U: tl.constexpr,
    SECOND: tl.constexpr,
):
    """Rematerialize each selected output from scalar moments, not tl.gather.

    Full-row values die before this loop. Output ordering and the fan reduction
    order are unchanged; the only duplicated work is pointwise Gate evaluation.
    """
    result = tl.full(ids.shape, 0.0, tl.float64 if FP64 else tl.float32)
    r3 = r * r * r
    for fan in tl.range(0, MAX_FAN, loop_unroll_factor=1):
        output = tl.load(IN_OUT + ids * MAX_FAN + fan, ids < DIN, -1)
        role = tl.load(IN_ROLE + ids * MAX_FAN + fan, ids < DIN, -1)
        valid = (ids < DIN) & (output >= 0)
        output = tl.maximum(output, 0)
        source = tl.load(SRC + output, valid, 0)
        gate = tl.load(GATE + output, valid, -1)
        code = tl.load(ACT + output, valid, 0)
        scale = tl.load(SCALE + output, valid, 0.0)
        xs = _read_sum(P, S, row, source, valid, DIN)
        xg = _read_sum(P, S, row, tl.maximum(gate, 0), valid & (gate >= 0), DIN)
        activation = _activation(tl.where(gate >= 0, xg, xs), code, scale)
        value = tl.where(gate >= 0, xs * activation, activation)
        dy = tl.load(DY + row * DOUT + output, valid, 0.0)
        coefficient = tl.load(COEF + output, valid, 0.0)
        gamma = _parameters(GAMMA, output, PARTS, False, False, GFLAGS, 1.0, FP64)
        g = dy * gamma
        norm_gradient = r * g - r3 * coefficient * value * a
        d1 = _activation_d1(xg, code, scale)
        jacobian = tl.where(
            role == 0,
            _activation_d1(xs, code, scale),
            tl.where(role == 1, activation, xs * d1),
        )
        if SECOND:
            us = _read_direction(U, row, source, valid, DIN, HAS_U)
            ug = _read_direction(
                U, row, tl.maximum(gate, 0), valid & (gate >= 0), DIN, HAS_U
            )
            t = tl.where(
                gate >= 0,
                us * activation + xs * d1 * ug,
                _activation_d1(xs, code, scale) * us,
            )
            h = _parameters(H, output, PARTS, False, False, HFLAGS, 0.0, FP64)
            norm_double = (
                r * dy * h
                - coefficient * value * r3 * c
                + 3.0 * coefficient * value * r3 * r * r * a * s
                - r3 * (s * g + a * coefficient * t)
            )
            curvature = tl.where(
                role == 0,
                _activation_d2(xs, code, scale) * us,
                tl.where(
                    role == 1,
                    d1 * ug,
                    d1 * us + xs * _activation_d2(xg, code, scale) * ug,
                ),
            )
            contribution = norm_double * jacobian + norm_gradient * curvature
        else:
            contribution = norm_gradient * jacobian
        result += tl.where(valid, contribution, 0.0)
    return result


@triton.jit
def backward_direct(
    P,
    S,
    DY,
    SRC,
    GATE,
    ACT,
    SCALE,
    IN_OUT,
    IN_ROLE,
    COEF,
    GAMMA,
    DZ,
    PARTIALS,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_FAN: tl.constexpr,
    PARTS: tl.constexpr,
    GFLAGS: tl.constexpr,
    GOFFSETS: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    FP64: tl.constexpr,
    DZFLAG: tl.constexpr,
    GGRADFLAGS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ids = tl.arange(0, BLOCK)
    value = _gate(P, S, SRC, GATE, ACT, SCALE, row, ids, DIN, DOUT)
    dy = tl.load(DY + row * DOUT + ids, ids < DOUT, 0.0)
    coefficient = tl.load(COEF + ids, ids < DOUT, 0.0)
    variance = _moment(
        value[:, None] * value[:, None], coefficient, 1.0, 1, "components_first", False
    )
    r = tl.rsqrt(variance + EPS)
    gamma = _parameters(GAMMA, ids, PARTS, False, False, GFLAGS, 1.0, FP64)
    a = tl.sum(dy * gamma * value, 0)
    _selected_gamma_partials(
        dy * value * r, PARTIALS, row, WIDTH, PARTS, GOFFSETS, False, GGRADFLAGS
    )
    if DZFLAG:
        dz = _gate_transpose_direct(
            P,
            S,
            DY,
            P,
            GAMMA,
            SRC,
            GATE,
            ACT,
            SCALE,
            IN_OUT,
            IN_ROLE,
            COEF,
            GAMMA,
            row,
            ids,
            r,
            a,
            0.0,
            0.0,
            DIN,
            DOUT,
            PARTS,
            GFLAGS,
            GFLAGS,
            FP64,
            MAX_FAN,
            False,
            False,
        )
        tl.store(DZ + row * DIN + ids, dz, ids < DIN)


@triton.jit
def double_backward_direct(
    P,
    S,
    DY,
    U,
    H,
    SRC,
    GATE,
    ACT,
    SCALE,
    IN_OUT,
    IN_ROLE,
    COEF,
    GAMMA,
    DDZ,
    DDY,
    PARTIALS,
    DIN: tl.constexpr,
    DOUT: tl.constexpr,
    BLOCK: tl.constexpr,
    MAX_FAN: tl.constexpr,
    PARTS: tl.constexpr,
    GFLAGS: tl.constexpr,
    HFLAGS: tl.constexpr,
    HAS_U: tl.constexpr,
    GOFFSETS: tl.constexpr,
    WIDTH: tl.constexpr,
    EPS: tl.constexpr,
    FP64: tl.constexpr,
    DDZFLAG: tl.constexpr,
    DDYFLAG: tl.constexpr,
    GGRADFLAGS: tl.constexpr,
):
    row = tl.program_id(0).to(tl.int64)
    ids = tl.arange(0, BLOCK)
    value = _gate(P, S, SRC, GATE, ACT, SCALE, row, ids, DIN, DOUT)
    t = _gate_tangent(P, S, U, SRC, GATE, ACT, SCALE, row, ids, DIN, DOUT, HAS_U)
    dy = tl.load(DY + row * DOUT + ids, ids < DOUT, 0.0)
    coefficient = tl.load(COEF + ids, ids < DOUT, 0.0)
    variance = _moment(
        value[:, None] * value[:, None], coefficient, 1.0, 1, "components_first", False
    )
    r = tl.rsqrt(variance + EPS)
    gamma = _parameters(GAMMA, ids, PARTS, False, False, GFLAGS, 1.0, FP64)
    h = _parameters(H, ids, PARTS, False, False, HFLAGS, 0.0, FP64)
    g = dy * gamma
    a = tl.sum(g * value, 0)
    s = tl.sum(t * coefficient * value, 0)
    c = tl.sum(t * g + dy * h * value, 0)
    r3 = r * r * r
    if DDYFLAG:
        ddy = r * (t * gamma + h * value) - r3 * s * value * gamma
        tl.store(DDY + row * DOUT + ids, ddy, ids < DOUT)
    _selected_gamma_partials(
        dy * (r * t - r3 * s * value),
        PARTIALS,
        row,
        WIDTH,
        PARTS,
        GOFFSETS,
        False,
        GGRADFLAGS,
    )
    if DDZFLAG:
        ddz = _gate_transpose_direct(
            P,
            S,
            DY,
            U,
            H,
            SRC,
            GATE,
            ACT,
            SCALE,
            IN_OUT,
            IN_ROLE,
            COEF,
            GAMMA,
            row,
            ids,
            r,
            a,
            s,
            c,
            DIN,
            DOUT,
            PARTS,
            GFLAGS,
            HFLAGS,
            FP64,
            MAX_FAN,
            HAS_U,
            True,
        )
        tl.store(DDZ + row * DIN + ids, ddz, ids < DIN)
