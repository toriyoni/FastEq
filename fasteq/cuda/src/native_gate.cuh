#pragma once

// Native CUDA translation of FastEq's fused_equivariant_gate.py.
// Activation opcodes, normalized scales, input-centric deterministic fan-in,
// and all mixed second-derivative terms retain the original FastEq formulas.
// This header is compiled by FastEq's existing native JIT/runtime.
template <typename T>
__device__ __forceinline__ T gate_activation(T z, int code, T scale) {
    using GateT = T;
    if (code == 0) return scale * z;
    const GateT sigmoid = GateT(1) / (GateT(1) + exp(-z));
    GateT value = z;
    if (code == 1) value = z * sigmoid;
    else if (code == 2) value = sigmoid;
    else if (code == 3) value = GateT(2) / (GateT(1) + exp(-GateT(2) * z)) - GateT(1);
    return scale * value;
}

template <typename T>
__device__ __forceinline__ T gate_activation_d1(T z, int code, T scale) {
    using GateT = T;
    if (code == 0) return scale;
    const GateT sigmoid = GateT(1) / (GateT(1) + exp(-z));
    const GateT ds = sigmoid * (GateT(1) - sigmoid);
    GateT value = GateT(1);
    if (code == 1) value = sigmoid + z * ds;
    else if (code == 2) value = ds;
    else if (code == 3) {
        const GateT tanh = GateT(2) / (GateT(1) + exp(-GateT(2) * z)) - GateT(1);
        value = GateT(1) - tanh * tanh;
    }
    return scale * value;
}

template <typename T>
__device__ __forceinline__ T gate_activation_d2(T z, int code, T scale) {
    using GateT = T;
    if (code == 0) return GateT(0);
    const GateT sigmoid = GateT(1) / (GateT(1) + exp(-z));
    const GateT ds = sigmoid * (GateT(1) - sigmoid);
    GateT value = GateT(0);
    if (code == 1) value = ds * (GateT(2) + z * (GateT(1) - GateT(2) * sigmoid));
    else if (code == 2) value = ds * (GateT(1) - GateT(2) * sigmoid);
    else if (code == 3) {
        const GateT tanh = GateT(2) / (GateT(1) + exp(-GateT(2) * z)) - GateT(1);
        value = -GateT(2) * tanh * (GateT(1) - tanh * tanh);
    }
    return scale * value;
}

#ifndef FASTEQ_GATE_ACTIVATION_ONLY
using GateT = FASTEQ_GATE_TYPE;
using GateIndex = long long;
constexpr int GateIn = FASTEQ_GATE_IN;
constexpr int GateOut = FASTEQ_GATE_OUT;
constexpr int GateFan = FASTEQ_GATE_FAN;

extern "C" __global__ void native_gate_forward(
    const GateT* x, const int* src, const int* gate, const signed char* act,
    const GateT* scale, GateT* y, GateIndex rows) {
    const GateIndex i = GateIndex(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * GateOut) return;
    const GateIndex row = i / GateOut;
    const int output = i % GateOut, source = src[output], g = gate[output];
    const GateT value = x[row * GateIn + source];
    const GateT z = g < 0 ? value : x[row * GateIn + g];
    const GateT activated = gate_activation(z, act[output], scale[output]);
    y[i] = g < 0 ? activated : value * activated;
}

extern "C" __global__ void native_gate_backward(
    const GateT* x, const GateT* gy, const int* src, const int* gate,
    const signed char* act, const GateT* scale, const int* in_out,
    const signed char* in_role, GateT* dx, GateIndex rows) {
    const GateIndex i = GateIndex(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * GateIn) return;
    const GateIndex row = i / GateIn;
    const int input = i % GateIn;
    GateT total = x[i] * GateT(0);
    #pragma unroll
    for (int fan = 0; fan < GateFan; ++fan) {
        const int output = in_out[input * GateFan + fan];
        if (output < 0) continue;
        const int role = in_role[input * GateFan + fan];
        const int source = src[output], g = gate[output];
        const GateT z = x[row * GateIn + (g < 0 ? source : g)];
        const GateT gradient = gy[row * GateOut + output];
        GateT contribution;
        if (role == 1) contribution = gradient * gate_activation(z, act[output], scale[output]);
        else {
            contribution = gradient * gate_activation_d1(z, act[output], scale[output]);
            if (role == 2) contribution = gradient * x[row * GateIn + source] *
                gate_activation_d1(z, act[output], scale[output]);
        }
        total += contribution;
    }
    dx[i] = total;
}

// Compute both second-adjoint outputs in one launch when both are requested.
// Null output pointers prune the absent branch without reviving None seeds.
extern "C" __global__ void native_gate_double(
    const GateT* x, const GateT* gy, const GateT* u, const int* src,
    const int* gate, const signed char* act, const GateT* scale,
    const int* in_out, const signed char* in_role, GateT* dx,
    GateT* dgy, GateIndex rows) {
    const GateIndex i = GateIndex(blockIdx.x) * blockDim.x + threadIdx.x;
    if (dx && i < rows * GateIn) {
        const GateIndex row = i / GateIn;
        const int input = i % GateIn;
        GateT total = x[i] * GateT(0);
        #pragma unroll
        for (int fan = 0; fan < GateFan; ++fan) {
            const int output = in_out[input * GateFan + fan];
            if (output < 0) continue;
            const int role = in_role[input * GateFan + fan];
            const int source = src[output], g = gate[output], gate_safe = g < 0 ? source : g;
            const GateT z = x[row * GateIn + gate_safe];
            const GateT source_value = x[row * GateIn + source];
            const GateT gradient = gy[row * GateOut + output];
            const GateT u_source = u[row * GateIn + source], u_gate = u[row * GateIn + gate_safe];
            const GateT d1 = gate_activation_d1(z, act[output], scale[output]);
            const GateT d2 = gate_activation_d2(z, act[output], scale[output]);
            GateT contribution = u[i] * gradient * d2;
            if (role == 1) contribution = u_gate * gradient * d1;
            else if (role == 2) contribution = gradient * (u_source * d1 + u_gate * source_value * d2);
            total += contribution;
        }
        dx[i] = total;
    }
    if (dgy && i < rows * GateOut) {
        const GateIndex row = i / GateOut;
        const int output = i % GateOut, source = src[output], g = gate[output];
        const int gate_safe = g < 0 ? source : g;
        const GateT z = x[row * GateIn + gate_safe];
        const GateT u_source = u[row * GateIn + source];
        const GateT d1 = gate_activation_d1(z, act[output], scale[output]);
        GateT value = u_source * d1;
        if (g >= 0) value = u_source * gate_activation(z, act[output], scale[output]) +
            u[row * GateIn + gate_safe] * x[row * GateIn + source] * d1;
        dgy[i] = value;
    }
}
#endif  // FASTEQ_GATE_ACTIVATION_ONLY
