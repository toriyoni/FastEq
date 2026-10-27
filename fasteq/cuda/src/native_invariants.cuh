#pragma once

// FastEq irreps_invariants.py: each copy is sum(x*x)/angular_width,
// never sqrt(sum(x*x)). The injected original CUDA warp_reduce_sum is reused
// with a power-of-two subgroup for small angular widths and a full warp for
// larger copies. All lanes reach the shuffle, including incomplete tails.
using InvariantT = FASTEQ_INVARIANT_TYPE;
using InvariantIndex = long long;
constexpr int InvariantD = FASTEQ_INVARIANT_DIM;
constexpr int InvariantC = FASTEQ_INVARIANT_CHANNELS;
constexpr int InvariantLanes = FASTEQ_INVARIANT_LANES;

template <bool Direction>
__device__ __forceinline__ InvariantT invariant_copy_sum(
    const InvariantT* x, const InvariantT* u, const int* bases, const int* widths,
    InvariantIndex copy, InvariantIndex rows, bool& valid) {
    valid = copy < rows * InvariantC;
    const int lane = threadIdx.x % InvariantLanes;
    const int channel = valid ? copy % InvariantC : 0;
    const InvariantIndex row = valid ? copy / InvariantC : 0;
    const int width = valid ? widths[channel] : 0;
    const int base = valid ? bases[channel] : 0;
    InvariantT total = InvariantT(0);
    for (int angle = lane; angle < width; angle += InvariantLanes) {
        const InvariantIndex offset = row * InvariantD + base + angle;
        const InvariantT value = x[offset];
        total += value * (Direction ? u[offset] : value);
    }
    return warp_reduce_sum(total);
}

extern "C" __global__ void native_invariant_forward(
    const InvariantT* x, const int* bases, const int* widths,
    InvariantT* y, InvariantIndex rows) {
    const InvariantIndex thread = InvariantIndex(blockIdx.x) * blockDim.x + threadIdx.x;
    const InvariantIndex copy = thread / InvariantLanes;
    bool valid;
    const InvariantT value = invariant_copy_sum<false>(x, nullptr, bases, widths, copy, rows, valid);
    if (valid && threadIdx.x % InvariantLanes == 0)
        y[copy] = value / InvariantT(widths[copy % InvariantC]);
}

extern "C" __global__ void native_invariant_backward(
    const InvariantT* x, const InvariantT* g, const int* index,
    const int* widths, InvariantT* dx, InvariantIndex rows) {
    const InvariantIndex i = InvariantIndex(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= rows * InvariantD) return;
    const InvariantIndex row = i / InvariantD;
    const int channel = index[i % InvariantD];
    dx[i] = InvariantT(2) * x[i] * g[row * InvariantC + channel] / InvariantT(widths[channel]);
}

// The existing first-backward context needs dX and dG. Compute both in one
// CUDA launch; nullable output pointers implement the requested-output mask.
extern "C" __global__ void native_invariant_double(
    const InvariantT* x, const InvariantT* g, const InvariantT* u,
    const int* bases, const int* widths, const int* index,
    InvariantT* dx, InvariantT* dg, InvariantIndex rows) {
    const InvariantIndex i = InvariantIndex(blockIdx.x) * blockDim.x + threadIdx.x;
    if (dx && i < rows * InvariantD) {
        const InvariantIndex row = i / InvariantD;
        const int channel = index[i % InvariantD];
        dx[i] = InvariantT(2) * u[i] * g[row * InvariantC + channel] / InvariantT(widths[channel]);
    }
    if (dg) {
        const InvariantIndex copy = i / InvariantLanes;
        bool valid;
        const InvariantT value = invariant_copy_sum<true>(x, u, bases, widths, copy, rows, valid);
        if (valid && threadIdx.x % InvariantLanes == 0)
            dg[copy] = InvariantT(2) * value / InvariantT(widths[copy % InvariantC]);
    }
}
