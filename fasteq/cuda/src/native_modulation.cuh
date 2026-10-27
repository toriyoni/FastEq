#pragma once
// Native extension of FastEq Gate modulation/per-copy adjoints. The host and
// autograd plans retain their pointer order, masks and absent-seed semantics.
using ModT=FASTEQ_MOD_TYPE;
using ModIndex=long long;

__device__ __forceinline__ ModT mod_mul(ModT a,ModT b) {
#if FASTEQ_MOD_FP64
    return __dmul_rn(a,b);
#else
    return __fmul_rn(a,b);
#endif
}
__device__ __forceinline__ ModT mod_add(ModT a,ModT b) {
#if FASTEQ_MOD_FP64
    return __dadd_rn(a,b);
#else
    return __fadd_rn(a,b);
#endif
}

extern "C" __global__ void FASTEQ_MOD_ENTRY(FASTEQ_MOD_SIGNATURE) {
    const ModIndex row=blockIdx.x;
#if FASTEQ_MOD_NEED_G
    __shared__ ModT gate_partial[FASTEQ_MOD_DIM];
#endif
    for(int coordinate=threadIdx.x;coordinate<FASTEQ_MOD_DIM;coordinate+=blockDim.x) {
        const int channel=index[coordinate];
        const ModIndex feature_offset=row*FASTEQ_MOD_DIM+coordinate;
        const ModIndex gate_offset=row*FASTEQ_MOD_CHANNELS+channel;
#if FASTEQ_MOD_ROLE == 0
        output[feature_offset]=mod_add(mod_mul(x[feature_offset],g[gate_offset]),residual[feature_offset]);
#elif FASTEQ_MOD_ROLE == 1
        const ModT seed=dy[feature_offset];
        if(FASTEQ_MOD_NEED_X) dx[feature_offset]=mod_mul(seed,g[gate_offset]);
#if FASTEQ_MOD_NEED_G
        gate_partial[coordinate]=mod_mul(x[feature_offset],seed);
#endif
#else
        const ModT seed=dy[feature_offset];
        const ModT direction_x=FASTEQ_MOD_HAS_U?u[feature_offset]:ModT(0);
        const ModT direction_g=FASTEQ_MOD_HAS_V?v[gate_offset]:ModT(0);
        if(FASTEQ_MOD_NEED_X) dx[feature_offset]=mod_mul(seed,direction_g);
        if(FASTEQ_MOD_NEED_DY)
            ddy[feature_offset]=mod_add(mod_mul(direction_x,g[gate_offset]),mod_mul(direction_g,x[feature_offset]));
#if FASTEQ_MOD_NEED_G
        gate_partial[coordinate]=mod_mul(seed,direction_x);
#endif
#endif
    }
#if FASTEQ_MOD_NEED_G
    __syncthreads();
    // Norm's per-copy angular transpose: power-of-two padded ordered tree.
    // Copy metadata changes only the source/target offsets, not the reduction.
    fasteq_modulation_parameter_partials(gate_partial,dy,dg,row);
#endif
}
