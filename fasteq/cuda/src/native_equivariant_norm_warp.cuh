#pragma once
// FastEq native forward Norm extension of its own formulas and warp reduction.
// Each complete warp owns one row; four warps in a CTA are independent.
// Uniform tail warps return before all shuffle math. There are no CTA barriers.
using NormT=FASTEQ_NORM_TYPE;
using NormIndex=long long;
constexpr int NormD=FASTEQ_NORM_DIM;
constexpr int NormItems=FASTEQ_NORM_BLOCK/32;

__device__ __forceinline__ NormT norm_add(NormT a,NormT b) {
#if FASTEQ_NORM_FP64
    return __dadd_rn(a,b);
#else
    return __fadd_rn(a,b);
#endif
}
__device__ __forceinline__ NormT norm_mul(NormT a,NormT b) {
#if FASTEQ_NORM_FP64
    return __dmul_rn(a,b);
#else
    return __fmul_rn(a,b);
#endif
}
// The step is a compile-time constant. This preserves the same ordered
// power-of-two additions while keeping the per-lane tree in registers.
// A run-time outer loop left the 32-item arrays in local stack memory.
template<int Distance>
__device__ __forceinline__ void norm_tree_stage(NormT (&values)[NormItems]) {
    #pragma unroll
    for(int item=0;item<Distance;++item)
        values[item]=norm_add(values[item],values[item+Distance]);
    if constexpr(Distance>1)norm_tree_stage<Distance/2>(values);
}
__device__ __forceinline__ NormT norm_local_tree(NormT (&values)[NormItems]) {
    if constexpr(NormItems>1)norm_tree_stage<NormItems/2>(values);
    return values[0];
}
__device__ __forceinline__ NormT norm_warp_sum(NormT (&values)[NormItems]) {
    const NormT total=warp_reduce_sum(norm_local_tree(values));
    return __shfl_sync(FULL_MASK,total,0);
}

extern "C" __global__ void FASTEQ_NORM_ENTRY(FASTEQ_NORM_SIGNATURE,NormIndex rows) {
    const int lane=threadIdx.x&31;
    const NormIndex row=NormIndex(blockIdx.x)*4+(threadIdx.x>>5);
    if(row>=rows)return;
    NormT value[NormItems];
    #pragma unroll
    for(int item=0;item<NormItems;++item) {
        const int i=lane+item*32;
        value[item]=i<NormD?x[row*NormD+i]:NormT(0);
    }
    #pragma unroll
    for(int group=0;group<FASTEQ_NORM_CENTERS;++group) {
        NormT selected[NormItems];
        #pragma unroll
        for(int item=0;item<NormItems;++item) {
            const int i=lane+item*32;
            selected[item]=i>=fasteq_norm_center_start[group] &&
                i<fasteq_norm_center_start[group]+fasteq_norm_center_width[group]?value[item]:NormT(0);
        }
        const NormT mean=norm_warp_sum(selected)/NormT(fasteq_norm_center_width[group]);
        #pragma unroll
        for(int item=0;item<NormItems;++item) {
            const int i=lane+item*32;
            if(i>=fasteq_norm_center_start[group] && i<fasteq_norm_center_start[group]+fasteq_norm_center_width[group])
                value[item]=norm_add(value[item],-mean);
        }
    }
    NormT variance[NormItems];
    #pragma unroll
    for(int item=0;item<NormItems;++item) {
        const int i=lane+item*32;
        variance[item]=i<NormD?norm_mul(norm_mul(value[item],value[item]),coefficient[i]):NormT(0);
    }
    const NormT inverse=rsqrt(norm_add(norm_warp_sum(variance),NormT(FASTEQ_NORM_EPS)));
    #pragma unroll
    for(int item=0;item<NormItems;++item) {
        const int i=lane+item*32;
        if(i<NormD)y[row*NormD+i]=norm_add(norm_mul(norm_mul(value[item],inverse),FASTEQ_NORM_GAMMA(i)),FASTEQ_NORM_BETA(i));
    }
}
