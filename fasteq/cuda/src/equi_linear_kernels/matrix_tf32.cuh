#pragma once

// Extends FastEq's FP64 MMA tiling and original TF32 conversion primitive.
// FP32 is split into a rounded high value and rounded residual. Three MMA
// products retain high*high, low*high, high*low; low*low is omitted. This is
// a high-accuracy algorithm, not a bit-exact FP32 multiply. No external kernel
// source is used. Fragment mapping follows NVIDIA's PTX ISA m16n8k8 contract:
// docs.nvidia.com/cuda/archive/12.8.1/parallel-thread-execution/index.html

#include "matrix_tf32_primitives.cuh"

template<int TK>
__device__ __forceinline__ int matrix_tf32_a_index(int row, int k) {
    return row * TK + (k ^ ((row & 7) * 4));
}
template<int TN>
__device__ __forceinline__ int matrix_tf32_b_index(int k, int col) {
    return k * TN + (col ^ ((k & 3) * 8));
}

template<int TM, int TN, int TK, bool AK, bool BN>
__device__ __forceinline__ void matrix_tf32_body(
    const float* a, const float* b, float* c, MatrixShape s,
    float alpha, float beta, int64_t tiles_m, int64_t split_span) {
    static_assert(TK % 32 == 0 && TN % 32 == 0, "TF32 shared tile alignment");
    constexpr int WM = TM / 2, WN = TN / 2;
    __shared__ unsigned ah[TM * TK], al[TM * TK];
    __shared__ unsigned bh[TK * TN], bl[TK * TN];
    const int lane=threadIdx.x&31, warp=threadIdx.x>>5;
    const int wm=warp/2, wn=warp%2;
    const int64_t m0=(blockIdx.y%tiles_m)*TM;
    const int64_t batch=blockIdx.y/tiles_m,n0=blockIdx.x*TN;
    a+=batch*s.ab; b+=batch*s.bb; c+=batch*s.cb;
    if (split_span) {
        a+=blockIdx.z*split_span*s.ac;
        b+=blockIdx.z*split_span*s.br;
        c+=blockIdx.z*gridDim.y/tiles_m*s.m*s.n;
        const int64_t remaining=s.k-blockIdx.z*split_span;
        s.k=remaining<split_span?remaining:split_span;
    }
    float acc[WM/16][WN/8][4]={};
    float total_high[WM/16][WN/8][4]={};
    // Keep corrections separate: repeatedly adding small corrections into a
    // large HH accumulator would throw away part of the recovered mantissa.
    float correction[WM/16][WN/8][4]={};
    for (int64_t k0=0;k0<s.k;k0+=TK) {
        for (int cell=threadIdx.x;cell<TM*TK;cell+=blockDim.x) {
            const int r=AK?cell/TK:cell%TM,k=AK?cell%TK:cell/TM;
            const float value=m0+r<s.m&&k0+k<s.k?a[(m0+r)*s.ar+(k0+k)*s.ac]:0;
            const unsigned high=matrix_tf32_round(value);
            const int i=matrix_tf32_a_index<TK>(r,k);
            ah[i]=high; al[i]=matrix_tf32_round(value-__uint_as_float(high));
        }
        for (int cell=threadIdx.x;cell<TK*TN;cell+=blockDim.x) {
            const int k=BN?cell/TN:cell%TK,col=BN?cell%TN:cell/TK;
            const float value=k0+k<s.k&&n0+col<s.n?b[(k0+k)*s.br+(n0+col)*s.bc]:0;
            const unsigned high=matrix_tf32_round(value);
            const int i=matrix_tf32_b_index<TN>(k,col);
            bh[i]=high; bl[i]=matrix_tf32_round(value-__uint_as_float(high));
        }
        __syncthreads();
        #pragma unroll
        for (int k=0;k<TK;k+=8) {
            unsigned b_hi[WN/8][2],b_lo[WN/8][2];
            #pragma unroll
            for (int j=0;j<WN/8;++j) {
                const int col=wn*WN+j*8+lane/4;
                #pragma unroll
                for (int q=0;q<2;++q) {
                    const int i=matrix_tf32_b_index<TN>(k+(lane&3)+q*4,col);
                    b_hi[j][q]=bh[i]; b_lo[j][q]=bl[i];
                }
            }
            #pragma unroll
            for (int i=0;i<WM/16;++i) {
                unsigned a_hi[4],a_lo[4];
                const int row=wm*WM+i*16+lane/4;
                #pragma unroll
                for (int q=0;q<4;++q) {
                    const int t=matrix_tf32_a_index<TK>(row+(q&1)*8,k+(lane&3)+(q/2)*4);
                    a_hi[q]=ah[t]; a_lo[q]=al[t];
                }
                #pragma unroll
                for (int j=0;j<WN/8;++j) {
                    matrix_tf32_mma(correction[i][j],a_lo,b_hi[j]);
                    matrix_tf32_mma(correction[i][j],a_hi,b_lo[j]);
                    matrix_tf32_mma(acc[i][j],a_hi,b_hi[j]);
                }
            }
        }
        // Bound HH accumulation depth. Long, near-cancelling contractions
        // otherwise lose more FP32 mantissa than the residual restores.
        if ((k0+TK)%128==0 || k0+TK>=s.k) {
            #pragma unroll
            for (int i=0;i<WM/16;++i) {
                #pragma unroll
                for (int j=0;j<WN/8;++j) {
                    #pragma unroll
                    for (int q=0;q<4;++q) {
                        total_high[i][j][q]+=acc[i][j][q];
                        acc[i][j][q]=0;
                    }
                }
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int i=0;i<WM/16;++i) {
        #pragma unroll
        for (int j=0;j<WN/8;++j) {
            #pragma unroll
            for (int q=0;q<4;++q) {
                const int64_t row=m0+wm*WM+i*16+lane/4+(q/2)*8;
                const int64_t col=n0+wn*WN+j*8+(lane&3)*2+(q&1);
                if (row<s.m&&col<s.n) {
                    float* dst=c+row*s.cr+col*s.cc;
                    *dst=alpha*(total_high[i][j][q]+correction[i][j][q])+(beta==0?0:beta**dst);
                }
            }
        }
    }
}

template<int TM, int TN, int TK, bool AK, bool BN>
__global__ void matrix_tf32_f32(
    const float* a,const float* b,float* c,MatrixShape s,
    float alpha,float beta,int64_t tiles_m,int64_t split_span) {
    matrix_tf32_body<TM,TN,TK,AK,BN>(a,b,c,s,alpha,beta,tiles_m,split_span);
}
