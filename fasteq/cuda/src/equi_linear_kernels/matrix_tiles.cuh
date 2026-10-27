// FastEq shared matrix tiles. Dense and continuous-Y FCTP readers use the
// same SIMT/MMA math loop and the original Linear shared swizzle/cp.async.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
#include "ptx_inst.cuh"
namespace fasteq_jit { namespace matrix_device {
struct MatrixShape {
    int64_t m, n, k;
    int64_t ar, ac, ab, br, bc, bb, cr, cc, cb;
};

#include "matrix_tf32.cuh"

// Original FP64 A swizzle: four-double groups XOR the row's low two bits.
__device__ __forceinline__ int a_swizzle(int row, int k) {
    return row * 16 + (((k >> 2) ^ (row & 3)) << 2) + (k & 3);
}
// Original FP64 B swizzle, generalized from WarpBTileG2SSwizzleAsync:
// each 16-column region is four shared rows; the four-column groups XOR K.
template<int TN>
__device__ __forceinline__ int b_swizzle(int k, int col) {
    const int y = (k >> 2) * (TN / 4) + (col >> 4) * 4 + (k & 3);
    const int x = (((col & 15) >> 2) ^ (k & 3)) * 4 + (col & 3);
    return y * 16 + x;
}

template<int TM, bool AK>
__device__ __forceinline__ int mma_a_index(int row, int k) {
    if constexpr (AK) return a_swizzle(row, k);
    else return b_swizzle<TM>(k, row);
}

template<int TN, bool BN>
__device__ __forceinline__ int mma_b_index(int k, int col) {
    if constexpr (BN) return b_swizzle<TN>(k, col);
    else return a_swizzle(col, k);
}

template<typename T, int TM, int TN, int TK, bool AK, bool BN, bool MMA>
struct DenseAReader {
    static constexpr bool FlatGrid=false;
    struct Storage { T values[MMA ? TM * 16 : TM * (TK + 1)]; };
    const T* a;
    __device__ __forceinline__ void shift(MatrixShape s, int64_t batch, int64_t offset) {
        a += batch * s.ab + offset * s.ac;
    }
    __device__ __forceinline__ void load(Storage& tile, MatrixShape s,
                                       int64_t m0, int64_t k0) const {
        T* sa = tile.values;
        if constexpr (MMA) {

    if constexpr(sizeof(T)==8) {
    // The contiguous two-double load uses FastEq's existing cp.async macro.
    // Predicate failures explicitly write zero, so every tail has defined data.
    if constexpr (AK) {
        for (int pair = threadIdx.x; pair < TM * 8; pair += blockDim.x) {
            const int row = pair / 8, k = (pair & 7) * 2;
            double* dst = sa + a_swizzle(row, k);
            const double* src = a + (m0 + row) * s.ar + (k0 + k);
            if (m0 + row < s.m && k0 + k + 1 < s.k && !(reinterpret_cast<uintptr_t>(src) & 15)) {
                const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm_cp_async_ca_l2_prefetch_128B(address, src, 16);
            } else {
                dst[0] = m0 + row < s.m && k0 + k < s.k ? src[0] : 0;
                dst[1] = m0 + row < s.m && k0 + k + 1 < s.k ? src[1] : 0;
            }
        }
    } else {
        // A transpose reuses the original B swizzle, so consecutive rows
        // remain a 16-byte global/shared copy instead of scalar gathers.
        for (int pair = threadIdx.x; pair < TM * 8; pair += blockDim.x) {
            const int k = pair / (TM / 2), row = (pair % (TM / 2)) * 2;
            double* dst = sa + b_swizzle<TM>(k, row);
            const double* src = a + (m0 + row) * s.ar + (k0 + k) * s.ac;
            if (s.ar == 1 && m0 + row + 1 < s.m && k0 + k < s.k && !(reinterpret_cast<uintptr_t>(src) & 15)) {
                const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm_cp_async_ca_l2_prefetch_128B(address, src, 16);
            } else {
                dst[0] = m0 + row < s.m && k0 + k < s.k ? src[0] : 0;
                dst[1] = m0 + row + 1 < s.m && k0 + k < s.k ? src[s.ar] : 0;
            }
        }
    }
    } else {
    // FP32 keeps all original mantissa bits in shared memory. Four adjacent
    // floats use the existing 16-byte cp.async and the original MMA swizzle.
    if constexpr(AK) {
        for(int group=threadIdx.x;group<TM*4;group+=blockDim.x) {
            const int row=group/4,k=(group&3)*4;
            float* dst=sa+a_swizzle(row,k);const float* src=a+(m0+row)*s.ar+(k0+k);
            if(m0+row<s.m&&k0+k+3<s.k&&!(reinterpret_cast<uintptr_t>(src)&15))
                asm_cp_async_ca_l2_prefetch_128B((unsigned)__cvta_generic_to_shared(dst),src,16);
            else {
                #pragma unroll
                for(int v=0;v<4;++v)dst[v]=m0+row<s.m&&k0+k+v<s.k?src[v]:0.f;
            }
        }
    } else {
        for(int group=threadIdx.x;group<16*(TM/4);group+=blockDim.x) {
            const int k=group/(TM/4),row=(group%(TM/4))*4;
            float* dst=sa+b_swizzle<TM>(k,row);const float* src=a+(m0+row)*s.ar+(k0+k)*s.ac;
            if(s.ar==1&&m0+row+3<s.m&&k0+k<s.k&&!(reinterpret_cast<uintptr_t>(src)&15))
                asm_cp_async_ca_l2_prefetch_128B((unsigned)__cvta_generic_to_shared(dst),src,16);
            else {
                #pragma unroll
                for(int v=0;v<4;++v)dst[v]=m0+row+v<s.m&&k0+k<s.k?src[v*s.ar]:0.f;
            }
        }
    }
    }

        } else {
            for (int cell=threadIdx.x;cell<TM*TK;cell+=blockDim.x) {
                const int row=AK?cell/TK:cell%TM,k=AK?cell%TK:cell/TM;
                T* dst=sa+row*(TK+1)+k;
                if(m0+row<s.m&&k0+k<s.k) {
                    const T* src=a+(m0+row)*s.ar+(k0+k)*s.ac;
                    const unsigned address=(unsigned)__cvta_generic_to_shared(dst);
                    asm_cp_async_ca(address,src,4);
                } else *dst=0;
            }
        }
    }
    __device__ __forceinline__ T read(const Storage& tile, int row, int k,
                                     int64_t, int64_t) const {
        if constexpr(MMA) return tile.values[mma_a_index<TM,AK>(row,k)];
        else return tile.values[row*(TK+1)+k];
    }
};

template<typename T,int TM,int TN,int TK,bool AK,bool BN,bool MMA>
__device__ __forceinline__ void load_dense_b(T* sb,const T* b,MatrixShape s,
                                           int64_t n0,int64_t k0) {
    if constexpr(MMA) {
    if constexpr(sizeof(T)==8) {
    if constexpr (BN) {
        for (int pair = threadIdx.x; pair < 8 * TN; pair += blockDim.x) {
            const int k = pair / (TN / 2), col = (pair % (TN / 2)) * 2;
            double* dst = sb + b_swizzle<TN>(k, col);
            const double* src = b + (k0 + k) * s.br + n0 + col;
            if (k0 + k < s.k && n0 + col + 1 < s.n && !(reinterpret_cast<uintptr_t>(src) & 15)) {
                const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm_cp_async_ca_l2_prefetch_256B(address, src, 16);
            } else {
                dst[0] = k0 + k < s.k && n0 + col < s.n ? src[0] : 0;
                dst[1] = k0 + k < s.k && n0 + col + 1 < s.n ? src[1] : 0;
            }
        }
    } else {
        // Symmetrically, a B transpose reuses the original A swizzle.
        for (int pair = threadIdx.x; pair < TN * 8; pair += blockDim.x) {
            const int col = pair / 8, k = (pair & 7) * 2;
            double* dst = sb + a_swizzle(col, k);
            const double* src = b + (k0 + k) * s.br + (n0 + col) * s.bc;
            if (s.br == 1 && k0 + k + 1 < s.k && n0 + col < s.n && !(reinterpret_cast<uintptr_t>(src) & 15)) {
                const unsigned address = static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm_cp_async_ca_l2_prefetch_256B(address, src, 16);
            } else {
                dst[0] = k0 + k < s.k && n0 + col < s.n ? src[0] : 0;
                dst[1] = k0 + k + 1 < s.k && n0 + col < s.n ? src[s.br] : 0;
            }
        }
    }
    } else {
    if constexpr(BN) {
        for(int group=threadIdx.x;group<4*TN;group+=blockDim.x) {
            const int k=group/(TN/4),col=(group%(TN/4))*4;
            float* dst=sb+b_swizzle<TN>(k,col);const float* src=b+(k0+k)*s.br+n0+col;
            if(k0+k<s.k&&n0+col+3<s.n&&!(reinterpret_cast<uintptr_t>(src)&15))
                asm_cp_async_ca_l2_prefetch_256B((unsigned)__cvta_generic_to_shared(dst),src,16);
            else {
                #pragma unroll
                for(int v=0;v<4;++v)dst[v]=k0+k<s.k&&n0+col+v<s.n?src[v]:0.f;
            }
        }
    } else {
        for(int group=threadIdx.x;group<TN*4;group+=blockDim.x) {
            const int col=group/4,k=(group&3)*4;
            float* dst=sb+a_swizzle(col,k);const float* src=b+(k0+k)*s.br+(n0+col)*s.bc;
            if(s.br==1&&k0+k+3<s.k&&n0+col<s.n&&!(reinterpret_cast<uintptr_t>(src)&15))
                asm_cp_async_ca_l2_prefetch_256B((unsigned)__cvta_generic_to_shared(dst),src,16);
            else {
                #pragma unroll
                for(int v=0;v<4;++v)dst[v]=k0+k+v<s.k&&n0+col<s.n?src[v*s.br]:0.f;
            }
        }
    }
    }

    } else {
        for(int cell=threadIdx.x;cell<TK*TN;cell+=blockDim.x) {
            const int k=BN?cell/TN:cell%TK,col=BN?cell%TN:cell/TK;
            T* dst=sb+k*(TN+1)+col;
            if(k0+k<s.k&&n0+col<s.n) {
                const T* src=b+(k0+k)*s.br+(n0+col)*s.bc;
                const unsigned address=(unsigned)__cvta_generic_to_shared(dst);
                asm_cp_async_ca(address,src,4);
            } else *dst=0;
        }
    }
}

template<typename T,int TM,int TN,int TK,bool AK,bool BN,bool MMA,class Reader>
__device__ __forceinline__ void load_matrix_tile(
    Reader reader,typename Reader::Storage& sa,T* sb,const T* b,MatrixShape s,
    int64_t m0,int64_t n0,int64_t k0) {
    reader.load(sa,s,m0,k0);
    load_dense_b<T,TM,TN,TK,AK,BN,MMA>(sb,b,s,n0,k0);
    asm_cp_async_commit_group();
}

template<class T,int TM,int TN,bool AK,bool BN,class Reader>
__device__ __forceinline__ void matrix_mma_body_fp64_acc(
    Reader reader,const T* b,T* c,MatrixShape s,
    T alpha,T beta,int64_t tiles_m,int64_t split_span) {
    constexpr int WM = TM / 2, WN = TN / 2;
    __shared__ typename Reader::Storage sa[2];
    __shared__ T sb[2][TN * 16];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int wm = warp / 2, wn = warp % 2;
    const int64_t m0 = (Reader::FlatGrid?int64_t(blockIdx.x):blockIdx.y%tiles_m) * TM;
    const int64_t batch = Reader::FlatGrid?0:blockIdx.y/tiles_m, n0 = Reader::FlatGrid?0:int64_t(blockIdx.x)*TN;
    const int64_t split = Reader::FlatGrid?blockIdx.y:blockIdx.z;
    const int64_t batches = Reader::FlatGrid?1:gridDim.y/tiles_m;
    reader.shift(s,batch,split*split_span); b += batch*s.bb; c += batch*s.cb;
    if (split_span) {
        b += split * split_span * s.br;
        c += split * batches * s.m * s.n;
        const int64_t remaining = s.k - split * split_span;
        s.k = remaining < split_span ? remaining : split_span;
    }
    double acc[WM / 16][WN / 8][4] = {};
    if (s.k) {
        load_matrix_tile<T,TM,TN,16,AK,BN,true>(reader,sa[0],sb[0],b,s,m0,n0,0);
        asm_cp_async_waitgroup(0);
        __syncthreads();
        int stage = 0;
        for (int64_t k0 = 0; k0 < s.k; k0 += 16) {
            if (k0 + 16 < s.k)
                load_matrix_tile<T,TM,TN,16,AK,BN,true>(reader,sa[1-stage],sb[1-stage],b,s,m0,n0,k0+16);
            #pragma unroll
            for (int k = 0; k < 16; k += 4) {
                double bv[WN / 8];
                #pragma unroll
                for (int j = 0; j < WN / 8; ++j)
                    bv[j] = sb[stage][mma_b_index<TN, BN>(k + (lane & 3), wn * WN + j * 8 + lane / 4)];
                #pragma unroll
                for (int i = 0; i < WM / 16; ++i) {
                    const int r = wm * WM + i * 16 + lane / 4;
                    const double av0 = reader.read(sa[stage],r,k+(lane&3),m0,k0);
                    const double av1 = reader.read(sa[stage],r+8,k+(lane&3),m0,k0);
                    #pragma unroll
                    for (int j = 0; j < WN / 8; ++j)
                        asm_mma_m16n8k4_f64_f64_f64_f64(
                            acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3],
                            av0, av1, bv[j], acc[i][j][0], acc[i][j][1], acc[i][j][2], acc[i][j][3]);
                }
            }
            asm_cp_async_waitgroup(0);
            __syncthreads();
            stage = 1 - stage;
        }
    }
    if constexpr(sizeof(T)==8) {
    const bool vector_output = s.cc == 1 && !(s.cr & 1) && !(reinterpret_cast<uintptr_t>(c) & 15);
    #pragma unroll
    for (int i = 0; i < WM / 16; ++i) {
        #pragma unroll
        for (int j = 0; j < WN / 8; ++j) {
            #pragma unroll
            for (int q = 0; q < 2; ++q) {
                const int64_t row = m0 + wm * WM + i * 16 + lane / 4 + q * 8;
                const int64_t col = n0 + wn * WN + j * 8 + (lane & 3) * 2;
                if (row < s.m && col < s.n) {
                    double* dst = c + row * s.cr + col * s.cc;
                    if (vector_output && col + 1 < s.n) {
                        const double2 old = beta == 0 ? make_double2(0, 0) : *reinterpret_cast<const double2*>(dst);
                        *reinterpret_cast<double2*>(dst) = make_double2(
                            alpha * acc[i][j][q * 2] + beta * old.x,
                            alpha * acc[i][j][q * 2 + 1] + beta * old.y);
                    } else {
                        dst[0] = alpha * acc[i][j][q * 2] + (beta == 0 ? 0 : beta * dst[0]);
                        if (col + 1 < s.n)
                            dst[s.cc] = alpha * acc[i][j][q * 2 + 1] + (beta == 0 ? 0 : beta * dst[s.cc]);
                    }
                }
            }
        }
    }    } else {
    // FP32 inputs convert exactly to FP64 registers at the unchanged MMA
    // operands above. Accumulation remains FP64; round once at final output.
    #pragma unroll
    for(int i=0;i<WM/16;++i) {
        #pragma unroll
        for(int j=0;j<WN/8;++j) {
            #pragma unroll
            for(int q=0;q<2;++q) {
                const int64_t row=m0+wm*WM+i*16+lane/4+q*8;
                const int64_t col=n0+wn*WN+j*8+(lane&3)*2;
                if(row<s.m&&col<s.n) {
                    float* dst=c+row*s.cr+col*s.cc;
                    dst[0]=float(alpha*acc[i][j][q*2]+(beta==0?0:beta*double(dst[0])));
                    if(col+1<s.n)dst[s.cc]=float(alpha*acc[i][j][q*2+1]+(beta==0?0:beta*double(dst[s.cc])));
                }
            }
        }
    }
    }

}

template<int TM,int TN,bool AK,bool BN>
__global__ void matrix_mma_f64(
    const double* a,const double* b,double* c,MatrixShape s,
    double alpha,double beta,int64_t tiles_m,int64_t split_span) {
    DenseAReader<double,TM,TN,16,AK,BN,true> reader{a};
    matrix_mma_body_fp64_acc<double,TM,TN,AK,BN>(reader,b,c,s,alpha,beta,tiles_m,split_span);
}

// Exact FP32 input promotion reuses the same original FastEq FP64 MMA body.
template<int TM,int TN,bool AK,bool BN>
__global__ void matrix_mma_promoted_f32(
    const float* a,const float* b,float* c,MatrixShape s,
    float alpha,float beta,int64_t tiles_m,int64_t split_span) {
    DenseAReader<float,TM,TN,16,AK,BN,true> reader{a};
    matrix_mma_body_fp64_acc<float,TM,TN,AK,BN>(reader,b,c,s,alpha,beta,tiles_m,split_span);
}

template<int TM,int TN,int TK,bool AK,bool BN,class Reader>
__device__ __forceinline__ void matrix_simt_body_f32(
    Reader reader,const float* b,float* c,MatrixShape s,
    float alpha,float beta,int64_t tiles_m,int64_t split_span) {
    // Extends the 32x32 ROLE8--10 tile to 8 warps with two column groups.
    __shared__ typename Reader::Storage sa[2];
    __shared__ float sb[2][TK][TN + 1];
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int64_t m0 = (Reader::FlatGrid?int64_t(blockIdx.x):blockIdx.y%tiles_m) * TM;
    const int64_t batch = Reader::FlatGrid?0:blockIdx.y/tiles_m, n0 = Reader::FlatGrid?0:int64_t(blockIdx.x)*TN;
    const int64_t split = Reader::FlatGrid?blockIdx.y:blockIdx.z;
    const int64_t batches = Reader::FlatGrid?1:gridDim.y/tiles_m;
    reader.shift(s,batch,split*split_span); b += batch*s.bb; c += batch*s.cb;
    if (split_span) {
        b += split * split_span * s.br;
        c += split * batches * s.m * s.n;
        const int64_t remaining = s.k - split * split_span;
        s.k = remaining < split_span ? remaining : split_span;
    }
    float acc[TM / 8][TN / 32] = {};
    int stage=0;
    if(s.k){load_matrix_tile<float,TM,TN,TK,AK,BN,false>(reader,sa[0],&sb[0][0][0],b,s,m0,n0,0);asm_cp_async_waitgroup(0);__syncthreads();}
    for(int64_t k0=0;k0<s.k;k0+=TK){
        if(k0+TK<s.k)load_matrix_tile<float,TM,TN,TK,AK,BN,false>(reader,sa[1-stage],&sb[1-stage][0][0],b,s,m0,n0,k0+TK);
        #pragma unroll
        for (int k = 0; k < TK; ++k) {
            float bv[TN / 32];
            #pragma unroll
            for (int j = 0; j < TN / 32; ++j) bv[j] = sb[stage][k][lane + j * 32];
            #pragma unroll
            for (int i = 0; i < TM / 8; ++i) {
                const float av = reader.read(sa[stage],warp+i*8,k,m0,k0);
                #pragma unroll
                for (int j = 0; j < TN / 32; ++j) acc[i][j] = fmaf(av, bv[j], acc[i][j]);
            }
        }
        asm_cp_async_waitgroup(0);
        __syncthreads();stage=1-stage;
    }
    #pragma unroll
    for (int i = 0; i < TM / 8; ++i) {
        #pragma unroll
        for (int j = 0; j < TN / 32; ++j) {
            const int64_t row = m0 + warp + i * 8, col = n0 + lane + j * 32;
            if (row < s.m && col < s.n) {
                float* dst = c + row * s.cr + col * s.cc;
                *dst = alpha * acc[i][j] + (beta == 0 ? 0 : beta * *dst);
            }
        }
    }
}

template<int TM,int TN,int TK,bool AK,bool BN>
__global__ void matrix_simt_f32(
    const float* a,const float* b,float* c,MatrixShape s,
    float alpha,float beta,int64_t tiles_m,int64_t split_span) {
    DenseAReader<float,TM,TN,TK,AK,BN,false> reader{a};
    matrix_simt_body_f32<TM,TN,TK,AK,BN>(reader,b,c,s,alpha,beta,tiles_m,split_span);
}

}} // namespace fasteq_jit::matrix_device
