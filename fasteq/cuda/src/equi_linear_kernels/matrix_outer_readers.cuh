// Continuous-Y FCTP extends FastEq's shared matrix tiles through A readers.
// The shared Y/X tiles replace the large materialized [edge,u,v] operand.
#pragma once
#include "matrix_tiles.cuh"

namespace fasteq_jit { namespace matrix_device {
template<class T>
__device__ __forceinline__ void copy_scalar(T* destination,const T* source,bool valid) {
    if(valid) {
        const unsigned address=(unsigned)__cvta_generic_to_shared(destination);
        if constexpr(sizeof(T)==8) asm_cp_async_ca(address,source,8);
        else asm_cp_async_ca(address,source,4);
    } else *destination=T(0);
}

template<class T,int TM,int TN,int TK,bool MMA>
struct OuterForwardAReader {
    static constexpr bool FlatGrid=true;
    using YReader=DenseAReader<T,TM,TN,TK,true,true,MMA>;
    struct __align__(16) Storage { typename YReader::Storage y; T x[TM]; };
    const T* x;
    const T* y;
    T coefficient;
    int64_t k_base=0;
    __device__ __forceinline__ void shift(MatrixShape,int64_t,int64_t offset) {
        k_base=offset;
    }
    __device__ __forceinline__ void load(Storage& tile,MatrixShape s,
                                        int64_t m0,int64_t k0) const {
        const int64_t uv=k_base+k0;
        const int u=uv>>6,v=uv&63;
        MatrixShape ys=s;
        ys.ar=64;ys.ac=1;ys.k=s.k-k0<TK?s.k-k0:TK;
        YReader reader{y+v};
        reader.load(tile.y,ys,m0,0);
        for(int row=threadIdx.x;row<TM;row+=blockDim.x)
            copy_scalar(tile.x+row,x+(m0+row)*64+u,m0+row<s.m&&u<64);
    }
    __device__ __forceinline__ T read(const Storage& tile,int row,int k,
                                     int64_t m0,int64_t k0) const {
        YReader reader{nullptr};
        return coefficient*(tile.x[row]*reader.read(tile.y,row,k,m0,k0));
    }
};

template<class T,int TM,int TN,int TK,bool MMA>
struct OuterTransposeAReader {
    static constexpr bool FlatGrid=true;
    static constexpr int TV=TM<64?TM:64;
    static constexpr int TU=TM<64?1:TM/64;
    struct __align__(16) Storage {
        T y[MMA?TK*TV:TK*(TV+1)];
        T x[TK*(TU+1)];
    };
    const T* x;
    const T* y;
    T coefficient;
    __device__ __forceinline__ void shift(MatrixShape,int64_t,int64_t offset) {
        x+=offset*64;y+=offset*64;
    }
    __device__ __forceinline__ void load(Storage& tile,MatrixShape s,
                                        int64_t m0,int64_t k0) const {
        MatrixShape ys=s;ys.br=64;ys.bc=1;ys.n=64;
        const int v0=TM<64?int(m0&63):0;
        load_dense_b<T,TM,TV,TK,true,true,MMA>(tile.y,y,ys,v0,k0);
        for(int cell=threadIdx.x;cell<TK*TU;cell+=blockDim.x) {
            const int k=cell/TU,u=cell%TU;
            copy_scalar(tile.x+k*(TU+1)+u,x+(k0+k)*64+(m0>>6)+u,
                        k0+k<s.k&&(m0>>6)+u<64);
        }
    }
    __device__ __forceinline__ T read(const Storage& tile,int row,int k,
                                     int64_t,int64_t) const {
        const int u=TM<64?0:row>>6,v=TM<64?row:row&63;
        const int index=MMA?b_swizzle<TV>(k,v):k*(TV+1)+v;
        return coefficient*(tile.x[k*(TU+1)+u]*tile.y[index]);
    }
};
}} // namespace fasteq_jit::matrix_device
