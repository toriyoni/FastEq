// Extend FastEq's native Linear matrix loop to continuous-Y shared UVW FCTP.
// Dense and virtual X*Y use one common SIMT/MMA loop; no outer product buffer,
// cuBLAS, ATen matrix operator, CUTLASS, or external CUDA implementation.
#include "equi_linear_kernels/matrix_outer_readers.cuh"
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cstdlib>
#include <cstring>

namespace fasteq_jit {
namespace {
using matrix_device::MatrixShape;
template<class T,int TM,bool Weight,bool Promote=false>
__global__ void fctp_matrix_run(
    const T* x,const T* y,const T* b,T* c,MatrixShape s,T coefficient,
    int64_t tiles_m,int64_t split_span) {
    constexpr int TN=64,TK=16;
    if constexpr(Weight) {
        matrix_device::OuterTransposeAReader<T,TM,TN,TK,(sizeof(T)==8||Promote)> reader{x,y,coefficient};
        if constexpr(sizeof(T)==8||Promote)
            matrix_device::matrix_mma_body_fp64_acc<T,TM,TN,false,true>(reader,b,c,s,T(1),T(0),tiles_m,split_span);
        else matrix_device::matrix_simt_body_f32<TM,TN,TK,false,true>(reader,b,c,s,1.,0.,tiles_m,split_span);
    } else {
        matrix_device::OuterForwardAReader<T,TM,TN,TK,(sizeof(T)==8||Promote)> reader{x,y,coefficient};
        if constexpr(sizeof(T)==8||Promote)
            matrix_device::matrix_mma_body_fp64_acc<T,TM,TN,true,true>(reader,b,c,s,T(1),T(0),tiles_m,split_span);
        else matrix_device::matrix_simt_body_f32<TM,TN,TK,true,true>(reader,b,c,s,1.,0.,tiles_m,split_span);
    }
}

template<class T>
__global__ void fctp_partial_reduce(const T* partial,T* result,int64_t total,int splits) {
    for(int64_t i=blockIdx.x*blockDim.x+threadIdx.x;i<total;i+=gridDim.x*blockDim.x) {
        T value=0;
        for(int split=0;split<splits;++split)value+=partial[split*total+i];
        result[i]=value;
    }
}

void check_features(const at::Tensor& destination,const at::Tensor& x,
                    const at::Tensor& y,const at::Tensor& b,bool weight) {
    TORCH_CHECK(destination.is_cuda()&&x.device()==destination.device()
                &&y.device()==destination.device()&&b.device()==destination.device(),
                "FastEq continuous FCTP requires one CUDA device");
    const auto dtype=destination.scalar_type();
    TORCH_CHECK((dtype==at::kFloat||dtype==at::kDouble)&&x.scalar_type()==dtype
                &&y.scalar_type()==dtype&&b.scalar_type()==dtype,
                "FastEq continuous FCTP requires matching FP32/FP64 tensors");
    TORCH_CHECK(destination.dim()==2&&x.dim()==2&&y.dim()==2&&b.dim()==2,
                "FastEq continuous FCTP expects rank-two matrix views");
    TORCH_CHECK(x.size(1)==64&&y.sizes()==x.sizes()&&destination.size(1)==64
                &&x.is_contiguous()&&y.is_contiguous()&&b.is_contiguous()&&destination.is_contiguous(),
                "FastEq continuous FCTP fast path requires contiguous 64-channel features");
    TORCH_CHECK(destination.size(0)==(weight?4096:x.size(0))
                &&b.size(0)==(weight?x.size(0):4096)&&b.size(1)==64,
                "FastEq continuous FCTP matrix shape mismatch");
}

template<class T,int TM,bool Weight,bool Promote=false>
void launch_fctp(at::Tensor& destination,const at::Tensor& x,const at::Tensor& y,
                 const at::Tensor& b,double coefficient) {
    const int64_t rows=destination.size(0),columns=64;
    const int64_t reduction=Weight?x.size(0):4096;
    if(!rows)return;
    const int64_t tiles_m=(rows+TM-1)/TM;
    const int target=at::cuda::getDeviceProperties(destination.get_device())->multiProcessorCount*4;
    int splits=reduction>=1024&&tiles_m<target?int((target+tiles_m-1)/tiles_m):1;
    splits=splits>16?16:splits;
    const int64_t memory_limit=(64LL<<20)/(rows*columns*sizeof(T));
    splits=splits>memory_limit?int(memory_limit):splits;
    const int64_t depth_limit=(reduction+255)/256;
    splits=splits>depth_limit?int(depth_limit):splits;
    splits=splits<1?1:splits;
    const int64_t span=splits>1?((reduction+splits*16-1)/(splits*16))*16:0;
    at::Tensor result=splits>1?at::empty({splits,rows,columns},destination.options()):destination;
    MatrixShape s{rows,columns,reduction,0,0,0,64,1,0,64,1,rows*columns};
    const auto stream=c10::cuda::getCurrentCUDAStream(destination.get_device());
    TORCH_CHECK(tiles_m<=2147483647,"FastEq continuous FCTP row grid exceeds CUDA grid.x limit");
    const dim3 grid(tiles_m,splits,1);
    fctp_matrix_run<T,TM,Weight,Promote><<<grid,(sizeof(T)==8||Promote)?128:256,0,stream>>>(
        x.const_data_ptr<T>(),y.const_data_ptr<T>(),b.const_data_ptr<T>(),result.mutable_data_ptr<T>(),
        s,T(coefficient),tiles_m,span);
    if(splits>1) {
        const int64_t total=rows*columns;
        const unsigned blocks=unsigned((total+255)/256>4096?4096:(total+255)/256);
        fctp_partial_reduce<T><<<blocks,256,0,stream>>>(result.const_data_ptr<T>(),
            destination.mutable_data_ptr<T>(),total,splits);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

template<bool Weight>
void evaluate_fctp(at::Tensor& destination,const at::Tensor& x,const at::Tensor& y,
                   const at::Tensor& b,double coefficient) {
    check_features(destination,x,y,b,Weight);
    const c10::cuda::CUDAGuard guard(destination.device());
    if(destination.scalar_type()==at::kDouble)launch_fctp<double,32,Weight>(destination,x,y,b,coefficient);
    else {
        const char* option=std::getenv("FASTEQ_EXPERIMENTAL_MATRIX_F32_F64MMA");
        if(option&&std::strcmp(option,"1")==0)
            launch_fctp<float,128,Weight,true>(destination,x,y,b,coefficient);
        else launch_fctp<float,128,Weight>(destination,x,y,b,coefficient);
    }
}
} // namespace

void fctp_forward_into(at::Tensor& destination,const at::Tensor& x,const at::Tensor& y,
                       const at::Tensor& weight,double coefficient) {
    evaluate_fctp<false>(destination,x,y,weight,coefficient);
}
void fctp_weight_into(at::Tensor& destination,const at::Tensor& x,const at::Tensor& y,
                      const at::Tensor& dz,double coefficient) {
    evaluate_fctp<true>(destination,x,y,dz,coefficient);
}
} // namespace fasteq_jit
