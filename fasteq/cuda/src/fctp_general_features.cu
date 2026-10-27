// General continuous-Y adjoints for FastEq's FCTP. The first measured fast
// domain is scalar C64; other sizes remain on the general FastEq CUDA plan.
// Reuses the original FCTP padded/vector shared loader, original Linear FP64
// MMA and asynchronous copies. It never materializes a rows x U x V tensor.
#include "fctp_shared.cuh"
#include "equi_linear_kernels/ptx_inst.cuh"
#include <ATen/ATen.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>
#include <type_traits>
#include <cstdlib>
#include <cstring>

namespace fasteq_jit {
namespace {
// On H100, exact FP32-to-FP64 register conversion can reduce shared-load/
// SIMT instructions while reusing the original FP64 Tensor Core primitive.
// It is explicitly opt-in, since consumer GPUs have different FP64 throughput.
template<typename T,bool DMMA=false> constexpr int feature_rows=std::is_same<T,double>::value?8:16;
template<typename T,bool DMMA=false> constexpr int feature_pitch=(std::is_same<T,double>::value||DMMA)?64:65;

template<typename T,bool DMMA=false>
__device__ __forceinline__ int weight_shared(int v,int w) {
    if constexpr(std::is_same<T,double>::value||DMMA) return v*64+(w^((v&7)*4));
    else return v*65+w;
}

template<typename T,int R>
__device__ __forceinline__ void feature_cache(T* cache,const T* input,long long start,long long rows) {
    for(int i=threadIdx.x;i<R*16;i+=blockDim.x) {
        int r=i/16,c=(i%16)*4;
        if(start+r<rows) fasteq_fctp::transpose_four_safe(cache,R+1,c,r,input+(start+r)*64+c);
        else {
            #pragma unroll
            for(int q=0;q<4;++q) cache[(c+q)*(R+1)+r]=0;
        }
    }
}

template<typename T,bool DMMA=false>
__device__ __forceinline__ void feature_weight_cache(T* cache,const T* weight) {
    if constexpr(std::is_same<T,double>::value) {
        if((reinterpret_cast<unsigned long long>(weight)&15)==0) {
            for(int pair=threadIdx.x;pair<2048;pair+=blockDim.x) {
                const int v=pair/32,w=(pair%32)*2;
                T* dst=cache+weight_shared<T,DMMA>(v,w);
                const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(dst));
                asm_cp_async_ca_l2_prefetch_128B(address,weight+v*64+w,16);
            }
        } else {
            for(int i=threadIdx.x;i<4096;i+=blockDim.x) {
                const int v=i/64,w=i%64;
                const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(cache+weight_shared<T,DMMA>(v,w)));
                asm_cp_async_ca(address,weight+i,8);
            }
        }
    } else {
        for(int i=threadIdx.x;i<4096;i+=blockDim.x) {
            const int v=i/64,w=i%64;
            const unsigned address=static_cast<unsigned>(__cvta_generic_to_shared(cache+weight_shared<T,DMMA>(v,w)));
            asm_cp_async_ca(address,weight+i,4);
        }
    }
    asm_cp_async_commit_group();
    asm_cp_async_waitgroup(0);
    __syncthreads();
}

template<typename T,int Mask,bool DMMA=false>
__global__ void fctp_feature_kernel(const T* weight,const T* x,const T* y,const T* dz,
                                    T* dx,T* dy,long long rows,T coefficient) {
    constexpr int R=feature_rows<T,DMMA>,P=R+1,WP=feature_pitch<T,DMMA>;
    extern __shared__ __align__(32) unsigned char raw[];
    T* zcache=reinterpret_cast<T*>(raw);
    T* xcache=zcache+64*P;
    T* ycache=xcache+64*P;
    T* wcache=ycache+64*P;
    T* xpartial=wcache+64*WP;
    const long long start=static_cast<long long>(blockIdx.x)*R;
    const int lane=threadIdx.x&31,warp=threadIdx.x>>5;
    feature_cache<T,R>(zcache,dz,start,rows);
    if constexpr(Mask&2) feature_cache<T,R>(xcache,x,start,rows);
    if constexpr(Mask&1) feature_cache<T,R>(ycache,y,start,rows);
    __syncthreads();
    if constexpr(std::is_same<T,double>::value||DMMA) {
        if constexpr(DMMA&&std::is_same<T,float>::value) {
        double ysum[2][4]={};
        const int r=lane/4,vbase=warp*16+lane/4;
        for(int u=0;u<64;++u) {
            feature_weight_cache<T,DMMA>(wcache,weight+u*4096);
            double product[2][4]={};
            #pragma unroll
            for(int k=0;k<64;k+=4) {
                const double a0=zcache[(k+(lane&3))*P+r];
                const double a1=zcache[(k+(lane&3))*P+r+8];
                #pragma unroll
                for(int j=0;j<2;++j) {
                    const double b=wcache[weight_shared<T,DMMA>(vbase+j*8,k+(lane&3))];
                    asm_mma_m16n8k4_f64_f64_f64_f64(product[j][0],product[j][1],product[j][2],product[j][3],
                        a0,a1,b,product[j][0],product[j][1],product[j][2],product[j][3]);
                }
            }
            if constexpr(Mask&1) {
                #pragma unroll
                for(int ri=0;ri<2;++ri) {
                    double sum=0;
                    #pragma unroll
                    for(int j=0;j<2;++j) {
                        #pragma unroll
                        for(int q=0;q<2;++q) {
                            const int v=warp*16+j*8+(lane&3)*2+q;
                            sum+=product[j][ri*2+q]*ycache[v*P+r+ri*8];
                        }
                    }
                    sum+=__shfl_down_sync(0xffffffff,sum,2,4);
                    sum+=__shfl_down_sync(0xffffffff,sum,1,4);
                    if((lane&3)==0) xpartial[(r+ri*8)*4+warp]=sum;
                }
                __syncthreads();
                if(threadIdx.x<R && start+threadIdx.x<rows) {
                    const int ri=threadIdx.x;
                    dx[(start+ri)*64+u]=coefficient*(xpartial[ri*4]+xpartial[ri*4+1]
                        +xpartial[ri*4+2]+xpartial[ri*4+3]);
                }
            }
            if constexpr(Mask&2) {
                const double a0=xcache[u*P+r],a1=xcache[u*P+r+8];
                #pragma unroll
                for(int j=0;j<2;++j) {
                    #pragma unroll
                    for(int q=0;q<2;++q) {
                        ysum[j][q]+=a0*product[j][q];
                        ysum[j][q+2]+=a1*product[j][q+2];
                    }
                }
            }
            __syncthreads();
        }
        if constexpr(Mask&2) {
            #pragma unroll
            for(int ri=0;ri<2;++ri) {
                if(start+r+ri*8<rows) {
                    #pragma unroll
                    for(int j=0;j<2;++j) {
                        #pragma unroll
                        for(int q=0;q<2;++q) {
                            const int v=warp*16+j*8+(lane&3)*2+q;
                            dy[(start+r+ri*8)*64+v]=coefficient*ysum[j][ri*2+q];
                        }
                    }
                }
            }
        }
        } else {
        double ysum[2][2]={};
        const int r=lane/4,vbase=warp*16+lane/4;
        for(int u=0;u<64;++u) {
            feature_weight_cache<T,DMMA>(wcache,weight+u*4096);
            double product[2][2]={};
            #pragma unroll
            for(int k=0;k<64;k+=4) {
                const double a=zcache[(k+(lane&3))*P+r];
                #pragma unroll
                for(int j=0;j<2;++j) {
                    const double b=wcache[weight_shared<T,DMMA>(vbase+j*8,k+(lane&3))];
                    asm_mma_m8n8k4_f64_f64_f64_f64(product[j][0],product[j][1],a,b,
                                                 product[j][0],product[j][1]);
                }
            }
            if constexpr(Mask&1) {
                double sum=0;
                #pragma unroll
                for(int j=0;j<2;++j) {
                    #pragma unroll
                    for(int q=0;q<2;++q) {
                        const int v=warp*16+j*8+(lane&3)*2+q;
                        sum+=product[j][q]*ycache[v*P+r];
                    }
                }
                sum+=__shfl_down_sync(0xffffffff,sum,2,4);
                sum+=__shfl_down_sync(0xffffffff,sum,1,4);
                if((lane&3)==0) xpartial[r*4+warp]=sum;
                __syncthreads();
                if(threadIdx.x<R && start+threadIdx.x<rows) {
                    const int ri=threadIdx.x;
                    dx[(start+ri)*64+u]=coefficient*(xpartial[ri*4]+xpartial[ri*4+1]
                        +xpartial[ri*4+2]+xpartial[ri*4+3]);
                }
            }
            if constexpr(Mask&2) {
                const double a=xcache[u*P+r];
                #pragma unroll
                for(int j=0;j<2;++j) {
                    #pragma unroll
                    for(int q=0;q<2;++q) ysum[j][q]+=a*product[j][q];
                }
            }
            __syncthreads(); // Every warp finishes before the next weight load.
        }
        if constexpr(Mask&2) {
            if(start+r<rows) {
                #pragma unroll
                for(int j=0;j<2;++j) {
                    #pragma unroll
                    for(int q=0;q<2;++q) {
                        const int v=warp*16+j*8+(lane&3)*2+q;
                        dy[(start+r)*64+v]=coefficient*ysum[j][q];
                    }
                }
            }
        }
        }
    } else {
        float ysum[8]={};
        const int v=warp*16+(lane&15),rbase=lane/16;
        for(int u=0;u<64;++u) {
            feature_weight_cache<T,DMMA>(wcache,weight+u*4096);
            float product[8]={};
            #pragma unroll
            for(int k=0;k<64;++k) {
                const float b=wcache[weight_shared<T,DMMA>(v,k)];
                #pragma unroll
                for(int i=0;i<8;++i) product[i]+=zcache[k*P+rbase+2*i]*b;
            }
            if constexpr(Mask&1) {
                #pragma unroll
                for(int i=0;i<8;++i) {
                    const int r=rbase+2*i;
                    float sum=product[i]*ycache[v*P+r];
                    sum+=__shfl_down_sync(0xffffffff,sum,8,16);
                    sum+=__shfl_down_sync(0xffffffff,sum,4,16);
                    sum+=__shfl_down_sync(0xffffffff,sum,2,16);
                    sum+=__shfl_down_sync(0xffffffff,sum,1,16);
                    if((lane&15)==0) xpartial[r*4+warp]=sum;
                }
                __syncthreads();
                if(threadIdx.x<R && start+threadIdx.x<rows) {
                    const int r=threadIdx.x;
                    dx[(start+r)*64+u]=coefficient*(xpartial[r*4]+xpartial[r*4+1]
                        +xpartial[r*4+2]+xpartial[r*4+3]);
                }
            }
            if constexpr(Mask&2) {
                #pragma unroll
                for(int i=0;i<8;++i) ysum[i]+=xcache[u*P+rbase+2*i]*product[i];
            }
            __syncthreads();
        }
        if constexpr(Mask&2) {
            #pragma unroll
            for(int i=0;i<8;++i) {
                const int r=rbase+2*i;
                if(start+r<rows) dy[(start+r)*64+v]=coefficient*ysum[i];
            }
        }
    }
}

template<typename T,int Mask,bool DMMA=false>
void launch_features(at::Tensor& dx,at::Tensor& dy,const at::Tensor& w,
                     const at::Tensor& x,const at::Tensor& y,const at::Tensor& dz,double coefficient) {
    constexpr int R=feature_rows<T,DMMA>;
    constexpr int shared=(3*64*(R+1)+64*feature_pitch<T,DMMA>+R*4)*sizeof(T);
    static_assert(shared<=48*1024,"FCTP feature tile exceeds the portable shared budget");
    const auto stream=c10::cuda::getCurrentCUDAStream(dz.get_device());
    fctp_feature_kernel<T,Mask,DMMA><<<(dz.size(0)+R-1)/R,128,shared,stream>>>(
        w.const_data_ptr<T>(),x.const_data_ptr<T>(),y.const_data_ptr<T>(),dz.const_data_ptr<T>(),
        dx.mutable_data_ptr<T>(),dy.mutable_data_ptr<T>(),dz.size(0),static_cast<T>(coefficient));
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
} // namespace

void fctp_features_into(at::Tensor& dx,at::Tensor& dy,const at::Tensor& w,
                        const at::Tensor& x,const at::Tensor& y,const at::Tensor& dz,
                        double coefficient,int mask) {
    TORCH_CHECK(mask>=1&&mask<=3,"FCTP feature output mask must be 1,2,3");
    TORCH_CHECK(dz.is_cuda()&&dz.dim()==2&&dz.size(1)==64&&dz.is_contiguous(),"FCTP DZ layout");
    TORCH_CHECK(w.is_cuda()&&w.is_contiguous()&&w.numel()==64*64*64,"FCTP shared W layout");
    auto valid=[&](const at::Tensor& t) {
        return t.device()==dz.device()&&t.scalar_type()==dz.scalar_type()&&t.is_contiguous();
    };
    TORCH_CHECK(valid(w)&&valid(x)&&valid(y)&&valid(dx)&&valid(dy),"FCTP operands must share device/dtype and packed storage");
    TORCH_CHECK(!(mask&1)||(y.sizes()==dz.sizes()&&dx.sizes()==dz.sizes()),"FCTP DX dimensions");
    TORCH_CHECK(!(mask&2)||(x.sizes()==dz.sizes()&&dy.sizes()==dz.sizes()),"FCTP DY dimensions");
    if(!dz.size(0))return;
    const c10::cuda::CUDAGuard guard(dz.device());
    if(dz.scalar_type()==at::kDouble) {
        if(mask==1)launch_features<double,1>(dx,dy,w,x,y,dz,coefficient);
        else if(mask==2)launch_features<double,2>(dx,dy,w,x,y,dz,coefficient);
        else launch_features<double,3>(dx,dy,w,x,y,dz,coefficient);
    } else {
        TORCH_CHECK(dz.scalar_type()==at::kFloat,"FCTP features support FP32/FP64");
        const char* option=std::getenv("FASTEQ_EXPERIMENTAL_FCTP_F32_F64MMA");
        if(option&&std::strcmp(option,"1")==0) {
            if(mask==1)launch_features<float,1,true>(dx,dy,w,x,y,dz,coefficient);
            else if(mask==2)launch_features<float,2,true>(dx,dy,w,x,y,dz,coefficient);
            else launch_features<float,3,true>(dx,dy,w,x,y,dz,coefficient);
            return;
        }
        if(mask==1)launch_features<float,1>(dx,dy,w,x,y,dz,coefficient);
        else if(mask==2)launch_features<float,2>(dx,dy,w,x,y,dz,coefficient);
        else launch_features<float,3>(dx,dy,w,x,y,dz,coefficient);
    }
}
} // namespace fasteq_jit
