// Adapter for the existing FastEq Hopper Linear pipeline. All matrix arithmetic
// is implemented in equi_linear_fwd_f32.cu; these kernels only change layout.
#include <torch/extension.h>
#include <mutex>
#include <unordered_set>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include "equi_linear_kernels/equi_linear_fwd_f32.cu"

struct LinearPackPath { int sx, sw, sy, ci, co, dim, angle; };
struct LinearPackPaths { LinearPackPath paths[32]; };

__global__ void pack_linear_input(const float* x, const float* w, float* px, float* pw,
    const int64_t* ids, LinearPackPaths meta, int64_t nodes, int din, int angles, int u, int v) {
    const auto p = meta.paths[blockIdx.y];
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t j=i; j < nodes * p.dim * u; j+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t i=j;
        const int c = i % u, a = (i / u) % p.dim;
        const int64_t n = i / (u * p.dim);
        px[(n * angles + p.angle + a) * u + c] = c < p.ci
            ? x[(ids ? ids[n] : n) * din + p.sx + c * p.dim + a] : 0.0f;
    }
    for (int64_t j=i; j < int64_t(u) * v; j+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t i=j;
        const int c = i / v;
        pw[(int64_t(blockIdx.y) * u * v) + i] = c < p.ci ? w[p.sw + i] : 0.0f;
    }
}

__global__ void unpack_linear_output(const float* py, float* y,
    const int64_t* ids, LinearPackPaths meta, int64_t nodes, int dout, int angles, int v) {
    const auto p = meta.paths[blockIdx.y];
    const int64_t i = int64_t(blockIdx.x) * blockDim.x + threadIdx.x;
    for (int64_t j=i; j < nodes * p.co * p.dim; j+=int64_t(gridDim.x)*blockDim.x) {
        const int64_t i=j;
        const int a = i % p.dim, c = (i / p.dim) % p.co;
        const int64_t n = i / (p.co * p.dim);
        y[(ids ? ids[n] : n) * dout + p.sy + c * p.dim + a] = py[(n * angles + p.angle + a) * v + c];
    }
}

// Resolve the module's entry points once per PyTorch primary device context.
// This also avoids a first-launch cuKernelGetFunction invalid-handle probe in
// CUDA 12.8 runtime stubs, which Compute Sanitizer reports as a host API error.
void prepare_linear_launchers(int device) {
    static std::mutex mutex;
    static std::unordered_set<int> ready;
    std::lock_guard<std::mutex> lock(mutex);
    if (ready.count(device)) return;
    C10_CUDA_CHECK(cudaFuncSetAttribute(pack_linear_input,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
    C10_CUDA_CHECK(cudaFuncSetAttribute(unpack_linear_output,
        cudaFuncAttributeMaxDynamicSharedMemorySize, 0));
    ready.insert(device);
}

at::Tensor linear_impl(const at::Tensor& x, const at::Tensor& w,
    std::vector<std::vector<int64_t>> paths, std::vector<double> coeff, int64_t dout,
    const at::Tensor& ids, at::Tensor out, int fusion) {
    TORCH_CHECK(x.is_cuda() && w.is_cuda() && x.device()==w.device() &&
        x.scalar_type()==at::kFloat && w.scalar_type()==at::kFloat,
        "WGMMA Linear requires CUDA FP32 features and weights on the same device");
    c10::cuda::CUDAGuard guard(x.device());
    prepare_linear_launchers(x.get_device());
    TORCH_CHECK(x.dim()==2 && x.is_contiguous() && w.is_contiguous(), "Contiguous inputs required");
    TORCH_CHECK(!paths.empty() && paths.size()<=16 && coeff.size()==paths.size(), "Invalid paths");
    TORCH_CHECK(fusion >= 0 && fusion <= 3, "Invalid Linear fusion mode");
    TORCH_CHECK(x.size(0)<INT_MAX && x.size(1)<INT_MAX && dout>0 && dout<INT_MAX, "Invalid dimensions");
    TORCH_CHECK(paths[0].size()==6,"Invalid path record");
    LinearPackPaths meta{}; LinearNativeAccess native{};
    int angles=0,u=0,v=paths[0][4],maxdim=0; int64_t ex=0,ew=0,ey=0;
    std::vector<int64_t> dims;
    for (size_t i=0;i<paths.size();++i) {
        TORCH_CHECK(paths[i].size()==6,"Invalid path record");
        auto p=paths[i];
        TORCH_CHECK(p[0]==ex && p[1]==ew && p[2]==ey && p[3]>0 && p[3]<=512 &&
            p[4]==v && p[5]>0 && p[5]<=31 && (dims.empty() || dims.back()!=p[5]),
            "WGMMA layout adapter requires contiguous one-to-one angular paths");
        meta.paths[i]={int(p[0]),int(p[1]),int(p[2]),int(p[3]),int(p[4]),int(p[5]),angles};
        native.paths[i]={int(p[0]),int(p[1]),int(p[2]),int(p[3]),int(p[5])};
        dims.push_back(p[5]); angles+=p[5];u=std::max(u,int(p[3]));maxdim=std::max(maxdim,int(p[5]));
        ex+=p[3]*p[5];ew+=p[3]*p[4];ey+=p[4]*p[5];
    }
    TORCH_CHECK(ex==x.size(1) && ew==w.numel() && ey==dout && v>0 && v%32==0,"Inconsistent Linear dimensions");
    u=(u+31)/32*32;
    const int64_t n=ids.defined() ? ids.numel() : x.size(0);
    if (ids.defined()) {
        TORCH_CHECK(ids.device()==x.device() && ids.scalar_type()==at::kLong &&
            ids.dim()==1 && ids.is_contiguous(),"Invalid group indices");
        TORCH_CHECK(out.device()==x.device() && out.scalar_type()==at::kFloat &&
            out.is_contiguous() && out.sizes()==at::IntArrayRef({x.size(0),dout}),"Invalid indexed output");
    } else out=at::empty({n,dout},x.options());
    if(n==0)return out;
    const auto index_ptr=ids.defined() ? ids.const_data_ptr<int64_t>() : nullptr;
    at::Tensor px, pw, py;
    if (!(fusion & 1)) {
        px=at::empty({n,angles,u},x.options());
        pw=at::empty({int64_t(paths.size()),u,v},w.options());
    }
    if (!(fusion & 2)) py=at::empty({n,angles,v},x.options());
    native.x=x.const_data_ptr<float>(); native.w=w.const_data_ptr<float>();
    native.y=out.mutable_data_ptr<float>(); native.ids=index_ptr;
    native.din=x.size(1); native.dout=dout;
    const auto stream=c10::cuda::getCurrentCUDAStream(x.get_device());
    const auto work=std::max(n*maxdim*u,int64_t(u)*v);
    if (!(fusion & 1)) {
    pack_linear_input<<<dim3(std::min<int64_t>(4096,(work+255)/256),paths.size()),256,0,stream>>>(
        x.const_data_ptr<float>(),w.const_data_ptr<float>(),px.mutable_data_ptr<float>(),pw.mutable_data_ptr<float>(),
        index_ptr,meta,n,x.size(1),angles,u,v);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    if (fusion) {
        multipath_equi_linear_fwd_f32_native(py.defined() ? py.mutable_data_ptr<float>() : nullptr,
            px.defined() ? px.mutable_data_ptr<float>() : nullptr,
            pw.defined() ? pw.mutable_data_ptr<float>() : nullptr,
            dims,n,angles,u,v,coeff,stream,native,fusion);
    } else multipath_equi_linear_fwd_f32_general(py.mutable_data_ptr<float>(),px.mutable_data_ptr<float>(),
        pw.mutable_data_ptr<float>(),dims,dims,n,angles,angles,u,v,coeff,stream);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    if (!(fusion & 2)) {
    unpack_linear_output<<<dim3(std::min<int64_t>(4096,(n*maxdim*v+255)/256),paths.size()),256,0,stream>>>(
        py.const_data_ptr<float>(),out.mutable_data_ptr<float>(),index_ptr,meta,n,dout,angles,v);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
    return out;
}
at::Tensor linear(const at::Tensor& x,const at::Tensor& w,
    std::vector<std::vector<int64_t>> paths,std::vector<double> coeff,int64_t dout,int fusion) {
    return linear_impl(x,w,paths,coeff,dout,at::Tensor(),at::Tensor(),fusion);
}
at::Tensor indexed_out(const at::Tensor& x,const at::Tensor& w,const at::Tensor& ids,
    at::Tensor out,std::vector<std::vector<int64_t>> paths,std::vector<double> coeff,int fusion) {
    TORCH_CHECK(out.dim()==2,"Expected matrix output");
    return linear_impl(x,w,paths,coeff,out.size(1),ids,out,fusion);
}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){
    namespace py=pybind11;
    m.def("linear",&linear,py::arg("x"),py::arg("w"),py::arg("paths"),
        py::arg("coeff"),py::arg("dout"),py::arg("fusion")=0);
    m.def("indexed_out",&indexed_out,py::arg("x"),py::arg("w"),py::arg("ids"),
        py::arg("out"),py::arg("paths"),py::arg("coeff"),py::arg("fusion")=0);
}
