// Sparse angular descriptors, dense channel contractions, bounded CUDA scratch.
// Autograd remains the four-role multilinear FastEq Function; this plan owns
// only stream-ordered CUDA packing/scattering and full-precision cuBLAS math.
#include "jit_runtime.h"
#include <ATen/ATen.h>
#include <ATen/core/grad_mode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <memory>
#include <vector>

namespace py = pybind11;
using Tensor = at::Tensor;
using Kernel = fasteq_jit::Kernel;

namespace {
uint64_t pointer(const Tensor& value) {
    return reinterpret_cast<uint64_t>(value.data_ptr());
}
uint64_t scalar_bits(double value, at::ScalarType dtype) {
    uint64_t bits = 0;
    if (dtype == at::kDouble) std::memcpy(&bits, &value, sizeof(value));
    else { float f = static_cast<float>(value); std::memcpy(&bits, &f, sizeof(f)); }
    return bits;
}

struct SegmentedPlan {
    std::array<int64_t, 4> dimensions;
    std::vector<std::array<int64_t, 12>> terms;
    std::vector<double> coefficients;
    std::shared_ptr<Kernel> pack, scatter, outer;
    Tensor descriptors, scales;
    std::vector<std::pair<int,int>> groups;
    std::vector<bool> angular_gemm;

    SegmentedPlan(const std::vector<int64_t>& dims,
                  const std::vector<std::vector<int64_t>>& paths,
                  const std::vector<double>& scales,
                  std::shared_ptr<Kernel> p, std::shared_ptr<Kernel> s,
                  std::shared_ptr<Kernel> o, const Tensor& tables, const Tensor& factors)
        : coefficients(scales), pack(std::move(p)), scatter(std::move(s)),
          outer(std::move(o)),descriptors(tables),scales(factors) {
        TORCH_CHECK(dims.size() == 4 && paths.size() == scales.size(), "Invalid TP plan");
        std::copy(dims.begin(), dims.end(), dimensions.begin());
        for (const auto& path : paths) {
            TORCH_CHECK(path.size() == 12 && path[0] == 3 && path[11], "Dense plan requires weighted UVW");
            std::array<int64_t,12> term;
            std::copy(path.begin(), path.end(), term.begin());
            terms.push_back(term);
        }
        for (int begin=0;begin<static_cast<int>(terms.size());) {
            int end=begin+1;
            const auto& a=terms[begin];
            for (;end<static_cast<int>(terms.size());++end) {
                const auto& b=terms[end];
                if (a[1]!=b[1] || a[6]!=b[6] || a[7]!=b[7] ||
                    a[8]!=b[8] || a[9]!=b[9] || a[10]!=b[10]) break;
            }
            bool contiguous=end-begin>=2 && a[3]>=end-begin && a[5]>=end-begin;
            for (int t=begin;t<end && contiguous;++t) {
                const auto& b=terms[t];
                const bool equal_coefficient=factors.scalar_type()==at::kFloat
                    ? static_cast<float>(coefficients[t])==static_cast<float>(coefficients[begin])
                    : coefficients[t]==coefficients[begin];
                contiguous=b[2]==a[2]+t-begin && b[4]==a[4]+t-begin &&
                    b[3]==a[3] && b[5]==a[5] && equal_coefficient;
            }
            groups.emplace_back(begin,end);angular_gemm.push_back(contiguous);begin=end;
        }
    }

    void packing(const Tensor& a, const Tensor& b, const Tensor& out, int op,
                 int64_t start, int64_t n, int64_t ba, int64_t sa, int64_t ca,
                 int64_t bb, int64_t sb, int64_t cb, int64_t co, uint64_t stream) const {
        const int64_t elements = op == 2 ? ca * cb * co : n * ca * (op == 1 ? cb : 1);
        pack->launch({pointer(a),pointer(b),pointer(out),static_cast<uint64_t>(op),
            static_cast<uint64_t>(start),static_cast<uint64_t>(n),
            static_cast<uint64_t>(a.size(0)),static_cast<uint64_t>(b.size(0)),
            static_cast<uint64_t>(a.size(1)),static_cast<uint64_t>(b.size(1)),
            static_cast<uint64_t>(ba),static_cast<uint64_t>(sa),static_cast<uint64_t>(ca),
            static_cast<uint64_t>(bb),static_cast<uint64_t>(sb),static_cast<uint64_t>(cb),
            static_cast<uint64_t>(co)}, (elements+255)/256,1,256,stream);
    }

    void scattering(const Tensor& temp, const Tensor& partner, Tensor& result,
                    int op, int64_t start, int64_t n, int64_t partner_base,
                    int64_t partner_stride, int64_t output_base, int64_t output_stride,
                    int64_t channels, int64_t contract, double coefficient, uint64_t stream) const {
        scatter->launch({pointer(temp),pointer(partner),pointer(result),static_cast<uint64_t>(op),
            static_cast<uint64_t>(start),static_cast<uint64_t>(n),
            static_cast<uint64_t>(partner.size(0)),static_cast<uint64_t>(partner.size(1)),
            static_cast<uint64_t>(partner_base),static_cast<uint64_t>(partner_stride),
            static_cast<uint64_t>(result.size(1)),static_cast<uint64_t>(output_base),
            static_cast<uint64_t>(output_stride),static_cast<uint64_t>(channels),
            static_cast<uint64_t>(contract),scalar_bits(coefficient,result.scalar_type())},
            (n*channels+255)/256,1,256,stream);
    }

    Tensor evaluate(int role, int required, int64_t rows,
                    const std::vector<Tensor>& values, Tensor result) const {
        TORCH_CHECK(values.size() == 4 && role >= 0 && role < 4, "Invalid TP operands");
        at::NoGradGuard no_grad;
        const c10::cuda::CUDAGuard guard(values[1].device());
        const auto stream = static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            c10::cuda::getCurrentCUDAStream(values[1].get_device()).stream()));
        const int64_t count = role == 3 ? rows : values[role].size(0);
        if (!result.defined() || result.numel() == 0)
            result = at::zeros({count,dimensions[role]},values[1].options());
        if (rows == 0 || dimensions[role] == 0) return result;
        TORCH_CHECK(values[0].size(0) == 1, "Shared UVW plan requires shared weights");
        constexpr int64_t budget = 64LL * 1024 * 1024;
        const int64_t element_bytes = values[1].element_size();
        // One weight/output-angular domain is one GEMM for every role.
        // Feature transposes reuse DZ@W^T across sparse angular terms, rather
        // than materializing Y*DZ/X*DZ and repeating a GEMM for each term.
        for (size_t g=0;g<groups.size();++g) {
            const auto& group=groups[g];
            const auto& p=terms[group.first];
            const int64_t c1=p[8],c2=p[9],co=p[10];
            const int64_t chunk=std::min<int64_t>(rows,std::min<int64_t>(32768,
                budget/element_bytes/(c1*c2+co)));
            TORCH_CHECK(chunk>0,"Angular UVW scratch exceeds plan budget");
            Tensor first=at::empty({chunk,c1*c2},values[1].options());
            Tensor second=at::empty({chunk,co},values[1].options());
            const auto weight=values[0].reshape({-1}).narrow(0,p[1],c1*c2*co).view({c1*c2,co});
            for (int64_t start=0;start<rows;start+=chunk) {
                const int64_t n=std::min(chunk,rows-start);
                Tensor a=first.narrow(0,0,n),b=second.narrow(0,0,n);
                if (role==0 || role==3) {
                    if (angular_gemm[g]) {
                        const int64_t angular=group.second-group.first;
                        const Tensor left=values[1].as_strided({n,c1,angular},
                            {values[1].size(1),p[3],1},values[1].storage_offset()+start*values[1].size(1)+p[2]);
                        const Tensor right=values[2].as_strided({n,angular,c2},
                            {values[2].size(1),1,p[5]},values[2].storage_offset()+start*values[2].size(1)+p[4]);
                        Tensor destination=a.view({n,c1,c2});
                        fasteq_jit::batched_gemm_into(destination,left,right,coefficients[group.first],0.);
                    } else outer->launch({pointer(values[1]),pointer(values[2]),pointer(a),
                        pointer(descriptors),pointer(scales),static_cast<uint64_t>(start),
                        static_cast<uint64_t>(n),static_cast<uint64_t>(values[1].size(0)),
                        static_cast<uint64_t>(values[2].size(0)),static_cast<uint64_t>(values[1].size(1)),
                        static_cast<uint64_t>(values[2].size(1)),static_cast<uint64_t>(group.first),
                        static_cast<uint64_t>(group.second),static_cast<uint64_t>(c1),static_cast<uint64_t>(c2)},
                        (n*c1*c2+255)/256,1,256,stream);
                    if (role==3) {
                        fasteq_jit::gemm_into(b,a,weight,1.,0.);
                        scattering(b,values[2],result,1,start,n,0,1,p[6],p[7],co,1,1.,stream);
                    } else {
                        packing(values[3],values[3],b,0,start,n,p[6],p[7],co,0,1,1,1,stream);
                        Tensor destination=result.reshape({-1}).narrow(0,p[1],c1*c2*co).view({c1*c2,co});
                        fasteq_jit::gemm_into(destination,a.transpose(0,1),b,1.,1.);
                    }
                } else {
                    packing(values[3],values[3],b,0,start,n,p[6],p[7],co,0,1,1,1,stream);
                    fasteq_jit::gemm_into(a,b,weight.transpose(0,1),1.,0.);
                    for (int t=group.first;t<group.second;++t) {
                        const auto& q=terms[t];
                        if (role==1)
                            scattering(a,values[2],result,2,start,n,q[4],q[5],q[2],q[3],c1,c2,coefficients[t],stream);
                        else
                            scattering(a,values[1],result,3,start,n,q[2],q[3],q[4],q[5],c2,c1,coefficients[t],stream);
                    }
                }
            }
        }
        return result;
    }

    std::vector<Tensor> evaluate_features(int required, int64_t rows,
                    const std::vector<Tensor>& values, Tensor resultx, Tensor resulty) const {
        TORCH_CHECK(values.size()==4 && values[0].size(0)==1,"Invalid shared feature VJP");
        at::NoGradGuard no_grad;
        const c10::cuda::CUDAGuard guard(values[1].device());
        const auto stream=static_cast<uint64_t>(reinterpret_cast<uintptr_t>(
            c10::cuda::getCurrentCUDAStream(values[1].get_device()).stream()));
        if (!resultx.defined() || resultx.numel()==0)
            resultx=at::zeros({rows,dimensions[1]},values[1].options());
        if (!resulty.defined() || resulty.numel()==0)
            resulty=at::zeros({rows,dimensions[2]},values[1].options());
        if (rows==0) return {resultx,resulty};
        constexpr int64_t budget=64LL*1024*1024;
        for (const auto& group:groups) {
            const auto& p=terms[group.first];
            const int64_t c1=p[8],c2=p[9],co=p[10];
            const int64_t chunk=std::min<int64_t>(rows,std::min<int64_t>(32768,
                budget/values[1].element_size()/(c1*c2+co)));
            TORCH_CHECK(chunk>0,"Paired UVW scratch exceeds plan budget");
            Tensor first=at::empty({chunk,c1*c2},values[1].options());
            Tensor second=at::empty({chunk,co},values[1].options());
            const Tensor weight=values[0].reshape({-1}).narrow(0,p[1],c1*c2*co).view({c1*c2,co});
            for (int64_t start=0;start<rows;start+=chunk) {
                const int64_t n=std::min(chunk,rows-start);
                Tensor a=first.narrow(0,0,n),b=second.narrow(0,0,n);
                packing(values[3],values[3],b,0,start,n,p[6],p[7],co,0,1,1,1,stream);
                fasteq_jit::gemm_into(a,b,weight.transpose(0,1),1.,0.);
                for (int t=group.first;t<group.second;++t) {
                    const auto& q=terms[t];
                    scattering(a,values[2],resultx,2,start,n,q[4],q[5],q[2],q[3],c1,c2,coefficients[t],stream);
                    scattering(a,values[1],resulty,3,start,n,q[2],q[3],q[4],q[5],c2,c1,coefficients[t],stream);
                }
            }
        }
        return {resultx,resulty};
    }
};
} // namespace

void bind_segmented_product(py::module_& module) {
    py::class_<SegmentedPlan,std::shared_ptr<SegmentedPlan>>(module,"SegmentedProductPlan")
        .def(py::init<const std::vector<int64_t>&,
                      const std::vector<std::vector<int64_t>>&,
                      const std::vector<double>&,
                      std::shared_ptr<Kernel>,std::shared_ptr<Kernel>,
                      std::shared_ptr<Kernel>,const Tensor&,const Tensor&>())
        .def("evaluate", &SegmentedPlan::evaluate,py::call_guard<py::gil_scoped_release>())
        .def("evaluate_features", &SegmentedPlan::evaluate_features,py::call_guard<py::gil_scoped_release>());
}
