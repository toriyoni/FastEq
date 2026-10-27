// Bindings for FastEq's extended continuous-Y FCTP CUDA kernels. The existing
// SegmentedProduct four-role autograd owns all differentiation and live inputs.
#include <ATen/ATen.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/pybind11.h>
namespace fasteq_jit {
void fctp_forward_into(at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,double);
void fctp_weight_into(at::Tensor&,const at::Tensor&,const at::Tensor&,const at::Tensor&,double);
void fctp_features_into(at::Tensor&,at::Tensor&,const at::Tensor&,const at::Tensor&,
                        const at::Tensor&,const at::Tensor&,double,int);
}
void bind_fctp_general(pybind11::module_& module) {
    module.def("fctp_forward_into",&fasteq_jit::fctp_forward_into);
    module.def("fctp_weight_into",&fasteq_jit::fctp_weight_into);
    module.def("fctp_features_into",&fasteq_jit::fctp_features_into);
}
