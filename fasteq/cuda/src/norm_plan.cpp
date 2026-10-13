// Selective Norm adjoints retain FastEq's Triton kernels and Python nodes.
// PyNode next_edges includes every tensor input, including non-differentiable
// coefficient tensors, but excludes Python descriptors and mask tuples.
#include <torch/csrc/autograd/function.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <vector>

namespace py = pybind11;

namespace {
std::vector<bool> norm_gradient_mask(size_t count) {
    const auto node = torch::autograd::get_current_node();
    // A mismatch is a conservative fallback, never evidence for pruning.
    if (!node || node->next_edges().size() != count) return {};
    std::vector<bool> requested(count);
    for (size_t i = 0; i < count; ++i)
        requested[i] = node->task_should_compute_output(i);
    return requested;
}
}  // namespace

void bind_norm(py::module_& module) {
    module.def("norm_gradient_mask", &norm_gradient_mask,
               py::call_guard<py::gil_scoped_release>());
}
