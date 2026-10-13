// Native autograd for FastEq's metadata-driven indexed Linear kernels.
#include "jit_runtime.h"

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <memory>
#include <vector>

namespace py = pybind11;
using Tensor = at::Tensor;
using Operands = std::array<Tensor, 3>;
using Kernel = fasteq_jit::Kernel;

namespace {
struct LinearPlan;
Tensor evaluate(const std::shared_ptr<LinearPlan>&, int, const Tensor&,
                const Operands&, int64_t);

// Bilinear transposes have exactly two operands. The destination is not saved
// and has no gradient edge. Selective grad requests unpack only the operand
// actually needed, preserving PyTorch's in-place version-check semantics.
struct LinearDerivative : torch::autograd::Node {
    std::shared_ptr<LinearPlan> plan;
    int output_role;
    int64_t elements;
    std::array<int, 2> input_roles;
    torch::autograd::SavedVariable types;
    std::array<torch::autograd::SavedVariable, 2> saved;

    LinearDerivative(std::shared_ptr<LinearPlan> p, int role, const Tensor& t,
                     const Operands& values, int64_t count)
        : plan(std::move(p)), output_role(role), elements(count), types(t, false) {
        int k = 0;
        for (int i = 0; i < 3; ++i) if (i != role) input_roles[k++] = i;
        for (int i = 0; i < 2; ++i)
            saved[i] = torch::autograd::SavedVariable(values[input_roles[i]], false);
        set_next_edges(torch::autograd::collect_next_edges(
            values[input_roles[0]], values[input_roles[1]]));
    }

    torch::autograd::variable_list apply(torch::autograd::variable_list&& cotangents) override {
        torch::autograd::variable_list gradients(2);
        if (cotangents.empty() || !cotangents[0].defined()) return gradients;
        Tensor node_types;
        const bool need0 = task_should_compute_output(0);
        const bool need1 = task_should_compute_output(1);
        std::array<Tensor, 2> required_values;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            node_types = types.unpack();
            if (need0) required_values[1] = saved[1].unpack();
            if (need1) required_values[0] = saved[0].unpack();
        }
        for (int k = 0; k < 2; ++k) {
            if (!(k == 0 ? need0 : need1)) continue;
            Operands values;
            values[output_role] = cotangents[0];
            values[input_roles[1 - k]] = required_values[1 - k];
            gradients[k] = evaluate(plan, input_roles[k], node_types, values, elements);
        }
        return gradients;
    }

    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        types.reset_data();
        for (auto& value : saved) value.reset_data();
    }

    std::string name() const override { return "FastEqIndexedLinearBackward"; }
};

struct LinearPlan : std::enable_shared_from_this<LinearPlan> {
    std::vector<std::shared_ptr<Kernel>> kernels;
    Tensor paths, scales;
    std::vector<Tensor> groups, path_ids, tiles;
    int64_t din, dout, numel, experts;
    bool ir_mul;
    std::vector<bool> covered;
    std::array<Tensor, 3> mma_tiles;
    int mma_max_dim = 0;

    LinearPlan(std::vector<std::shared_ptr<Kernel>> ks, Tensor ps, Tensor cs,
               std::vector<Tensor> gs, std::vector<Tensor> ids, std::vector<Tensor> ts,
               int64_t input, int64_t output, int64_t weight, int64_t heads,
               bool angular_first, std::vector<bool> coverage)
        : kernels(std::move(ks)), paths(std::move(ps)), scales(std::move(cs)),
          groups(std::move(gs)), path_ids(std::move(ids)), tiles(std::move(ts)),
          din(input), dout(output), numel(weight), experts(heads),
          ir_mul(angular_first), covered(std::move(coverage)) {
        TORCH_CHECK((kernels.size() == 5 || kernels.size() == 6 || kernels.size() == 8) && groups.size() == 3 && path_ids.size() == 3
                    && tiles.size() == 3 && covered.size() == 3, "Invalid Linear plan");
        TORCH_CHECK(paths.is_cuda() && paths.scalar_type() == at::kInt && paths.dim() == 2
                    && paths.size(1) == 6 && scales.numel() == paths.size(0), "Invalid path metadata");
    }

    Tensor launch(int role, const Tensor& node_types, const Operands& original, int64_t elements) const {
        TORCH_CHECK(role >= 0 && role < 3, "Invalid Linear transpose role");
        Operands inputs;
        int reference = (role + 1) % 3;
        const auto& prototype = original[reference];
        TORCH_CHECK(prototype.is_cuda() && prototype.device() == paths.device()
                    && node_types.device() == prototype.device(), "Linear tensors must share a CUDA device");
        TORCH_CHECK(prototype.scalar_type() == scales.scalar_type(), "Linear plan dtype mismatch");
        TORCH_CHECK(node_types.dim() == 1 && node_types.scalar_type() == at::kLong, "Expected int64 node types");
        const auto nodes = node_types.numel();
        int chunks = role == 1 ? (nodes + 31) / 32 : 1;
        int64_t chunk_span = 32;
        const bool tensor_core = elements == 1 && nodes >= 1024
            && mma_tiles[role].defined() && mma_tiles[role].size(0)
            && (role == 2 ? kernels.size() >= 6 : kernels.size() == 8)
            && (role != 1 || experts * numel * prototype.element_size() <= (64LL << 20));
        const auto& selected_tiles = tensor_core ? mma_tiles[role] : tiles[role];
        const int64_t chunk_bytes = elements * experts * numel * prototype.element_size();
        // Keep shared/small-element reductions below the scratch budget by
        // increasing their node span. Switching a large shared map to one
        // atomic per node and weight cell creates severe contention.
        constexpr int64_t scratch_budget = 64LL << 20;
        const bool atomic_weights = role == 1 &&
            (elements > 8 || chunk_bytes > scratch_budget);
        if (role == 1 && !atomic_weights && chunk_bytes > 0) {
            const auto max_chunks = std::max<int64_t>(1, scratch_budget / chunk_bytes);
            if (chunks > max_chunks) {
                const auto minimum_span = (nodes + max_chunks - 1) / max_chunks;
                chunk_span = ((minimum_span + 31) / 32) * 32;
                chunks = (nodes + chunk_span - 1) / chunk_span;
            }
        }
        const auto shape = role == 1 ? std::vector<int64_t>{elements, experts, numel}
                                   : std::vector<int64_t>{nodes, role == 0 ? din : dout};
        auto result = at::empty(shape, prototype.options());
        if (!covered[role] || !nodes || !selected_tiles.size(0) || atomic_weights) result.zero_();
        if (!nodes || !selected_tiles.size(0)) return result;
        for (int k = 0; k < 3; ++k) if (k != role) inputs[k] = original[k].contiguous();
        const auto types_contiguous = node_types.contiguous();
        Tensor partial;
        if (role == 1 && chunks > 1 && !atomic_weights)
            partial = at::empty({chunks, elements, experts, numel}, prototype.options());
        auto args = [&](const Operands& values, const Tensor& destination, const Tensor& table) {
            std::vector<uint64_t> packed;
            for (const auto& value : values)
                packed.push_back(value.defined() ? reinterpret_cast<uint64_t>(value.data_ptr()) : 0);
            packed.push_back(reinterpret_cast<uint64_t>(types_contiguous.data_ptr()));
            packed.push_back(reinterpret_cast<uint64_t>(destination.data_ptr()));
            packed.push_back(reinterpret_cast<uint64_t>(paths.data_ptr()));
            packed.push_back(reinterpret_cast<uint64_t>(scales.data_ptr()));
            packed.push_back(reinterpret_cast<uint64_t>(groups[role].data_ptr()));
            packed.push_back(reinterpret_cast<uint64_t>(path_ids[role].data_ptr()));
            packed.push_back(reinterpret_cast<uint64_t>(table.data_ptr()));
            packed.insert(packed.end(), {uint64_t(nodes), uint64_t(din), uint64_t(dout), uint64_t(numel),
                uint64_t(experts), uint64_t(ir_mul), uint64_t(elements), uint64_t(table.size(0)),
                uint64_t(chunks), uint64_t(chunk_span)});
            return packed;
        };
        const auto stream = reinterpret_cast<uint64_t>(c10::cuda::getCurrentCUDAStream(result.get_device()).stream());
        const auto count = selected_tiles.size(0);
        const auto kernel = tensor_core ? (role == 0 ? 6 : role == 1 ? 7 : 5) : atomic_weights ? 4 : role;
        const int tile_rows = scales.scalar_type() == at::kFloat ? 32 : 64;
        kernels[kernel]->launch(args(inputs, partial.defined() ? partial : result, selected_tiles),
                              tensor_core && role != 1 ? (nodes * mma_max_dim + tile_rows - 1) / tile_rows : role == 1 && !atomic_weights ? chunks : nodes,
                              role == 1 && !atomic_weights ? count * elements : count,
                              128, stream);
        if (partial.defined()) {
            Operands reduction;
            reduction[0] = partial;
            kernels[3]->launch(args(reduction, result, tiles[role]), tiles[role].size(0) * elements, 1, 128, stream);
        }
        return result;
    }

    Tensor forward(const Tensor& features, const Tensor& weight, const Tensor& types) {
        TORCH_CHECK(features.dim() == 2 && features.size(1) == din && features.size(0) == types.numel(),
                    "Features do not match Linear dimensions");
        TORCH_CHECK(weight.dim() == 3 && weight.size(1) == experts && weight.size(2) == numel,
                    "Expected weights [elements, experts, weight_numel]");
        return evaluate(shared_from_this(), 2, types, {features, weight, Tensor()}, weight.size(0));
    }

    Tensor contract(int role, const Tensor& types, const Tensor& x, const Tensor& w, const Tensor& y) {
        return evaluate(shared_from_this(), role, types, {x, w, y}, w.size(0));
    }

    void configure_tensor_core(Tensor table, int max_dim) {
        TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kInt && table.dim() == 2 && table.size(1) == 4,
                    "Invalid Tensor-Core tile metadata");
        TORCH_CHECK(scales.scalar_type() == at::kDouble && max_dim > 0, "FP64 Tensor-Core plan required");
        mma_tiles[2] = std::move(table);
        mma_max_dim = max_dim;
    }

    void configure_derivative_tensor_core(Tensor dx, Tensor dw) {
        TORCH_CHECK(kernels.size() == 8 && scales.scalar_type() == at::kDouble,
                    "FP64 derivative Tensor-Core kernels required");
        for (const auto& table : {dx, dw})
            TORCH_CHECK(table.is_cuda() && table.scalar_type() == at::kInt && table.dim() == 2 && table.size(1) == 4,
                        "Invalid derivative Tensor-Core tile metadata");
        mma_tiles[0] = std::move(dx);
        mma_tiles[1] = std::move(dw);
    }

    void configure_simt_tiles(Tensor dx, Tensor dw, Tensor forward, int max_dim) {
        TORCH_CHECK(kernels.size() == 8 && scales.scalar_type() == at::kFloat && max_dim > 0,
                    "FP32 SIMT tile kernels required");
        for (const auto& table : {dx, dw, forward})
            TORCH_CHECK(table.device() == paths.device() && table.scalar_type() == at::kInt
                        && table.dim() == 2 && table.size(1) == 4, "Invalid SIMT tile metadata");
        mma_tiles = {std::move(dx), std::move(dw), std::move(forward)};
        mma_max_dim = max_dim;
    }
};

Tensor evaluate(const std::shared_ptr<LinearPlan>& plan, int role, const Tensor& types,
                const Operands& inputs, int64_t elements) {
    bool tracked = torch::autograd::GradMode::is_enabled() && !c10::InferenceMode::is_enabled();
    bool differentiable = false;
    for (int k = 0; k < 3; ++k) if (k != role) differentiable |= inputs[k].requires_grad();
    Tensor result;
    {
        at::NoGradGuard guard;
        result = plan->launch(role, types, inputs, elements);
    }
    if (tracked && differentiable)
        torch::autograd::set_history(result, std::make_shared<LinearDerivative>(plan, role, types, inputs, elements));
    return result;
}
} // namespace

void bind_indexed_linear(py::module_& module) {
    py::class_<LinearPlan, std::shared_ptr<LinearPlan>>(module, "IndexedLinearPlan")
        .def(py::init<std::vector<std::shared_ptr<Kernel>>, Tensor, Tensor,
                      std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>,
                      int64_t, int64_t, int64_t, int64_t, bool, std::vector<bool>>())
        .def("forward", &LinearPlan::forward, py::call_guard<py::gil_scoped_release>())
        .def("contract", &LinearPlan::contract, py::call_guard<py::gil_scoped_release>())
        .def("configure_tensor_core", &LinearPlan::configure_tensor_core)
        .def("configure_derivative_tensor_core", &LinearPlan::configure_derivative_tensor_core)
        .def("configure_simt_tiles", &LinearPlan::configure_simt_tiles);
}
