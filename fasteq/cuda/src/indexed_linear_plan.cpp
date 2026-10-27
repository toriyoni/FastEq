// Native autograd for FastEq's metadata-driven indexed Linear kernels.
#include "jit_runtime.h"

#include <ATen/ATen.h>
#include <ATen/Context.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <algorithm>
#include <array>
#include <cstring>
#include <cstdlib>
#include <memory>
#include <set>
#include <vector>

namespace py = pybind11;
using Tensor = at::Tensor;
using Operands = std::array<Tensor, 3>;
using Kernel = fasteq_jit::Kernel;

namespace fasteq_jit {
// Reusable FastEq matrix primitive. The CUDA implementation extends FastEq's
// own indexed Linear SIMT tiles and original FP64 MMA/async-copy primitives.
// The caller supplies legal strided views and owns the recursive autograd.
void gemm_into(Tensor& destination, const Tensor& left, const Tensor& right,
               double alpha, double beta) {
    native_gemm_into(destination, left, right, alpha, beta);
}
void batched_gemm_into(Tensor& destination, const Tensor& left, const Tensor& right,
                       double alpha, double beta) {
    native_gemm_into(destination, left, right, alpha, beta);
}
}  // namespace fasteq_jit

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
    struct GemmPath {
        int64_t sx, sw, sy, ci, co, dim;
        double scale;
    };
    std::vector<std::shared_ptr<Kernel>> kernels;
    Tensor paths, scales;
    std::vector<Tensor> groups, path_ids, tiles;
    int64_t din, dout, numel, experts;
    bool ir_mul;
    std::vector<bool> covered;
    std::array<Tensor, 3> mma_tiles;
    int mma_max_dim = 0;
    std::vector<GemmPath> gemm_paths;
    // Cached once at shared-GEMM configuration; launch never scans paths or queries the device.
    bool adaptive_o3_c64_h100 = false;
    // FastEq's SO(3) serial forward / input transpose, compiled once
    // by Triton and launched through the same resident CUDA driver as the
    // original kernels. Autograd remains in LinearDerivative, which prunes
    // dW when a caller requests only position/input derivatives.
    // Slots 0/2 are dX/forward; slot 1 packs W.T for dX (not the dW kernel).
    std::array<std::shared_ptr<Kernel>, 3> serial_kernels;
    std::array<std::array<int64_t, 4>, 3> serial_launch;
    int64_t serial_nodes = 0;

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

    Tensor shared_weight_partials(const Operands& values, int64_t nodes) const {
        const auto& prototype = values[0];
        const auto bytes = experts * numel * prototype.element_size();
        const auto max_chunks = std::max<int64_t>(1, (64LL << 20) / bytes);
        int64_t span = 32;
        if ((nodes + 31) / 32 > max_chunks)
            span = (((nodes + max_chunks - 1) / max_chunks + 31) / 32) * 32;
        const auto chunks = (nodes + span - 1) / span;
        const auto full = nodes / span, tail = nodes % span;
        auto partial = at::empty({chunks, 1, experts, numel}, prototype.options());
        auto result = at::empty({1, experts, numel}, prototype.options());
        if (!covered[1]) result.zero_();
        std::set<std::array<int64_t, 2>> written;
        for (const auto& path : gemm_paths) {
            for (int64_t expert = 0; expert < experts; ++expert) {
                auto pack = [&](const Tensor& original, int64_t width, int64_t base, int64_t channels) {
                    const auto value = original.contiguous();
                    const auto offset = base + (ir_mul ? expert * channels : expert * channels * path.dim);
                    return value.as_strided({nodes, path.dim, channels},
                        {width, ir_mul ? experts * channels : 1, ir_mul ? 1 : path.dim},
                        value.storage_offset() + offset).contiguous().view({nodes * path.dim, channels});
                };
                auto x = pack(values[0], din, path.sx, path.ci);
                auto y = pack(values[2], dout, path.sy, path.co);
                const bool first = written.insert({path.sw, expert}).second;
                if (full) {
                    auto left = x.narrow(0, 0, full * span * path.dim).view({full, span * path.dim, path.ci}).transpose(1, 2);
                    auto right = y.narrow(0, 0, full * span * path.dim).view({full, span * path.dim, path.co});
                    auto destination = partial.as_strided({full, path.ci, path.co},
                        {experts * numel, path.co, 1}, expert * numel + path.sw);
                    fasteq_jit::batched_gemm_into(destination, left, right, path.scale, first ? 0 : 1);
                }
                if (tail) {
                    auto left = x.narrow(0, full * span * path.dim, tail * path.dim).t();
                    auto right = y.narrow(0, full * span * path.dim, tail * path.dim);
                    auto destination = partial.as_strided({path.ci, path.co}, {path.co, 1},
                        (full * experts + expert) * numel + path.sw);
                    fasteq_jit::gemm_into(destination, left, right, path.scale, first ? 0 : 1);
                }
            }
        }
        std::vector<uint64_t> args{reinterpret_cast<uint64_t>(partial.data_ptr()), 0, 0, 0,
            reinterpret_cast<uint64_t>(result.data_ptr()), reinterpret_cast<uint64_t>(paths.data_ptr()),
            reinterpret_cast<uint64_t>(scales.data_ptr()), reinterpret_cast<uint64_t>(groups[1].data_ptr()),
            reinterpret_cast<uint64_t>(path_ids[1].data_ptr()), reinterpret_cast<uint64_t>(tiles[1].data_ptr()),
            uint64_t(nodes), uint64_t(din), uint64_t(dout), uint64_t(numel), uint64_t(experts),
            uint64_t(ir_mul), 1, uint64_t(tiles[1].size(0)), uint64_t(chunks), uint64_t(span)};
        const auto stream = reinterpret_cast<uint64_t>(c10::cuda::getCurrentCUDAStream(result.get_device()).stream());
        kernels[3]->launch(args, tiles[1].size(0), 1, 128, stream);
        return result;
    }

    Tensor shared_gemm(int role, const Operands& values, int64_t nodes) const {
        const auto& prototype = values[(role + 1) % 3];
        c10::cuda::CUDAGuard device_guard(prototype.device());
        if (role == 1 && prototype.scalar_type() == at::kFloat && nodes >= 8192)
            return shared_weight_partials(values, nodes);
        Operands inputs;
        for (int k = 0; k < 3; ++k) if (k != role) inputs[k] = values[k].contiguous();
        const auto shape = role == 1 ? std::vector<int64_t>{1, experts, numel}
                                   : std::vector<int64_t>{nodes, role == 0 ? din : dout};
        auto result = at::empty(shape, prototype.options());
        if (!covered[role]) result.zero_();
        std::set<std::array<int64_t, 4>> written;
        auto feature = [&](const Tensor& value, int64_t width, int64_t base,
                           int64_t channels, int64_t dim, int64_t expert, int64_t angle) {
            const auto offset = base + (ir_mul ? angle * experts * channels + expert * channels
                                              : expert * channels * dim + angle);
            return value.as_strided({nodes, channels}, {width, ir_mul ? 1 : dim},
                                    value.storage_offset() + offset);
        };
        for (const auto& path : gemm_paths) {
            for (int64_t expert = 0; expert < experts; ++expert) {
                Tensor w;
                if (role != 1)
                    w = inputs[1].as_strided({path.ci, path.co}, {path.co, 1},
                                             inputs[1].storage_offset() + expert * numel + path.sw);
                for (int64_t angle = 0; angle < path.dim; ++angle) {
                    Tensor left, right, destination;
                    std::array<int64_t, 4> key;
                    if (role == 1) {
                        left = feature(inputs[0], din, path.sx, path.ci, path.dim, expert, angle).t();
                        right = feature(inputs[2], dout, path.sy, path.co, path.dim, expert, angle);
                        destination = result.as_strided({path.ci, path.co}, {path.co, 1},
                                                        expert * numel + path.sw);
                        key = {path.sw, expert, 0, 0};
                    } else if (role == 0) {
                        left = feature(inputs[2], dout, path.sy, path.co, path.dim, expert, angle);
                        right = w.t();
                        destination = feature(result, din, path.sx, path.ci, path.dim, expert, angle);
                        key = {path.sx, expert, angle, path.dim};
                    } else {
                        left = feature(inputs[0], din, path.sx, path.ci, path.dim, expert, angle);
                        right = w;
                        destination = feature(result, dout, path.sy, path.co, path.dim, expert, angle);
                        key = {path.sy, expert, angle, path.dim};
                    }
                    const bool first = written.insert(key).second;
                    fasteq_jit::gemm_into(destination, left, right,
                                          path.scale, first ? 0 : 1);
                }
            }
        }
        return result;
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
        if (role != 1 && serial_kernels[role] && nodes == serial_nodes
            && elements == 1 && experts == 1 && prototype.scalar_type() == at::kFloat) {
            c10::cuda::CUDAGuard device_guard(prototype.device());
            auto weight = original[1].contiguous();
            auto feature = original[role == 2 ? 0 : 2].contiguous();
            // Resident executables specialize aligned pointers. A contiguous
            // view may still have a storage offset, so retain the original
            // kernels for unaligned operands instead of violating that ABI.
            if (reinterpret_cast<uint64_t>(weight.data_ptr()) % 16 == 0
                && reinterpret_cast<uint64_t>(feature.data_ptr()) % 16 == 0) {
                auto result = at::empty({nodes, role == 2 ? dout : din}, prototype.options());
                const auto& config = serial_launch[role];
                const auto stream = reinterpret_cast<uint64_t>(
                    c10::cuda::getCurrentCUDAStream(result.get_device()).stream());
                if (role == 0) {
                    auto packed = at::empty_like(weight);
                    const auto& pack = serial_launch[1];
                    serial_kernels[1]->launch(
                        {reinterpret_cast<uint64_t>(weight.data_ptr()),
                         reinterpret_cast<uint64_t>(packed.data_ptr()), 0},
                        pack[0], pack[1], pack[2], stream, pack[3]);
                    weight = std::move(packed);
                }
                serial_kernels[role]->launch(
                    {reinterpret_cast<uint64_t>(weight.data_ptr()),
                     reinterpret_cast<uint64_t>(feature.data_ptr()),
                     reinterpret_cast<uint64_t>(result.data_ptr()), 0},
                    config[0], config[1], config[2], stream, config[3]);
                return result;
            }
        }
        // Reuse only already-configured FastEq fused tiles in the measured exact O(3) domain.
        // FP64 forward above the measured-safe R49152 keeps shared GEMM; derivatives retain fusion.
        const bool adaptive_fused = adaptive_o3_c64_h100 && elements == 1
            && nodes >= 1024 && nodes <= 131072
            && (prototype.scalar_type() == at::kFloat || prototype.scalar_type() == at::kDouble)
            && !(role == 2 && prototype.scalar_type() == at::kDouble && nodes > 49152)
            && mma_max_dim == 5 && mma_tiles[role].defined() && mma_tiles[role].size(0)
            && (role == 2 ? kernels.size() >= 6 : kernels.size() == 8)
            && (role != 1 || elements * experts * numel * prototype.element_size() <= (64LL << 20));
        if (!adaptive_fused && elements == 1 && nodes >= 1024 && !gemm_paths.empty() &&
            !(role == 1 && prototype.scalar_type() == at::kFloat && nodes >= 8192 &&
              experts * numel * prototype.element_size() > (64LL << 20)))
            return shared_gemm(role, original, nodes);
        int chunks = role == 1 ? (nodes + 31) / 32 : 1;
        int64_t chunk_span = 32;
        const char* indexed_tiles_option = std::getenv("FASTEQ_EXPERIMENTAL_LINEAR_INDEXED_TILES");
        const bool indexed_tiles = role != 2 && elements > 1 && elements <= 4
            && prototype.scalar_type() == at::kFloat && indexed_tiles_option
            && std::strcmp(indexed_tiles_option, "1") == 0;
        const bool tensor_core = (elements == 1 || indexed_tiles) && nodes >= 1024
            && mma_tiles[role].defined() && mma_tiles[role].size(0)
            && (role == 2 ? kernels.size() >= 6 : kernels.size() == 8)
            && (role != 1 || elements * experts * numel * prototype.element_size() <= (64LL << 20));
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
                              (role == 1 && !atomic_weights) || (tensor_core && indexed_tiles) ? count * elements : count,
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

    void configure_serial(std::vector<std::shared_ptr<Kernel>> executables,
                          const std::vector<std::vector<int64_t>>& launches,
                          int64_t nodes) {
        TORCH_CHECK(executables.size() == 3 && launches.size() == 3 && nodes > 0
                    && experts == 1 && !ir_mul && scales.scalar_type() == at::kFloat,
                    "Invalid shared SO(3) Linear configuration");
        for (int k = 0; k < 3; ++k) {
            const auto& config = launches[k];
            TORCH_CHECK(executables[k] && config.size() == 4
                        && config[0] > 0 && config[0] < (1LL << 31)
                        && config[1] > 0 && config[1] <= 65535
                        && config[2] > 0 && config[2] <= 1024
                        && config[3] >= 0 && config[3] <= 49152,
                        "Invalid shared SO(3) Linear launch");
            const int role = k == 0 ? 2 : k == 1 ? 0 : 1;
            serial_kernels[role] = std::move(executables[k]);
            std::copy(config.begin(), config.end(), serial_launch[role].begin());
        }
        serial_nodes = nodes;
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

    void configure_shared_gemm(const std::vector<std::vector<int64_t>>& descriptions,
                               const std::vector<double>& coefficients) {
        TORCH_CHECK(descriptions.size() == size_t(paths.size(0)) &&
                    coefficients.size() == descriptions.size(), "Invalid shared GEMM paths");
        std::vector<GemmPath> records;
        for (size_t i = 0; i < descriptions.size(); ++i) {
            const auto& p = descriptions[i];
            TORCH_CHECK(p.size() == 6 && p[3] >= 64 && p[4] >= 64 && p[5] > 0 &&
                        p[0] >= 0 && p[0] + experts * p[3] * p[5] <= din &&
                        p[2] >= 0 && p[2] + experts * p[4] * p[5] <= dout &&
                        p[1] >= 0 && p[1] + p[3] * p[4] <= numel,
                        "Invalid shared GEMM descriptor");
            records.push_back({p[0], p[1], p[2], p[3], p[4], p[5], coefficients[i]});
        }
        gemm_paths = std::move(records);
        adaptive_o3_c64_h100 = false;
        static constexpr std::array<std::array<int64_t, 6>, 3> measured_paths{{
            {{0, 0, 0, 64, 64, 1}},
            {{64, 4096, 64, 64, 64, 3}},
            {{256, 8192, 256, 64, 64, 5}}
        }};
        if (ir_mul || experts != 1 || din != 576 || dout != 576 || numel != 12288
            || descriptions.size() != measured_paths.size()) return;
        for (size_t i = 0; i < measured_paths.size(); ++i) {
            if (coefficients[i] != 0.125
                || !std::equal(descriptions[i].begin(), descriptions[i].end(), measured_paths[i].begin()))
                return;
        }
        // These driver property queries run only during plan setup, never inside a timed launch.
        CUdevice device;
        int major = 0, minor = 0;
        char name[256] = {};
        TORCH_CHECK(cuDeviceGet(&device, paths.get_device()) == CUDA_SUCCESS
                    && cuDeviceGetAttribute(&major, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, device) == CUDA_SUCCESS
                    && cuDeviceGetAttribute(&minor, CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR, device) == CUDA_SUCCESS
                    && cuDeviceGetName(name, sizeof(name), device) == CUDA_SUCCESS,
                    "Unable to identify adaptive Linear device");
        adaptive_o3_c64_h100 = major == 9 && minor == 0 && std::strstr(name, "H100") != nullptr;
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
    module.def("native_gemm_into", [](Tensor destination, const Tensor& left,
                                      const Tensor& right, double alpha, double beta) {
        fasteq_jit::native_gemm_into(destination, left, right, alpha, beta);
    }, py::arg("destination"), py::arg("left"), py::arg("right"),
       py::arg("alpha") = 1.0, py::arg("beta") = 0.0,
       py::call_guard<py::gil_scoped_release>());
    py::class_<LinearPlan, std::shared_ptr<LinearPlan>>(module, "IndexedLinearPlan")
        .def(py::init<std::vector<std::shared_ptr<Kernel>>, Tensor, Tensor,
                      std::vector<Tensor>, std::vector<Tensor>, std::vector<Tensor>,
                      int64_t, int64_t, int64_t, int64_t, bool, std::vector<bool>>())
        .def("forward", &LinearPlan::forward, py::call_guard<py::gil_scoped_release>())
        .def("contract", &LinearPlan::contract, py::call_guard<py::gil_scoped_release>())
        .def("configure_serial", &LinearPlan::configure_serial)
        .def("configure_tensor_core", &LinearPlan::configure_tensor_core)
        .def("configure_derivative_tensor_core", &LinearPlan::configure_derivative_tensor_core)
        .def("configure_simt_tiles", &LinearPlan::configure_simt_tiles)
        .def("configure_shared_gemm", &LinearPlan::configure_shared_gemm);
}
