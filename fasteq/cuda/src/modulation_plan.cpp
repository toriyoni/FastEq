// Native host/autograd plan for FastEq's existing Triton modulation kernels.
#include "jit_runtime.h"

#include <ATen/ATen.h>
#include <c10/core/InferenceMode.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAGraphsC10Utils.h>
#include <c10/cuda/CUDAStream.h>
#include <torch/csrc/autograd/functions/basic_ops.h>
#include <torch/csrc/autograd/functions/utils.h>
#include <torch/csrc/autograd/saved_variable.h>
#include <torch/csrc/utils/pybind.h>
#include <pybind11/stl.h>
#include <array>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using Tensor = at::Tensor;
using Values = torch::autograd::variable_list;
using Kernel = fasteq_jit::Kernel;

namespace {

// Nodes can be destroyed by a GIL-free autograd worker. This holder also
// avoids a PythonPlan -> nativePlan -> bound PythonPlan callback cycle.
struct Compiler {
    PyObject* object;
    explicit Compiler(const py::object& value) : object(value.ptr()) { Py_INCREF(object); }
    ~Compiler() {
#if PY_VERSION_HEX >= 0x030D0000
        const bool finalizing = Py_IsFinalizing();
#else
        const bool finalizing = _Py_IsFinalizing();
#endif
        if (!Py_IsInitialized() || finalizing) return;
        py::gil_scoped_acquire gil;
        Py_DECREF(object);
    }
    py::object borrow() const { return py::reinterpret_borrow<py::object>(object); }
};

struct Executable {
    std::shared_ptr<Kernel> kernel;
    unsigned threads = 0, shared = 0;
};

struct ModulationPlan;
std::array<Tensor, 2> first(const std::shared_ptr<ModulationPlan>&,
                          const Tensor&, const Tensor&, const Tensor&, unsigned);

struct ModulationDouble : torch::autograd::Node {
    std::shared_ptr<ModulationPlan> plan;
    std::array<torch::autograd::SavedVariable, 3> saved;
    ModulationDouble(std::shared_ptr<ModulationPlan> p,
                     const Tensor& x, const Tensor& g, const Tensor& dy)
        : plan(std::move(p)), saved{torch::autograd::SavedVariable(x, false),
              torch::autograd::SavedVariable(g, false),
              torch::autograd::SavedVariable(dy, false)} {
        set_next_edges(torch::autograd::collect_next_edges(x, g, dy));
    }
    Values apply(Values&& seeds) override;
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& value : saved) value.reset_data();
    }
    std::string name() const override { return "FastEqModulationDoubleBackward"; }
};

struct ModulationBackward : torch::autograd::Node {
    std::shared_ptr<ModulationPlan> plan;
    std::array<torch::autograd::SavedVariable, 2> saved;
    ModulationBackward(std::shared_ptr<ModulationPlan> p,
                       const Tensor& x, const Tensor& g, const Tensor& r)
        : plan(std::move(p)), saved{torch::autograd::SavedVariable(x, false),
              torch::autograd::SavedVariable(g, false)} {
        set_next_edges(torch::autograd::collect_next_edges(x, g, r));
    }
    Values apply(Values&& gradients) override {
        Values result(3);
        if (gradients.empty() || !gradients[0].defined()) return result;
        Tensor x, g;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            x = saved[0].unpack();
            g = saved[1].unpack();
        }
        const unsigned requested = unsigned(task_should_compute_output(0)) |
            (unsigned(task_should_compute_output(1)) << 1);
        if (requested) {
            auto values = first(plan, x, g, gradients[0], requested);
            result[0] = values[0];
            result[1] = values[1];
        }
        // Keep the incoming gradient's history: force-loss uses its indirect
        // route through the forward output, in addition to the direct adjoint.
        if (task_should_compute_output(2)) result[2] = gradients[0];
        return result;
    }
    void release_variables() override {
        std::lock_guard<std::mutex> lock(mutex_);
        for (auto& value : saved) value.reset_data();
    }
    std::string name() const override { return "FastEqModulationBackward"; }
};

struct ModulationPlan : std::enable_shared_from_this<ModulationPlan> {
    Tensor index;
    int64_t dim, channels;
    at::ScalarType dtype;
    std::shared_ptr<Compiler> compiler;
    std::mutex cache_mutex;
    std::unordered_map<uint64_t, Executable> cache;

    ModulationPlan(Tensor indices, int64_t d, int64_t c, const py::object& spec)
        : index(std::move(indices)), dim(d), channels(c),
          dtype(spec.attr("_kwargs")["FP64"].cast<bool>() ? at::kDouble : at::kFloat),
          compiler(std::make_shared<Compiler>(spec)) {
        TORCH_CHECK(index.is_cuda() && index.scalar_type() == at::kInt &&
                    index.is_contiguous() && index.numel() == dim && dim > 0 && channels > 0,
                    "Invalid native modulation metadata");
    }

    void check(const Tensor& x, const Tensor& g, const Tensor& r) const {
        TORCH_CHECK(x.is_cuda() && x.device() == index.device() &&
                    (x.scalar_type() == at::kFloat || x.scalar_type() == at::kDouble),
                    "Native modulation requires CUDA FP32/FP64");
        TORCH_CHECK(x.scalar_type() == dtype,
                    "Native modulation dtype does not match its prepared compiler");
        TORCH_CHECK(x.dim() > 0 && x.size(-1) == dim && x.sizes() == r.sizes(),
                    "Native modulation feature shape mismatch");
        TORCH_CHECK(g.dim() == x.dim() && g.size(-1) == channels,
                    "Native modulation gate shape mismatch");
        for (int64_t k = 0; k + 1 < x.dim(); ++k)
            TORCH_CHECK(x.size(k) == g.size(k), "Native modulation leading shape mismatch");
        TORCH_CHECK(g.device() == x.device() && r.device() == x.device() &&
                    g.scalar_type() == x.scalar_type() && r.scalar_type() == x.scalar_type(),
                    "Native modulation operands must share device and dtype");
    }

    // All live pointers are packed on every call. Only their alignment is part
    // of the executable key; no tensor address or parameter value is retained.
    Executable prepare(unsigned stage, const Values& arguments,
                       unsigned requested, unsigned seeds,
                       std::vector<uint64_t>& pointers) {
        uint64_t alignment = 0;
        for (size_t k = 0; k < arguments.size(); ++k) {
            const auto pointer = reinterpret_cast<uint64_t>(arguments[k].data_ptr());
            pointers.push_back(pointer);
            alignment |= uint64_t(pointer % 16 == 0) << k;
        }
        const uint64_t key = stage | (uint64_t(requested) << 2) |
            (uint64_t(seeds) << 5) | (alignment << 7);
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            const auto found = cache.find(key);
            if (found != cache.end()) return found->second;
        }
        TORCH_CHECK(c10::cuda::currentStreamCaptureStatusMayInitCtx() ==
                        c10::cuda::CaptureStatus::None,
                    "Warm FastEq fused operations and derivatives before graph capture");
        Executable executable;
        {
            // Never hold cache_mutex while waiting for the GIL. Another thread
            // may already own the GIL while invoking the public plan.
            py::gil_scoped_acquire gil;
            py::object prepared = compiler->borrow().attr("prepare")(
                stage, arguments, requested, seeds);
            if (!prepared.is_none()) {
                const auto tuple = prepared.cast<py::tuple>();
                TORCH_CHECK(tuple.size() == 3, "Invalid modulation executable");
                executable.kernel = tuple[0].cast<std::shared_ptr<Kernel>>();
                executable.threads = tuple[1].cast<unsigned>();
                executable.shared = tuple[2].cast<unsigned>();
            }
        }
        {
            std::lock_guard<std::mutex> lock(cache_mutex);
            cache.emplace(key, executable);
        }
        return executable;
    }

    void launch(const Executable& executable, std::vector<uint64_t>& pointers,
                int64_t rows) const {
        pointers.push_back(0);  // Triton's checked, unused global-scratch ABI slot.
        const auto stream = c10::cuda::getCurrentCUDAStream(index.get_device()).stream();
        executable.kernel->launch(pointers, rows, 1, executable.threads,
                                 reinterpret_cast<uint64_t>(stream), executable.shared);
    }

    Tensor forward(const Tensor& x, const Tensor& g, const Tensor& r) {
        check(x, g, r);
        c10::cuda::CUDAGuard guard(x.device());
        const bool tracked = torch::autograd::compute_requires_grad(x, g, r) &&
            !c10::InferenceMode::is_enabled();
        const auto rows = x.numel() / dim;
        Tensor output;
        bool fallback = false;
        {
            at::NoGradGuard no_grad;
            output = at::empty(x.sizes(), x.options());
            if (rows) {
                Values arguments{x.contiguous().view({rows, dim}), g.contiguous(),
                                 r.contiguous(), index, output};
                std::vector<uint64_t> pointers;
                auto executable = prepare(0, arguments, 0, 0, pointers);
                fallback = !executable.kernel;
                if (!fallback) launch(executable, pointers, rows);
            }
        }
        if (fallback) {
            py::gil_scoped_acquire gil;
            return compiler->borrow().attr("fallback_forward")(x, g, r).cast<Tensor>();
        }
        if (tracked) torch::autograd::set_history(output,
            std::make_shared<ModulationBackward>(shared_from_this(), x, g, r));
        return output;
    }

    size_t cached_executables() {
        std::lock_guard<std::mutex> lock(cache_mutex);
        size_t count = 0;
        for (const auto& entry : cache) count += bool(entry.second.kernel);
        return count;
    }
};

std::array<Tensor, 2> first(const std::shared_ptr<ModulationPlan>& plan,
                          const Tensor& x, const Tensor& g, const Tensor& dy,
                          unsigned requested) {
    c10::cuda::CUDAGuard guard(x.device());
    const bool tracked = torch::autograd::compute_requires_grad(x, g, dy) &&
        !c10::InferenceMode::is_enabled();
    std::array<Tensor, 2> outputs;
    const auto rows = x.numel() / plan->dim;
    bool fallback = false;
    {
        at::NoGradGuard no_grad;
        if (requested & 1) outputs[0] = at::empty(x.sizes(), x.options());
        if (requested & 2) outputs[1] = at::empty(g.sizes(), g.options());
        if (rows) {
            Values arguments{x.contiguous().view({rows, plan->dim}), g.contiguous(),
                dy.contiguous(), plan->index, outputs[0].defined() ? outputs[0] : x,
                outputs[1].defined() ? outputs[1] : g};
            std::vector<uint64_t> pointers;
            auto executable = plan->prepare(1, arguments, requested, 0, pointers);
            fallback = !executable.kernel;
            if (!fallback) plan->launch(executable, pointers, rows);
        }
    }
    if (fallback) {
        py::gil_scoped_acquire gil;
        py::tuple values = plan->compiler->borrow().attr("fallback_first")(
            x, g, dy, requested).cast<py::tuple>();
        for (size_t k = 0; k < 2; ++k)
            if (!values[k].is_none()) outputs[k] = values[k].cast<Tensor>();
        return outputs;
    }
    if (tracked) {
        auto node = std::make_shared<ModulationDouble>(plan, x, g, dy);
        // Undefined DX or DG still occupies its output_nr, matching Python's
        // set_materialize_grads(False) and the order of the two seed slots.
        torch::autograd::set_history(Values{outputs[0], outputs[1]}, node);
    }
    return outputs;
}

Values ModulationDouble::apply(Values&& seeds) {
    Values outputs(3);
    const Tensor u = seeds.size() > 0 ? seeds[0] : Tensor();
    const Tensor v = seeds.size() > 1 ? seeds[1] : Tensor();
    const unsigned requested = unsigned(task_should_compute_output(0) && v.defined()) |
        (unsigned(task_should_compute_output(1) && u.defined()) << 1) |
        (unsigned(task_should_compute_output(2)) << 2);
    if (!requested) return outputs;
    std::array<Tensor, 3> originals;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (size_t k = 0; k < 3; ++k) originals[k] = saved[k].unpack();
    }
    const auto& x = originals[0];
    const auto& g = originals[1];
    const auto& dy = originals[2];
    const auto rows = x.numel() / plan->dim;
    const bool third_order = at::GradMode::is_enabled() &&
        ((u.defined() && u.requires_grad()) || (v.defined() && v.requires_grad()));
    c10::cuda::CUDAGuard guard(x.device());
    {
        at::NoGradGuard no_grad;
        for (size_t k = 0; k < 3; ++k)
            if (requested & (1 << k))
                outputs[k] = at::empty(originals[k].sizes(), originals[k].options());
        if (rows) {
            const unsigned present = unsigned(u.defined()) | (unsigned(v.defined()) << 1);
            Values arguments{x.contiguous().view({rows, plan->dim}), g.contiguous(),
                dy.contiguous(), u.defined() ? u.contiguous() : x,
                v.defined() ? v.contiguous() : g, plan->index};
            for (size_t k = 0; k < 3; ++k)
                arguments.push_back(outputs[k].defined() ? outputs[k] : originals[k]);
            std::vector<uint64_t> pointers;
            auto executable = plan->prepare(2, arguments, requested, present, pointers);
            if (executable.kernel) plan->launch(executable, pointers, rows);
            else {
                py::gil_scoped_acquire gil;
                plan->compiler->borrow().attr("fallback_double")(arguments, requested, present);
            }
        }
    }
    if (third_order) {
        auto error = std::make_shared<torch::autograd::Error>(
            "trying to differentiate twice a function that was marked with @once_differentiable");
        torch::autograd::set_history(outputs, error);
    }
    return outputs;
}
}  // namespace

void bind_modulation(py::module_& module) {
    py::class_<ModulationPlan, std::shared_ptr<ModulationPlan>>(module, "ModulationPlan")
        .def(py::init<Tensor, int64_t, int64_t, const py::object&>())
        .def("forward", &ModulationPlan::forward, py::call_guard<py::gil_scoped_release>())
        .def_property_readonly("cached_executables", &ModulationPlan::cached_executables);
}
