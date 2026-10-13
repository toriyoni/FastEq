// FastEq CUDA JIT compiler and driver bridge. No model or EQX dependencies.
#include "jit_runtime.h"
#include <nvrtc.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <memory>
#include <stdexcept>
#include <tuple>

namespace py = pybind11;
namespace fasteq_jit {
void driver_check(CUresult result) {
    if (result == CUDA_SUCCESS) return;
    const char* detail = nullptr;
    cuGetErrorString(result, &detail);
    throw std::runtime_error(std::string("FastEq CUDA: ") + (detail ? detail : "driver failure"));
}

class ContextScope {
    bool pushed_;
public:
    explicit ContextScope(CUcontext wanted) {
        CUcontext active = nullptr;
        driver_check(cuCtxGetCurrent(&active));
        pushed_ = active != wanted;
        if (pushed_) driver_check(cuCtxPushCurrent(wanted));
    }
    ~ContextScope() {
        if (pushed_) {
            CUcontext ignored;
            cuCtxPopCurrent(&ignored);
        }
    }
};

Kernel::Kernel(const std::string& binary, const std::string& symbol) {
    driver_check(cuCtxGetCurrent(&context_));
    driver_check(cuModuleLoadData(&module_, binary.data()));
    try {
        driver_check(cuModuleGetFunction(&function_, module_, symbol.c_str()));
        driver_check(cuFuncGetAttribute(&registers, CU_FUNC_ATTRIBUTE_NUM_REGS, function_));
        driver_check(cuFuncGetAttribute(&local_bytes, CU_FUNC_ATTRIBUTE_LOCAL_SIZE_BYTES, function_));
    } catch (...) {
        cuModuleUnload(module_);
        throw;
    }
}

Kernel::~Kernel() {
    // CUDA shutdown may already have destroyed the context.
    if (module_ && cuCtxPushCurrent(context_) == CUDA_SUCCESS) {
        cuModuleUnload(module_);
        CUcontext ignored;
        cuCtxPopCurrent(&ignored);
    }
}

int Kernel::active_blocks(unsigned threads, unsigned shared_bytes) const {
    ContextScope scope(context_);
    int count = 0;
    driver_check(cuOccupancyMaxActiveBlocksPerMultiprocessor(&count, function_, threads, shared_bytes));
    return count;
}

void Kernel::launch(const std::vector<uint64_t>& arguments, unsigned gx, unsigned gy,
                    unsigned threads, uint64_t stream, unsigned shared_bytes) const {
    if (!gx || !gy) return;
    ContextScope scope(context_);
    std::vector<void*> pointers;
    pointers.reserve(arguments.size());
    for (const auto& value : arguments) pointers.push_back(const_cast<uint64_t*>(&value));
    driver_check(cuLaunchKernel(function_, gx, gy, 1, threads, 1, 1, shared_bytes,
                               reinterpret_cast<CUstream>(stream), pointers.data(), nullptr));
}

struct Compilation {
    nvrtcProgram program = nullptr;
    ~Compilation() { if (program) nvrtcDestroyProgram(&program); }
};

std::string compile_source(const std::string& source, const std::vector<std::string>& options) {
    Compilation compilation;
    auto result = nvrtcCreateProgram(&compilation.program, source.c_str(), "fasteq.cu", 0, nullptr, nullptr);
    if (result != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(result));
    std::vector<const char*> args;
    for (const auto& option : options) args.push_back(option.c_str());
    result = nvrtcCompileProgram(compilation.program, args.size(), args.data());
    if (result != NVRTC_SUCCESS) {
        size_t size = 0;
        nvrtcGetProgramLogSize(compilation.program, &size);
        std::string log(size, '\0');
        nvrtcGetProgramLog(compilation.program, log.data());
        throw std::runtime_error("FastEq CUDA compilation failed:\n" + log);
    }
    size_t size = 0;
    result = nvrtcGetCUBINSize(compilation.program, &size);
    if (result != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(result));
    std::string binary(size, '\0');
    result = nvrtcGetCUBIN(compilation.program, binary.data());
    if (result != NVRTC_SUCCESS) throw std::runtime_error(nvrtcGetErrorString(result));
    return binary;
}
}  // namespace fasteq_jit

#ifdef FASTEQ_INDEXED_LINEAR
void bind_indexed_linear(py::module_&);
#endif
#ifdef FASTEQ_NORM
void bind_norm(py::module_&);
#endif
#ifdef FASTEQ_MODULATION
void bind_modulation(py::module_&);
#endif
#ifdef FASTEQ_SEGMENTED_PRODUCT
void bind_segmented_product(py::module_&);
#endif

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    using fasteq_jit::Kernel;
    module.def("compile", [](const std::string& source, const std::vector<std::string>& options) {
        std::string result;
        { py::gil_scoped_release release; result = fasteq_jit::compile_source(source, options); }
        return py::bytes(result);
    });
    module.def("version", []() {
        int major = 0, minor = 0;
        nvrtcVersion(&major, &minor);
        return py::make_tuple(major, minor);
    });
    py::class_<Kernel, std::shared_ptr<Kernel>>(module, "Kernel")
        .def(py::init([](py::bytes binary, const std::string& symbol) {
            return std::make_shared<Kernel>(static_cast<std::string>(binary), symbol);
        }))
        .def_readonly("registers", &Kernel::registers)
        .def_readonly("local_bytes", &Kernel::local_bytes)
        .def("active_blocks", &Kernel::active_blocks)
        .def("launch", &Kernel::launch, py::arg("arguments"), py::arg("gx"), py::arg("gy"),
             py::arg("threads"), py::arg("stream"), py::arg("shared_bytes") = 0,
             py::call_guard<py::gil_scoped_release>());
    using Job = std::tuple<std::shared_ptr<Kernel>, std::vector<uint64_t>, unsigned, unsigned, unsigned, unsigned>;
    module.def("launch", [](const std::vector<Job>& jobs, uint64_t stream) {
        py::gil_scoped_release release;
        for (const auto& [kernel, args, gx, gy, threads, shared] : jobs)
            kernel->launch(args, gx, gy, threads, stream, shared);
    });
#ifdef FASTEQ_INDEXED_LINEAR
    bind_indexed_linear(module);
#endif
#ifdef FASTEQ_NORM
    bind_norm(module);
#endif
#ifdef FASTEQ_MODULATION
    bind_modulation(module);
#endif
#ifdef FASTEQ_SEGMENTED_PRODUCT
    bind_segmented_product(module);
#endif
}
