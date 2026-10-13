// FastEq CUDA JIT infrastructure. GPU algorithms live in FastEq operator files.
#pragma once
#include <cuda.h>
#include <cstdint>
#include <string>
#include <vector>

namespace fasteq_jit {
class Kernel {
    CUmodule module_ = nullptr;
    CUfunction function_ = nullptr;
    CUcontext context_ = nullptr;
public:
    int registers = 0;
    int local_bytes = 0;
    Kernel(const std::string& binary, const std::string& symbol);
    ~Kernel();
    Kernel(const Kernel&) = delete;
    Kernel& operator=(const Kernel&) = delete;
    int active_blocks(unsigned threads, unsigned shared_bytes) const;
    void launch(const std::vector<uint64_t>& arguments, unsigned gx, unsigned gy,
                unsigned threads, uint64_t stream, unsigned shared_bytes = 0) const;
};
}  // namespace fasteq_jit
