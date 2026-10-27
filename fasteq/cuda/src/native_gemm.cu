// FastEq's full-precision matrix primitive.
//
// FP32 extends indexed_linear.cu ROLE8--10's shared-memory SIMT GEMM:
// each warp owns output rows and each lane owns adjacent column groups.
// FP64 extends equi_linear_fwd_f64.cu's 2-stage shared swizzle and original
// m16n8k4 MMA instruction. General strides/tails replace model-specific offsets.
// No cuBLAS, CUTLASS, ATen matrix operator, or borrowed CUDA implementation.
#include "jit_runtime.h"
#include "equi_linear_kernels/matrix_tiles.cuh"
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>
#include <c10/cuda/CUDAException.h>
#include <ATen/ATen.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cstdlib>
#include <cstring>

namespace fasteq_jit {
namespace {
using matrix_device::MatrixShape;
using matrix_device::matrix_mma_f64;
using matrix_device::matrix_mma_promoted_f32;
using matrix_device::matrix_simt_f32;
using matrix_device::matrix_tf32_f32;

template<class T>
__global__ void matrix_partial_reduce(const T* partial, T* c, MatrixShape s,
                                     int64_t batch, int splits, T alpha, T beta) {
    const int64_t total = batch * s.m * s.n;
    for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x;
         index < total; index += gridDim.x * blockDim.x) {
        T value = 0;
        for (int split = 0; split < splits; ++split) value += partial[split * total + index];
        const int64_t matrix = index / (s.m * s.n), row = (index / s.n) % s.m, col = index % s.n;
        T* dst = c + matrix * s.cb + row * s.cr + col * s.cc;
        *dst = alpha * value + (beta == T(0) ? T(0) : beta * *dst);
    }
}

template<int TM, int TN, int TK, bool AK, bool BN>
void launch_simt(at::Tensor& c, const at::Tensor& a, const at::Tensor& b,
                 MatrixShape s, int64_t batch, double alpha, double beta,
                 int splits, int64_t span, cudaStream_t stream) {
    const int64_t tiles_m = (s.m + TM - 1) / TM;
    const dim3 grid((s.n + TN - 1) / TN, tiles_m * batch, splits);
    TORCH_CHECK(tiles_m * batch <= 65535, "FastEq native GEMM row/batch grid too large");
    matrix_simt_f32<TM, TN, TK, AK, BN><<<grid, 256, 0, stream>>>(
        a.const_data_ptr<float>(), b.const_data_ptr<float>(), c.mutable_data_ptr<float>(),
        s, alpha, beta, tiles_m, span);
}

template<bool AK, bool BN>
void launch_matrix(at::Tensor& destination, const at::Tensor& a, const at::Tensor& b,
                   MatrixShape original, int64_t batch, double alpha, double beta) {
    MatrixShape s = original;
    at::Tensor c = destination;
    // The original Linear dW uses bounded ordered partials. Extend that
    // strategy to general transposes with long K and too few output tiles.
    // Splits are fixed contiguous K intervals, never floating-point atomics.
    const bool fp64 = c.scalar_type() == at::kDouble;
    const char* tf32_opt = std::getenv("FASTEQ_EXPERIMENTAL_MATRIX_TF32X3");
    const bool tf32x3 = !fp64 && tf32_opt && std::strcmp(tf32_opt, "1") == 0;
    const char* promote_opt = std::getenv("FASTEQ_EXPERIMENTAL_MATRIX_F32_F64MMA");
    const bool promote = !fp64 && promote_opt && std::strcmp(promote_opt, "1") == 0;
    const int tile_m = fp64 || tf32x3 || promote ? 64 : 128;
    const int tile_n = fp64 || tf32x3 || promote || s.n < 128 ? 64 : 128;
    const int64_t blocks = ((s.m + tile_m - 1) / tile_m)
        * ((s.n + tile_n - 1) / tile_n) * batch;
    const int target_blocks = at::cuda::getDeviceProperties(c.get_device())->multiProcessorCount * 4;
    int splits = s.k >= 1024 && blocks < target_blocks
        ? int((target_blocks + blocks - 1) / blocks) : 1;
    splits = splits > 16 ? 16 : splits;
    // Keep each K interval useful and cap this primitive's temporary at 64 MiB.
    // A caller's own scratch is separate (e.g. TP's outer tiles add another 64 MiB).
    const int64_t memory_splits = (64LL << 20) / (batch * s.m * s.n * c.element_size());
    const int64_t reduction_splits = (s.k + 255) / 256;
    splits = splits > memory_splits ? int(memory_splits) : splits;
    splits = splits > reduction_splits ? int(reduction_splits) : splits;
    splits = splits < 1 ? 1 : splits;
    const int64_t span = splits > 1 ? ((s.k + splits * 16 - 1) / (splits * 16)) * 16 : 0;
    if (splits > 1) {
        c = at::empty({splits, batch, s.m, s.n}, destination.options());
        s.cr = s.n; s.cc = 1; s.cb = s.m * s.n;
    }
    const auto stream = c10::cuda::getCurrentCUDAStream(c.get_device());
    if (c.scalar_type() == at::kDouble) {
        constexpr int TM = 64, TN = 64;
        const int64_t tiles_m = (s.m + TM - 1) / TM;
        const dim3 grid((s.n + TN - 1) / TN, tiles_m * batch, splits);
        TORCH_CHECK(tiles_m * batch <= 65535, "FastEq native GEMM row/batch grid too large");
        matrix_mma_f64<TM, TN, AK, BN><<<grid, 128, 0, stream>>>(
            a.const_data_ptr<double>(), b.const_data_ptr<double>(), c.mutable_data_ptr<double>(),
            s, splits > 1 ? 1.0 : alpha, splits > 1 ? 0.0 : beta, tiles_m, span);
    } else {
        const double scale = splits > 1 ? 1 : alpha, old = splits > 1 ? 0 : beta;
        if (promote) {
            const int64_t tiles_m = (s.m + 63) / 64;
            const dim3 grid((s.n + 63) / 64, tiles_m * batch, splits);
            TORCH_CHECK(tiles_m * batch <= 65535, "FastEq native GEMM row/batch grid too large");
            matrix_mma_promoted_f32<64, 64, AK, BN><<<grid, 128, 0, stream>>>(
                a.const_data_ptr<float>(), b.const_data_ptr<float>(), c.mutable_data_ptr<float>(),
                s, scale, old, tiles_m, span);
        } else if (tf32x3) {
            const int64_t tiles_m = (s.m + 63) / 64;
            const dim3 grid((s.n + 63) / 64, tiles_m * batch, splits);
            TORCH_CHECK(tiles_m * batch <= 65535, "FastEq native GEMM row/batch grid too large");
            matrix_tf32_f32<64, 64, 32, AK, BN><<<grid, 128, 0, stream>>>(
                a.const_data_ptr<float>(), b.const_data_ptr<float>(), c.mutable_data_ptr<float>(),
                s, scale, old, tiles_m, span);
        } else if (s.n >= 128)
            launch_simt<128, 128, 16, AK, BN>(c, a, b, s, batch, scale, old, splits, span, stream);
        else launch_simt<128, 64, 16, AK, BN>(c, a, b, s, batch, scale, old, splits, span, stream);
    }
    if (splits > 1) {
        const int64_t total = batch * s.m * s.n;
        const unsigned blocks = unsigned((total + 255) / 256 > 4096 ? 4096 : (total + 255) / 256);
        if (c.scalar_type() == at::kDouble)
            matrix_partial_reduce<double><<<blocks, 256, 0, stream>>>(c.const_data_ptr<double>(),
                destination.mutable_data_ptr<double>(), original, batch, splits, alpha, beta);
        else matrix_partial_reduce<float><<<blocks, 256, 0, stream>>>(c.const_data_ptr<float>(),
                destination.mutable_data_ptr<float>(), original, batch, splits, alpha, beta);
    }
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}
}  // namespace

void native_gemm_into(at::Tensor& destination, const at::Tensor& left,
                      const at::Tensor& right, double alpha, double beta) {
    const int64_t dims = destination.dim();
    TORCH_CHECK(dims == 2 || dims == 3, "FastEq native GEMM expects rank 2 or 3");
    TORCH_CHECK(left.dim() == dims && right.dim() == dims, "FastEq native GEMM rank mismatch");
    TORCH_CHECK(destination.is_cuda() && left.device() == destination.device() && right.device() == destination.device(),
                "FastEq native GEMM operands must share a CUDA device");
    TORCH_CHECK(destination.scalar_type() == left.scalar_type() && left.scalar_type() == right.scalar_type()
                && (left.scalar_type() == at::kFloat || left.scalar_type() == at::kDouble),
                "FastEq native GEMM supports matching FP32 or FP64 operands");
    const int64_t batch = dims == 3 ? destination.size(0) : 1;
    const int axis = dims - 2;
    MatrixShape s{destination.size(axis), destination.size(axis + 1), left.size(axis + 1),
        left.stride(axis), left.stride(axis + 1), dims == 3 ? left.stride(0) : 0,
        right.stride(axis), right.stride(axis + 1), dims == 3 ? right.stride(0) : 0,
        destination.stride(axis), destination.stride(axis + 1), dims == 3 ? destination.stride(0) : 0};
    TORCH_CHECK(left.size(axis) == s.m && right.size(axis) == s.k && right.size(axis + 1) == s.n,
                "FastEq native GEMM matrix shape mismatch");
    TORCH_CHECK(dims == 2 || (left.size(0) == batch && right.size(0) == batch),
                "FastEq native GEMM batch shape mismatch");
    if (!s.m || !s.n || !batch) return;
    const c10::cuda::CUDAGuard guard(destination.device());
    if (alpha == 0) {
        // addmm semantics: alpha=0 ignores A/B, including their NaNs; beta=0
        // likewise ignores the old destination. No matrix backend is involved.
        const int64_t total = batch * s.m * s.n;
        const unsigned blocks = unsigned((total + 255) / 256 > 4096 ? 4096 : (total + 255) / 256);
        const auto stream = c10::cuda::getCurrentCUDAStream(destination.get_device());
        if (destination.scalar_type() == at::kDouble)
            matrix_partial_reduce<double><<<blocks, 256, 0, stream>>>(nullptr,
                destination.mutable_data_ptr<double>(), s, batch, 0, 0.0, beta);
        else matrix_partial_reduce<float><<<blocks, 256, 0, stream>>>(nullptr,
                destination.mutable_data_ptr<float>(), s, batch, 0, 0.0f, beta);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
        return;
    }
    // Keep the original CUDA tiles and their ordered reductions. Very large
    // row/batch domains exceed grid.y even when their tensors still fit in
    // memory; partition metadata views rather than reject a legal matrix.
    const bool fp64 = destination.scalar_type() == at::kDouble;
    const char* promote_option = std::getenv("FASTEQ_EXPERIMENTAL_MATRIX_F32_F64MMA");
    const char* tf32_option = std::getenv("FASTEQ_EXPERIMENTAL_MATRIX_TF32X3");
    const bool wide_tile = fp64 || (promote_option && std::strcmp(promote_option, "1") == 0)
        || (tf32_option && std::strcmp(tf32_option, "1") == 0);
    const int64_t tile_rows = wide_tile ? 64 : 128;
    const int64_t row_tiles = (s.m + tile_rows - 1) / tile_rows;
    if (row_tiles * batch > 65535) {
        const int64_t row_chunk = std::min<int64_t>(s.m, 65535 * tile_rows);
        const int64_t chunk_tiles = (row_chunk + tile_rows - 1) / tile_rows;
        const int64_t batch_chunk = std::max<int64_t>(1, 65535 / chunk_tiles);
        for (int64_t first_batch = 0; first_batch < batch; first_batch += batch_chunk) {
            const int64_t count = std::min(batch_chunk, batch - first_batch);
            const auto a_batch = dims == 3 ? left.narrow(0, first_batch, count) : left;
            const auto b_batch = dims == 3 ? right.narrow(0, first_batch, count) : right;
            const auto c_batch = dims == 3 ? destination.narrow(0, first_batch, count) : destination;
            for (int64_t first_row = 0; first_row < s.m; first_row += row_chunk) {
                const int64_t rows = std::min(row_chunk, s.m - first_row);
                auto c_rows = c_batch.narrow(axis, first_row, rows);
                const auto a_rows = a_batch.narrow(axis, first_row, rows);
                native_gemm_into(c_rows, a_rows, b_batch, alpha, beta);
            }
        }
        return;
    }
    if (s.ac == 1) {
        if (s.bc == 1) launch_matrix<true, true>(destination, left, right, s, batch, alpha, beta);
        else launch_matrix<true, false>(destination, left, right, s, batch, alpha, beta);
    } else {
        if (s.bc == 1) launch_matrix<false, true>(destination, left, right, s, batch, alpha, beta);
        else launch_matrix<false, false>(destination, left, right, s, batch, alpha, beta);
    }
}
}  // namespace fasteq_jit
