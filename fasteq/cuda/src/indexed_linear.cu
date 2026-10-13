// FastEq general indexed equivariant Linear.
//
// Metadata-driven kernels: four angular warps share a tile of channel weights.
// Input gradients transpose dense weight tiles through padded shared memory;
// Weight gradients split the node reduction into independent chunks, with
// their node span increased by the host plan to bound temporary storage.
// This extends the MACE-specific dense Linear kernels without their fixed
// l=0..3 output or path-count restrictions. No external backend is called.

using scalar = FASTEQ_LINEAR_SCALAR;

__device__ __forceinline__ long long feature_offset(
    int base, int channels, int dim, int expert, int channel, int angle,
    int experts, int ir_mul) {
    return base + (ir_mul ? (long long)angle * experts * channels + expert * channels + channel
                         : (long long)(expert * channels + channel) * dim + angle);
}

// paths: [sx, sw, sy, ci, co, angular_dim]
// groups: [offset, channels, angular_dim, path_ids_start, path_count]
// tiles: [group, expert, channel_start, angular_start] (role 0/2),
//        [group, expert, weight_cell_start, 0] (role 1/3).
extern "C" __global__ void run(
    const scalar* x, const scalar* weight, const scalar* y,
    const long long* types, scalar* result,
    const int* paths, const scalar* scales, const int* groups,
    const int* path_ids, const int* tiles,
    long long nodes, long long din, long long dout, long long numel,
    int experts, int ir_mul, int elements, int tile_count, int chunks, long long chunk_span) {
#if FASTEQ_LINEAR_ROLE >= 8 && FASTEQ_LINEAR_ROLE <= 10
    // Full-precision FP32 SIMT GEMM. A 32x32 output tile reuses 32-wide
    // A/B reduction tiles in padded shared memory. Four warps each own
    // eight rows and one column per lane; no TF32 conversion is involved.
    // ROLE8: X W; ROLE9: Y W^T; ROLE10: chunked X^T Y.
    const int* tile = tiles + blockIdx.y * 4;
    const int* group = groups + tile[0] * 5;
    const int expert = tile[1], lane = threadIdx.x & 31, warp = threadIdx.x / 32;
#if FASTEQ_LINEAR_ROLE == 10
    const long long first_row = tile[2], row_count = group[1];
    const int first_col = tile[3], col_count = group[2];
#else
    const long long first_row = (long long)blockIdx.x * 32, row_count = nodes * group[2];
    const int first_col = tile[2], col_count = group[1];
#endif
    if (first_row >= row_count) return; // uniform whole-block predicate
    __shared__ scalar a_tile[32][33];
    __shared__ scalar b_tile[32][33];
    scalar total[8] = {0, 0, 0, 0, 0, 0, 0, 0};
    for (int j = 0; j < group[4]; ++j) {
        const int p = path_ids[group[3] + j];
        const int* path = paths + p * 6;
#if FASTEQ_LINEAR_ROLE == 10
        const long long reduction = chunk_span * path[5];
#elif FASTEQ_LINEAR_ROLE == 9
        const long long reduction = path[4];
#else
        const long long reduction = path[3];
#endif
        scalar partial[8] = {0, 0, 0, 0, 0, 0, 0, 0};
        for (long long k0 = 0; k0 < reduction; k0 += 32) {
            for (int cell = threadIdx.x; cell < 1024; cell += 128) {
                const int r = cell / 32, c = cell % 32;
#if FASTEQ_LINEAR_ROLE == 10
                // Load X along its channel axis, then transpose into A.
                const long long k = k0 + r;
                const long long node = (long long)blockIdx.x * chunk_span + k / path[5];
                const int angle = k % path[5], channel = first_row + c;
                a_tile[c][r] = k < reduction && node < nodes && channel < path[3]
                    ? x[node * din + feature_offset(path[0], path[3], path[5], expert,
                                                   channel, angle, experts, ir_mul)] : scalar(0);
                const int column = first_col + c;
                b_tile[r][c] = k < reduction && node < nodes && column < path[4]
                    ? y[node * dout + feature_offset(path[2], path[4], path[5], expert,
                                                    column, angle, experts, ir_mul)] : scalar(0);
#else
                const long long row = first_row + r, k = k0 + c;
                const long long node = row / group[2];
                const int angle = row % group[2];
#if FASTEQ_LINEAR_ROLE == 9
                a_tile[r][c] = row < row_count && k < reduction
                    ? y[node * dout + feature_offset(path[2], path[4], path[5], expert,
                                                    k, angle, experts, ir_mul)] : scalar(0);
                // Coalesced weight rows become columns in the B tile.
                const int channel = first_col + r;
                b_tile[c][r] = channel < path[3] && k < reduction
                    ? weight[(long long)expert * numel + path[1] + (long long)channel * path[4] + k]
                    : scalar(0);
#else
                a_tile[r][c] = row < row_count && k < reduction
                    ? x[node * din + feature_offset(path[0], path[3], path[5], expert,
                                                   k, angle, experts, ir_mul)] : scalar(0);
                const long long u = k0 + r;
                const int column = first_col + c;
                b_tile[r][c] = u < reduction && column < path[4]
                    ? weight[(long long)expert * numel + path[1] + u * path[4] + column]
                    : scalar(0);
#endif
#endif
            }
            __syncthreads();
            #pragma unroll
            for (int k = 0; k < 32; ++k) {
                const scalar b = b_tile[k][lane];
                #pragma unroll
                for (int i = 0; i < 8; ++i)
                    partial[i] += a_tile[warp + 4 * i][k] * b;
            }
            __syncthreads();
        }
        #pragma unroll
        for (int i = 0; i < 8; ++i) total[i] += scales[p] * partial[i];
    }
    const int column = first_col + lane;
    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        const long long row = first_row + warp + 4 * i;
        if (row < row_count && column < col_count) {
#if FASTEQ_LINEAR_ROLE == 10
            result[((long long)blockIdx.x * experts + expert) * numel + group[0]
                   + row * group[2] + column] = total[i];
#else
            const auto offset = feature_offset(group[0], group[1], group[2], expert,
                                               column, row % group[2], experts, ir_mul);
            result[(row / group[2]) * (FASTEQ_LINEAR_ROLE == 9 ? din : dout) + offset] = total[i];
#endif
        }
    }
#elif FASTEQ_LINEAR_ROLE == 6
    // Shared-weight dX: Y [node*angle, co] times W^T [co, ci].
    // Reuse the same FastEq FP64 MMA primitive as forward, with a
    // transposed weight load and input-channel output tile metadata.
    const int* tile = tiles + blockIdx.y * 4;
    const int* group = groups + tile[0] * 5;
    const int lane = threadIdx.x & 31;
    const int expert = tile[1];
    const long long first = ((long long)blockIdx.x * 4 + threadIdx.x / 32) * 16;
    if (first >= nodes * group[2]) return;
    const long long row0 = first + lane / 4, row1 = row0 + 8;
    const int col = tile[2] + lane / 4;
    double total[4] = {0, 0, 0, 0};
    for (int j = 0; j < group[4]; ++j) {
        const int p = path_ids[group[3] + j];
        const int* path = paths + p * 6;
        double partial[4] = {0, 0, 0, 0};
        for (int k = 0; k < path[4]; k += 4) {
            const int v = k + lane % 4;
            double a0 = 0, a1 = 0, b = 0;
            if (v < path[4]) {
                if (row0 < nodes * group[2]) {
                    const auto offset = feature_offset(path[2], path[4], path[5], expert, v,
                                                       row0 % group[2], experts, ir_mul);
                    a0 = y[(row0 / group[2]) * dout + offset];
                }
                if (row1 < nodes * group[2]) {
                    const auto offset = feature_offset(path[2], path[4], path[5], expert, v,
                                                       row1 % group[2], experts, ir_mul);
                    a1 = y[(row1 / group[2]) * dout + offset];
                }
                if (col < path[3]) b = weight[(long long)expert * numel + path[1] + (long long)col * path[4] + v];
            }
            asm_mma_m16n8k4_f64_f64_f64_f64(partial[0], partial[1], partial[2], partial[3],
                a0, a1, b, partial[0], partial[1], partial[2], partial[3]);
        }
        for (int i = 0; i < 4; ++i) total[i] += scales[p] * partial[i];
    }
    const int col0 = tile[2] + (lane % 4) * 2;
    for (int i = 0; i < 4; ++i) {
        const long long row = i < 2 ? row0 : row1;
        const int column = col0 + i % 2;
        if (row < nodes * group[2] && column < group[1]) {
            const auto offset = feature_offset(group[0], group[1], group[2], expert, column,
                                               row % group[2], experts, ir_mul);
            result[(row / group[2]) * din + offset] = total[i];
        }
    }
#elif FASTEQ_LINEAR_ROLE == 7
    // Shared-weight dW: X^T [ci, node*angle] times Y [node*angle, co].
    // Four independent warps cover 64 input channels by 8 output channels.
    // Node chunks write disjoint partials; ROLE3 preserves the existing
    // bounded scratch allocation and deterministic final chunk reduction.
    const int* tile = tiles + blockIdx.y * 4;
    const int* group = groups + tile[0] * 5;
    const int lane = threadIdx.x & 31;
    const int expert = tile[1];
    const int first = tile[2] + (threadIdx.x / 32) * 16;
    if (first >= group[1]) return;
    const int row0 = first + lane / 4, row1 = row0 + 8;
    const int col = tile[3] + lane / 4;
    double total[4] = {0, 0, 0, 0};
    for (int j = 0; j < group[4]; ++j) {
        const int p = path_ids[group[3] + j];
        const int* path = paths + p * 6;
        double partial[4] = {0, 0, 0, 0};
        for (long long k = 0; k < chunk_span * path[5]; k += 4) {
            const long long position = k + lane % 4;
            const long long node = (long long)blockIdx.x * chunk_span + position / path[5];
            const int angle = position % path[5];
            double a0 = 0, a1 = 0, b = 0;
            if (node < nodes && position < chunk_span * path[5]) {
                if (row0 < path[3]) a0 = x[node * din + feature_offset(
                    path[0], path[3], path[5], expert, row0, angle, experts, ir_mul)];
                if (row1 < path[3]) a1 = x[node * din + feature_offset(
                    path[0], path[3], path[5], expert, row1, angle, experts, ir_mul)];
                if (col < path[4]) b = y[node * dout + feature_offset(
                    path[2], path[4], path[5], expert, col, angle, experts, ir_mul)];
            }
            asm_mma_m16n8k4_f64_f64_f64_f64(partial[0], partial[1], partial[2], partial[3],
                a0, a1, b, partial[0], partial[1], partial[2], partial[3]);
        }
        for (int i = 0; i < 4; ++i) total[i] += scales[p] * partial[i];
    }
    const int col0 = tile[3] + (lane % 4) * 2;
    for (int i = 0; i < 4; ++i) {
        const int row = i < 2 ? row0 : row1, column = col0 + i % 2;
        if (row < group[1] && column < group[2])
            result[((long long)blockIdx.x * experts + expert) * numel + group[0]
                   + (long long)row * group[2] + column] = total[i];
    }
#elif FASTEQ_LINEAR_ROLE == 5
    // Reuse FastEq's existing FP64 m16n8k4 Tensor-Core primitive, while
    // generalizing its MACE-only mapping to arbitrary irrep metadata.
    // This path is selected only for shared element weights.
    const int* tile = tiles + blockIdx.y * 4;
    const int* group = groups + tile[0] * 5;
    const int lane = threadIdx.x & 31;
    const int expert = tile[1];
    const long long first = ((long long)blockIdx.x * 4 + threadIdx.x / 32) * 16;
    if (first >= nodes * group[2]) return;
    const long long row0 = first + lane / 4, row1 = row0 + 8;
    const int col = tile[2] + lane / 4;
    double total[4] = {0, 0, 0, 0};
    for (int j = 0; j < group[4]; ++j) {
        const int p = path_ids[group[3] + j];
        const int* path = paths + p * 6;
        double partial[4] = {0, 0, 0, 0};
        for (int k = 0; k < path[3]; k += 4) {
            const int u = k + lane % 4;
            double a0 = 0, a1 = 0, b = 0;
            if (u < path[3]) {
                if (row0 < nodes * group[2]) {
                    const auto offset = feature_offset(path[0], path[3], path[5], expert, u,
                                                       row0 % group[2], experts, ir_mul);
                    a0 = x[(row0 / group[2]) * din + offset];
                }
                if (row1 < nodes * group[2]) {
                    const auto offset = feature_offset(path[0], path[3], path[5], expert, u,
                                                       row1 % group[2], experts, ir_mul);
                    a1 = x[(row1 / group[2]) * din + offset];
                }
                if (col < path[4]) b = weight[(long long)expert * numel + path[1] + (long long)u * path[4] + col];
            }
            asm_mma_m16n8k4_f64_f64_f64_f64(partial[0], partial[1], partial[2], partial[3],
                a0, a1, b, partial[0], partial[1], partial[2], partial[3]);
        }
        // Preserve FP64 path coefficients: the legacy kernel stored cg_T<float>.
        for (int i = 0; i < 4; ++i) total[i] += scales[p] * partial[i];
    }
    const int col0 = tile[2] + (lane % 4) * 2;
    for (int i = 0; i < 4; ++i) {
        const long long row = i < 2 ? row0 : row1;
        const int column = col0 + i % 2;
        if (row < nodes * group[2] && column < group[1]) {
            const auto offset = feature_offset(group[0], group[1], group[2], expert, column,
                                               row % group[2], experts, ir_mul);
            result[(row / group[2]) * dout + offset] = total[i];
        }
    }
#elif FASTEQ_LINEAR_ROLE == 0 || FASTEQ_LINEAR_ROLE == 2
    const long long node = blockIdx.x;
    if (node >= nodes) return;
    const int* tile = tiles + blockIdx.y * 4;
    const int* group = groups + tile[0] * 5;
    const int expert = tile[1];
    const int channel = tile[2] + (threadIdx.x & 31);
    const int angle = tile[3] + threadIdx.x / 32;
    const long long element = types[node];
    scalar total = 0;
#if FASTEQ_LINEAR_ROLE == 0
    // Channel-contiguous global loads become channel-contiguous transpose
    // reads in shared memory; four angular warps reuse each weight tile.
    __shared__ scalar transposed[32][33];
    for (int j = 0; j < group[4]; ++j) {
        const int p = path_ids[group[3] + j];
        const int* path = paths + p * 6;
        scalar value = 0;
        for (int v0 = 0; v0 < path[4]; v0 += 32) {
            for (int cell = threadIdx.x; cell < 1024; cell += blockDim.x) {
                const int u = tile[2] + cell / 32;
                const int v = v0 + cell % 32;
                transposed[cell % 32][cell / 32] =
                    u < path[3] && v < path[4]
                    ? weight[(element * experts + expert) * numel + path[1]
                             + (long long)u * path[4] + v] : scalar(0);
            }
            __syncthreads();
            if (channel < group[1] && angle < group[2]) {
                const int end = min(32, path[4] - v0);
                for (int v = 0; v < end; ++v) {
                    const long long offset = feature_offset(
                        path[2], path[4], path[5], expert, v0 + v, angle, experts, ir_mul);
                    value += transposed[v][threadIdx.x & 31] * y[node * dout + offset];
                }
            }
            __syncthreads();
        }
        total += scales[p] * value;
    }
#else
    if (channel >= group[1] || angle >= group[2]) return;
    for (int j = 0; j < group[4]; ++j) {
        const int p = path_ids[group[3] + j];
        const int* path = paths + p * 6;
        scalar value = 0;
        for (int u = 0; u < path[3]; ++u) {
            const long long offset = feature_offset(
                path[0], path[3], path[5], expert, u, angle, experts, ir_mul);
            value += x[node * din + offset] *
                weight[(element * experts + expert) * numel + path[1]
                       + (long long)u * path[4] + channel];
        }
        total += scales[p] * value;
    }
#endif
    if (channel < group[1] && angle < group[2]) {
        const long long offset = feature_offset(
            group[0], group[1], group[2], expert, channel, angle, experts, ir_mul);
        result[node * (FASTEQ_LINEAR_ROLE == 0 ? din : dout) + offset] = total;
    }
#elif FASTEQ_LINEAR_ROLE == 1
    // Each thread owns one weight cell. There are no atomics: all chunks write
    // distinct partial reductions and the next phase reduces the chunk axis.
    const int chunk = blockIdx.x;
    const int element = blockIdx.y / tile_count;
    const int* tile = tiles + (blockIdx.y % tile_count) * 4;
    const int* group = groups + tile[0] * 5;
    const int cell = tile[2] + threadIdx.x;
    if (cell >= group[1] * group[2]) return;
    const int u = cell / group[2], v = cell % group[2];
    scalar total = 0;
    const long long first = (long long)chunk * chunk_span;
    for (int j = 0; j < group[4]; ++j) {
        const int p = path_ids[group[3] + j];
        const int* path = paths + p * 6;
        scalar value = 0;
        for (long long node = first; node < nodes && node < first + chunk_span; ++node) {
            if (types[node] != element) continue;
            for (int m = 0; m < path[5]; ++m) {
                const long long sx = feature_offset(path[0], path[3], path[5], tile[1], u, m, experts, ir_mul);
                const long long sy = feature_offset(path[2], path[4], path[5], tile[1], v, m, experts, ir_mul);
                value += x[node * din + sx] * y[node * dout + sy];
            }
        }
        total += scales[p] * value;
    }
    result[((long long)chunk * elements * experts + element * experts + tile[1])
           * numel + group[0] + cell] = total;
#elif FASTEQ_LINEAR_ROLE == 4
    // Many element classes (including per-row weights) use bounded storage:
    // each node contributes only to its selected element's weight matrix.
    const long long node = blockIdx.x;
    if (node >= nodes) return;
    const int* tile = tiles + blockIdx.y * 4;
    const int* group = groups + tile[0] * 5;
    const int cell = tile[2] + threadIdx.x;
    if (cell >= group[1] * group[2]) return;
    const int u = cell / group[2], v = cell % group[2];
    scalar total = 0;
    for (int j = 0; j < group[4]; ++j) {
        const int p = path_ids[group[3] + j];
        const int* path = paths + p * 6;
        scalar value = 0;
        for (int m = 0; m < path[5]; ++m) {
            const long long sx = feature_offset(path[0], path[3], path[5], tile[1], u, m, experts, ir_mul);
            const long long sy = feature_offset(path[2], path[4], path[5], tile[1], v, m, experts, ir_mul);
            value += x[node * din + sx] * y[node * dout + sy];
        }
        total += scales[p] * value;
    }
    atomicAdd(result + (types[node] * experts + tile[1]) * numel + group[0] + cell, total);
#else
    const int element = blockIdx.x / tile_count;
    const int* tile = tiles + (blockIdx.x % tile_count) * 4;
    const int* group = groups + tile[0] * 5;
    const int cell = tile[2] + threadIdx.x;
    if (cell >= group[1] * group[2]) return;
    const long long offset = (long long)(element * experts + tile[1]) * numel + group[0] + cell;
    scalar total = 0;
    for (int chunk = 0; chunk < chunks; ++chunk)
        total += x[(long long)chunk * elements * experts * numel + offset];
    result[offset] = total;
#endif
}
