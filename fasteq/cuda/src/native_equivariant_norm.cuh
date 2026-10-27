#pragma once

// FastEq weighted-moment/centering/affine formulas, expressed in native CUDA.
// The row statistics extend FastEq's existing warp + shared block reduction.
// Parameter partials retain FastEq's per-copy angular sum; the cross-row
// transpose retains its fixed 256-row ordered tree with a padded shared tile.

using NormT = FASTEQ_NORM_TYPE;
using NormIndex = long long;

__device__ __forceinline__ NormT norm_add(NormT a, NormT b) {
#if FASTEQ_NORM_FP64
    return __dadd_rn(a, b);
#else
    return __fadd_rn(a, b);
#endif
}
__device__ __forceinline__ NormT norm_mul(NormT a, NormT b) {
#if FASTEQ_NORM_FP64
    return __dmul_rn(a, b);
#else
    return __fmul_rn(a, b);
#endif
}

#if FASTEQ_NORM_ROLE < 3
constexpr int NormD = FASTEQ_NORM_DIM;
constexpr int NormThreads = FASTEQ_NORM_THREADS;
constexpr int NormItems = FASTEQ_NORM_BLOCK / NormThreads;
constexpr int NormWarps = NormThreads / 32;

template <int Count>
__device__ __forceinline__ void norm_block_statistics(NormT (&values)[Count], NormT* workspace) {
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    #pragma unroll
    for (int statistic = 0; statistic < Count; ++statistic) {
        values[statistic] = warp_reduce_sum(values[statistic]);
        if (lane == 0) workspace[statistic * NormWarps + warp] = values[statistic];
    }
    __syncthreads();
    if (warp == 0) {
        #pragma unroll
        for (int statistic = 0; statistic < Count; ++statistic) {
            NormT total = lane < NormWarps ? workspace[statistic * NormWarps + lane] : NormT(0);
            total = warp_reduce_sum(total);
            if (lane == 0) workspace[statistic * NormWarps] = total;
        }
    }
    __syncthreads();
    #pragma unroll
    for (int statistic = 0; statistic < Count; ++statistic)
        values[statistic] = workspace[statistic * NormWarps];
    // All warps finish reading the broadcast before reusing this workspace.
    __syncthreads();
}

__device__ __forceinline__ NormT norm_local_tree(NormT (&values)[NormItems]) {
    #pragma unroll
    for (int distance = NormItems / 2; distance > 0; distance >>= 1)
        #pragma unroll
        for (int item = 0; item < distance; ++item)
            values[item] = norm_add(values[item], values[item + distance]);
    return values[0];
}

template <bool Two>
__device__ __forceinline__ void norm_project(
    NormT (&value)[NormItems], NormT (&direction)[NormItems], NormT* workspace) {
    #pragma unroll
    for (int group = 0; group < FASTEQ_NORM_CENTERS; ++group) {
        NormT selected[NormItems], tangent[NormItems];
        #pragma unroll
        for (int item = 0; item < NormItems; ++item) {
            const int coordinate = threadIdx.x + item * NormThreads;
            const bool member = coordinate >= fasteq_norm_center_start[group] &&
                coordinate < fasteq_norm_center_start[group] + fasteq_norm_center_width[group];
            selected[item] = member ? value[item] : NormT(0);
            tangent[item] = Two && member ? direction[item] : NormT(0);
        }
        NormT statistics[Two ? 2 : 1];
        statistics[0] = norm_local_tree(selected);
        if (Two) statistics[1] = norm_local_tree(tangent);
        norm_block_statistics(statistics, workspace);
        const NormT mean = statistics[0] / NormT(fasteq_norm_center_width[group]);
        const NormT tangent_mean = Two ? statistics[1] / NormT(fasteq_norm_center_width[group]) : NormT(0);
        #pragma unroll
        for (int item = 0; item < NormItems; ++item) {
            const int coordinate = threadIdx.x + item * NormThreads;
            if (coordinate >= fasteq_norm_center_start[group] &&
                coordinate < fasteq_norm_center_start[group] + fasteq_norm_center_width[group]) {
                value[item] = norm_add(value[item], -mean);
                if (Two) direction[item] = norm_add(direction[item], -tangent_mean);
            }
        }
    }
}

extern "C" __global__ void FASTEQ_NORM_ENTRY(FASTEQ_NORM_SIGNATURE) {
    const NormIndex row = blockIdx.x;
    __shared__ NormT statistics_workspace[4 * NormWarps];
#if FASTEQ_NORM_GPARTIAL
    __shared__ NormT parameter_values[NormD];
#endif
    NormT v[NormItems], tangent[NormItems], b[NormItems];
    NormT variance_values[NormItems], a_values[NormItems], s_values[NormItems], c_values[NormItems];
    #pragma unroll
    for (int item = 0; item < NormItems; ++item) {
        const int coordinate = threadIdx.x + item * NormThreads;
        const bool valid = coordinate < NormD;
        v[item] = valid ? x[row * NormD + coordinate] : NormT(0);
        b[item] = valid ? coefficient[coordinate] : NormT(0);
#if FASTEQ_NORM_ROLE == 2
        tangent[item] = valid && FASTEQ_NORM_HAS_U ? u[row * NormD + coordinate] : NormT(0);
#else
        tangent[item] = NormT(0);
#endif
    }
#if FASTEQ_NORM_ROLE == 2
    norm_project<true>(v, tangent, statistics_workspace);
#else
    norm_project<false>(v, tangent, statistics_workspace);
#endif
    #pragma unroll
    for (int item = 0; item < NormItems; ++item) {
        const int coordinate = threadIdx.x + item * NormThreads;
        variance_values[item] = norm_mul(norm_mul(v[item], v[item]), b[item]);
        a_values[item] = s_values[item] = c_values[item] = NormT(0);
#if FASTEQ_NORM_ROLE > 0
        if (coordinate < NormD) {
            const NormT seed = dy[row * NormD + coordinate];
            const NormT gamma = FASTEQ_NORM_GAMMA(coordinate);
            const NormT g = norm_mul(seed, gamma);
            a_values[item] = norm_mul(g, v[item]);
#if FASTEQ_NORM_ROLE == 2
            const NormT hvalue = FASTEQ_NORM_H(coordinate);
            s_values[item] = norm_mul(norm_mul(tangent[item], b[item]), v[item]);
            c_values[item] = norm_add(norm_mul(tangent[item], g),
                                     norm_mul(norm_mul(seed, hvalue), v[item]));
#endif
        }
#endif
    }
#if FASTEQ_NORM_ROLE == 0
    NormT statistics[1] = {norm_local_tree(variance_values)};
#elif FASTEQ_NORM_ROLE == 1
    NormT statistics[2] = {norm_local_tree(variance_values), norm_local_tree(a_values)};
#else
    NormT statistics[4] = {norm_local_tree(variance_values), norm_local_tree(a_values),
                          norm_local_tree(s_values), norm_local_tree(c_values)};
#endif
    norm_block_statistics(statistics, statistics_workspace);
    const NormT r = rsqrt(norm_add(statistics[0], NormT(FASTEQ_NORM_EPS)));
    const NormT r3 = norm_mul(norm_mul(r, r), r);
#if FASTEQ_NORM_ROLE > 0
    const NormT a = statistics[1];
#endif
#if FASTEQ_NORM_ROLE == 2
    const NormT s = statistics[2], c = statistics[3];
#endif
    NormT result_x[NormItems];
    #pragma unroll
    for (int item = 0; item < NormItems; ++item) {
        const int coordinate = threadIdx.x + item * NormThreads;
        result_x[item] = NormT(0);
        if (coordinate < NormD) {
            const NormT gamma = FASTEQ_NORM_GAMMA(coordinate);
#if FASTEQ_NORM_ROLE == 0
            y[row * NormD + coordinate] = norm_add(norm_mul(norm_mul(v[item], r), gamma),
                                                    FASTEQ_NORM_BETA(coordinate));
#elif FASTEQ_NORM_ROLE == 1
            const NormT seed = dy[row * NormD + coordinate];
            const NormT g = norm_mul(seed, gamma);
            result_x[item] = norm_add(norm_mul(r, g),
                -norm_mul(norm_mul(norm_mul(r3, b[item]), v[item]), a));
#if FASTEQ_NORM_GPARTIAL
            parameter_values[coordinate] = norm_mul(norm_mul(seed, v[item]), r);
#endif
#else
            const NormT seed = dy[row * NormD + coordinate];
            const NormT hvalue = FASTEQ_NORM_H(coordinate);
            const NormT kvalue = FASTEQ_NORM_K(coordinate);
            const NormT g = norm_mul(seed, gamma);
            const NormT term0 = norm_mul(norm_mul(r, seed), hvalue);
            const NormT term1 = norm_mul(norm_mul(norm_mul(b[item], v[item]), r3), c);
            const NormT term2 = norm_mul(norm_mul(norm_mul(norm_mul(norm_mul(norm_mul(norm_mul(
                NormT(3), b[item]), v[item]), r3), r), r), a), s);
            const NormT term3 = norm_mul(r3, norm_add(norm_mul(s, g), norm_mul(norm_mul(a, b[item]), tangent[item])));
            result_x[item] = norm_add(norm_add(norm_add(term0, -term1), term2), -term3);
            if (FASTEQ_NORM_DDYFLAG) {
                const NormT direct = norm_mul(r, norm_add(norm_mul(tangent[item], gamma), norm_mul(hvalue, v[item])));
                const NormT correction = norm_mul(norm_mul(norm_mul(r3, s), v[item]), gamma);
                ddy[row * NormD + coordinate] = norm_add(norm_add(direct, -correction), kvalue);
            }
#if FASTEQ_NORM_GPARTIAL
            parameter_values[coordinate] = norm_mul(seed, norm_add(norm_mul(r, tangent[item]),
                -norm_mul(norm_mul(r3, s), v[item])));
#endif
#endif
        }
    }
#if FASTEQ_NORM_ROLE > 0
#if FASTEQ_NORM_ROLE == 1
    const bool write_x = FASTEQ_NORM_DXFLAG;
#else
    const bool write_x = FASTEQ_NORM_DDXFLAG;
#endif
    if (write_x) {
        norm_project<false>(result_x, tangent, statistics_workspace);
        #pragma unroll
        for (int item = 0; item < NormItems; ++item) {
            const int coordinate = threadIdx.x + item * NormThreads;
            if (coordinate < NormD) dx[row * NormD + coordinate] = result_x[item];
        }
    }
#if FASTEQ_NORM_GPARTIAL
    __syncthreads();
    fasteq_norm_parameter_partials(parameter_values, dy, partials, row);
#else
    fasteq_norm_parameter_partials(nullptr, dy, partials, row);
#endif
#endif
}
#else
// Original FastEq _reduce_rows: exact fixed 256-row tree, BC=32 channel tile.
// Add rows r and r+128 first to retain only half the tree in shared memory.
// Its 128x33 tile is 16.5 KiB FP32 / 33 KiB FP64, below the default 48 KiB.
extern "C" __global__ void FASTEQ_NORM_ENTRY(
    const NormT* input, NormT* output, NormIndex rows, int width) {
    __shared__ NormT tile[128][33];
    const NormIndex row_base = NormIndex(blockIdx.x) * 256;
    const int column_base = blockIdx.y * 32;
    for (int cell = threadIdx.x; cell < 128 * 32; cell += 128) {
        const int r = cell / 32, c = cell % 32;
        const int column = column_base + c;
        const NormT first = row_base + r < rows && column < width ? input[(row_base + r) * width + column] : NormT(0);
        const NormT second = row_base + r + 128 < rows && column < width ? input[(row_base + r + 128) * width + column] : NormT(0);
        tile[r][c] = norm_add(first, second);
    }
    __syncthreads();
    #pragma unroll
    for (int distance = 64; distance > 0; distance >>= 1) {
        for (int cell = threadIdx.x; cell < distance * 32; cell += 128) {
            const int r = cell / 32, c = cell % 32;
            tile[r][c] = norm_add(tile[r][c], tile[r + distance][c]);
        }
        __syncthreads();
    }
    if (threadIdx.x < 32 && column_base + threadIdx.x < width)
        output[NormIndex(blockIdx.x) * width + column_base + threadIdx.x] = tile[0][threadIdx.x];
}
#endif
