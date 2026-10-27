#pragma once

// Dense degree rotations extend FastEq FCTP's channel-vector loader and
// padded shared transpose. The Python layer supplies only degree/pair tables.
// No EQX CUDA template, ATen matrix arithmetic, or external executor is used.

#if defined(FASTEQ_FRAME_ROLE)
using FrameT = FASTEQ_ROTATION_TYPE;
constexpr int FrameC = FASTEQ_ROTATION_CHANNELS;
constexpr int FrameNI = FASTEQ_ROTATION_NI;
constexpr int FrameNO = FASTEQ_ROTATION_NO;

extern "C" __global__ void FASTEQ_ROTATION_ENTRY(
    const FrameT* features, const FrameT* coefficients,
    const FrameT* cotangent, FrameT* result) {
    const long long edge_id = blockIdx.x;
    const int tid = threadIdx.x;
    constexpr int tile_channels = FASTEQ_FRAME_TILE_CHANNELS;
    constexpr int input_pitch = FrameNI + (FrameNI % 2 == 0);
    constexpr int output_pitch = FrameNO + (FrameNO % 2 == 0);

#if FASTEQ_FRAME_ROLE == 1
    // Cache every source/destination angle once per edge. Each warp owns
    // independent matching matrix cells; feature channels are contiguous
    // in global memory and become odd-pitch shared columns with FCTP's loader.
    __shared__ FrameT source_tile[tile_channels][input_pitch];
    __shared__ FrameT seed_tile[tile_channels][output_pitch];
    constexpr int pair_rounds = (FASTEQ_ROTATION_PAIRS + 3) / 4;
    FrameT accum[pair_rounds ? pair_rounds : 1] = {};
    const int warp = tid >> 5, lane = tid & 31;
    for (int start = 0; start < FrameC; start += tile_channels) {
        for (int cell = tid; cell < FrameNI * tile_channels / 4; cell += 128) {
            const int angle = cell / (tile_channels / 4);
            const int column = 4 * (cell % (tile_channels / 4));
            const FrameT* p = features + (edge_id * FrameNI + angle) * FrameC + start + column;
            if (start + column + 3 < FrameC &&
                (reinterpret_cast<unsigned long long>(p) % alignof(typename fasteq_fctp::Vector4<FrameT>::type)) == 0)
                fasteq_fctp::transpose_four(&source_tile[0][0], input_pitch, column, angle, p);
            else
                for (int item = 0; item < 4; ++item)
                    source_tile[column + item][angle] = start + column + item < FrameC ? p[item] : FrameT(0);
        }
        for (int cell = tid; cell < FrameNO * tile_channels / 4; cell += 128) {
            const int angle = cell / (tile_channels / 4);
            const int column = 4 * (cell % (tile_channels / 4));
            const FrameT* p = cotangent + (edge_id * FrameNO + angle) * FrameC + start + column;
            if (start + column + 3 < FrameC &&
                (reinterpret_cast<unsigned long long>(p) % alignof(typename fasteq_fctp::Vector4<FrameT>::type)) == 0)
                fasteq_fctp::transpose_four(&seed_tile[0][0], output_pitch, column, angle, p);
            else
                for (int item = 0; item < 4; ++item)
                    seed_tile[column + item][angle] = start + column + item < FrameC ? p[item] : FrameT(0);
        }
        __syncthreads();
        #pragma unroll
        for (int round = 0; round < pair_rounds; ++round) {
            const int pair = 4 * round + warp;
            if (pair < FASTEQ_ROTATION_PAIRS) {
                const int input = fasteq_frame_pair_in[pair];
                const int output = fasteq_frame_pair_out[pair];
                #pragma unroll
                for (int channel = lane; channel < tile_channels; channel += 32)
                    accum[round] += source_tile[channel][input] * seed_tile[channel][output];
            }
        }
        __syncthreads();
    }
    #pragma unroll
    for (int round = 0; round < pair_rounds; ++round) {
        const FrameT sum = warp_reduce_sum(accum[round]);
        const int pair = 4 * round + warp;
        if (lane == 0 && pair < FASTEQ_ROTATION_PAIRS)
            result[(edge_id * FrameNO + fasteq_frame_pair_out[pair]) * FrameNI +
                   fasteq_frame_pair_in[pair]] = sum;
    }
#else
    constexpr int input_angles = FASTEQ_FRAME_ROLE == 2 ? FrameNI : FrameNO;
    constexpr int output_angles = FASTEQ_FRAME_ROLE == 2 ? FrameNO : FrameNI;
    constexpr int pitch = input_angles + (input_angles % 2 == 0);
    __shared__ FrameT feature_tile[tile_channels][pitch];
    const int start = blockIdx.y * tile_channels;
    const FrameT* input = FASTEQ_FRAME_ROLE == 2 ? features : cotangent;
    for (int cell = tid; cell < input_angles * tile_channels / 4; cell += 128) {
        const int angle = cell / (tile_channels / 4);
        const int column = 4 * (cell % (tile_channels / 4));
        const FrameT* p = input + (edge_id * input_angles + angle) * FrameC + start + column;
        if (start + column + 3 < FrameC &&
            (reinterpret_cast<unsigned long long>(p) % alignof(typename fasteq_fctp::Vector4<FrameT>::type)) == 0)
            fasteq_fctp::transpose_four(&feature_tile[0][0], pitch, column, angle, p);
        else
            for (int item = 0; item < 4; ++item)
                feature_tile[column + item][angle] = start + column + item < FrameC ? p[item] : FrameT(0);
    }
    __syncthreads();
    if (tid < tile_channels && start + tid < FrameC) {
        #pragma unroll
        for (int output = 0; output < output_angles; ++output) {
            FrameT sum = 0;
            #pragma unroll
            for (int source = 0; source < input_angles; ++source) {
                const int original_out = FASTEQ_FRAME_ROLE == 2 ? output : source;
                const int original_in = FASTEQ_FRAME_ROLE == 2 ? source : output;
                if (fasteq_frame_degree_out[original_out] == fasteq_frame_degree_in[original_in])
                    sum += coefficients[(edge_id * FrameNO + original_out) * FrameNI + original_in] *
                           feature_tile[tid][source];
            }
            result[(edge_id * output_angles + output) * FrameC + start + tid] = sum;
        }
    }
#endif
}
#endif

#if defined(FASTEQ_ROTARY_PACKED_WARPS)
#if defined(FASTEQ_ROTARY_ROLE)
using RotaryT = FASTEQ_ROTATION_TYPE;
constexpr int RotaryC = FASTEQ_ROTATION_CHANNELS;
constexpr int RotaryH = FASTEQ_ROTATION_HEADS;
constexpr int RotaryHC = RotaryC / RotaryH;
constexpr int RotaryA = FASTEQ_ROTATION_ANGULAR;
constexpr int RotaryM = FASTEQ_ROTATION_MMAX;
constexpr int RotaryP = FASTEQ_ROTATION_PAIRS;

extern "C" __global__ void FASTEQ_ROTATION_ENTRY(
    const RotaryT* query, const RotaryT* key, const RotaryT* phase,
    const RotaryT* score, RotaryT* result) {
    const long long edge_id = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#if FASTEQ_ROTARY_ROLE < 2
    // A warp owns a real/imaginary angular pair. Phase coefficients and score
    // are warp-broadcast, while both real and imaginary features are read
    // once per channel and produce both destinations together.
    // Extend the existing paired-angular warp mapping across global channels.
    // HC16 uses two head subgroups per warp instead of leaving half idle.
    constexpr int channel_tiles = (RotaryC + 31) / 32;
    const int task = blockIdx.y * 4 + warp;
    const int pair = task / channel_tiles;
    const int global_channel = 32 * (task % channel_tiles) + lane;
    const int head = global_channel / RotaryHC;
    const int channel = global_channel % RotaryHC;
    const int phase_lane = lane - lane % RotaryHC;
    if (pair >= RotaryP) return;  // uniform predicate within each full warp
    const int order = fasteq_rotary_order[pair];
    const int real_angle = fasteq_rotary_real[pair];
    const int imag_angle = fasteq_rotary_imag[pair];
    RotaryT pr = 0, pi = 0, multiplier = 0;
    if (lane == phase_lane) {
        const long long offset = ((edge_id * (RotaryM + 1) + order) * RotaryH + head) * 2;
        pr = phase[offset]; pi = phase[offset + 1];
        multiplier = score[edge_id * RotaryH + head];
    }
    pr = __shfl_sync(0xffffffff, pr, phase_lane);
    pi = __shfl_sync(0xffffffff, pi, phase_lane);
    multiplier = __shfl_sync(0xffffffff, multiplier, phase_lane);
    if (channel < RotaryHC) {
        const RotaryT* other = FASTEQ_ROTARY_ROLE == 0 ? key : query;
        const long long real_offset = (edge_id * RotaryA + real_angle) * RotaryC + head * RotaryHC + channel;
        const RotaryT real = other[real_offset];
        RotaryT real_result = pr * real;
        if (imag_angle >= 0) {
            const long long imag_offset = (edge_id * RotaryA + imag_angle) * RotaryC + head * RotaryHC + channel;
            const RotaryT imaginary = other[imag_offset];
            const RotaryT sign = FASTEQ_ROTARY_ROLE == 0 ? RotaryT(-1) : RotaryT(1);
            real_result += sign * pi * imaginary;
            result[imag_offset] = (pr * imaginary - sign * pi * real) * multiplier;
        }
        result[real_offset] = real_result * multiplier;
    }
#else
    // Original FastEq warp reduction now owns one complete head/order task.
    // Four independent tasks share a block; no shared partial tree/barriers.
    constexpr int phase_orders = FASTEQ_ROTARY_ROLE == 2 ? RotaryM + 1 : 1;
    const int task = blockIdx.y * 4 + warp;
    if (task >= RotaryH * phase_orders) return;  // uniform within a warp
    const int head = task / phase_orders;
    const int selected_order = task % phase_orders;
    RotaryT real_sum = 0, imag_sum = 0;
    for (int index = lane; index < RotaryP * RotaryHC; index += 32) {
        const int pair = index / RotaryHC, channel = index % RotaryHC;
        const int order = fasteq_rotary_order[pair];
        if (FASTEQ_ROTARY_ROLE == 2 && order != selected_order) continue;
        const long long r = (edge_id * RotaryA + fasteq_rotary_real[pair]) * RotaryC + head * RotaryHC + channel;
        const int imag_angle = fasteq_rotary_imag[pair];
        RotaryT dot = query[r] * key[r], cross = 0;
        if (imag_angle >= 0) {
            const long long i = (edge_id * RotaryA + imag_angle) * RotaryC + head * RotaryHC + channel;
            const RotaryT qr = query[r], qi = query[i], kr = key[r], ki = key[i];
            dot += qi * ki;
            cross = qi * kr - qr * ki;
        }
        if (FASTEQ_ROTARY_ROLE == 2) {
            real_sum += dot; imag_sum += cross;
        } else {
            const long long p = ((edge_id * (RotaryM + 1) + order) * RotaryH + head) * 2;
            real_sum += phase[p] * dot + phase[p + 1] * cross;
        }
    }
    real_sum = warp_reduce_sum(real_sum);
    imag_sum = warp_reduce_sum(imag_sum);
    if (lane == 0) {
        if (FASTEQ_ROTARY_ROLE == 2) {
            const long long offset = ((edge_id * (RotaryM + 1) + selected_order) * RotaryH + head) * 2;
            const RotaryT multiplier = score[edge_id * RotaryH + head];
            result[offset] = real_sum * multiplier;
            result[offset + 1] = imag_sum * multiplier;
        } else result[edge_id * RotaryH + head] = real_sum;
    }
#endif
}
#endif

#else
#if defined(FASTEQ_ROTARY_ROLE)
using RotaryT = FASTEQ_ROTATION_TYPE;
constexpr int RotaryC = FASTEQ_ROTATION_CHANNELS;
constexpr int RotaryH = FASTEQ_ROTATION_HEADS;
constexpr int RotaryHC = RotaryC / RotaryH;
constexpr int RotaryA = FASTEQ_ROTATION_ANGULAR;
constexpr int RotaryM = FASTEQ_ROTATION_MMAX;
constexpr int RotaryP = FASTEQ_ROTATION_PAIRS;

extern "C" __global__ void FASTEQ_ROTATION_ENTRY(
    const RotaryT* query, const RotaryT* key, const RotaryT* phase,
    const RotaryT* score, RotaryT* result) {
    const long long edge_id = blockIdx.x;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
#if FASTEQ_ROTARY_ROLE < 2
    // A warp owns a real/imaginary angular pair. Phase coefficients and score
    // are warp-broadcast, while both real and imaginary features are read
    // once per channel and produce both destinations together.
    constexpr int channel_tiles = (RotaryHC + 31) / 32;
    constexpr int pair_tiles = (RotaryP + 3) / 4;
    const int head = blockIdx.y / (channel_tiles * pair_tiles);
    const int tile = blockIdx.y % (channel_tiles * pair_tiles);
    const int pair = 4 * (tile / channel_tiles) + warp;
    const int channel = 32 * (tile % channel_tiles) + lane;
    if (pair >= RotaryP) return;  // uniform predicate within each full warp
    const int order = fasteq_rotary_order[pair];
    const int real_angle = fasteq_rotary_real[pair];
    const int imag_angle = fasteq_rotary_imag[pair];
    RotaryT pr = 0, pi = 0, multiplier = 0;
    if (lane == 0) {
        const long long offset = ((edge_id * (RotaryM + 1) + order) * RotaryH + head) * 2;
        pr = phase[offset]; pi = phase[offset + 1];
        multiplier = score[edge_id * RotaryH + head];
    }
    pr = __shfl_sync(0xffffffff, pr, 0);
    pi = __shfl_sync(0xffffffff, pi, 0);
    multiplier = __shfl_sync(0xffffffff, multiplier, 0);
    if (channel < RotaryHC) {
        const RotaryT* other = FASTEQ_ROTARY_ROLE == 0 ? key : query;
        const long long real_offset = (edge_id * RotaryA + real_angle) * RotaryC + head * RotaryHC + channel;
        const RotaryT real = other[real_offset];
        RotaryT real_result = pr * real;
        if (imag_angle >= 0) {
            const long long imag_offset = (edge_id * RotaryA + imag_angle) * RotaryC + head * RotaryHC + channel;
            const RotaryT imaginary = other[imag_offset];
            const RotaryT sign = FASTEQ_ROTARY_ROLE == 0 ? RotaryT(-1) : RotaryT(1);
            real_result += sign * pi * imaginary;
            result[imag_offset] = (pr * imaginary - sign * pi * real) * multiplier;
        }
        result[real_offset] = real_result * multiplier;
    }
#else
    constexpr int phase_orders = FASTEQ_ROTARY_ROLE == 2 ? RotaryM + 1 : 1;
    const int head = blockIdx.y / phase_orders;
    const int selected_order = blockIdx.y % phase_orders;
    __shared__ RotaryT phases[2 * (RotaryM + 1)];
    __shared__ RotaryT real_partial[4], imag_partial[4];
    for (int p = tid; p < 2 * (RotaryM + 1); p += 128)
        phases[p] = phase[((edge_id * (RotaryM + 1) + p / 2) * RotaryH + head) * 2 + p % 2];
    __syncthreads();
    RotaryT real_sum = 0, imag_sum = 0;
    for (int index = tid; index < RotaryP * RotaryHC; index += 128) {
        const int pair = index / RotaryHC, channel = index % RotaryHC;
        const int order = fasteq_rotary_order[pair];
        if (FASTEQ_ROTARY_ROLE == 2 && order != selected_order) continue;
        const long long r = (edge_id * RotaryA + fasteq_rotary_real[pair]) * RotaryC + head * RotaryHC + channel;
        const int imag_angle = fasteq_rotary_imag[pair];
        RotaryT dot = query[r] * key[r], cross = 0;
        if (imag_angle >= 0) {
            const long long i = (edge_id * RotaryA + imag_angle) * RotaryC + head * RotaryHC + channel;
            const RotaryT qr = query[r], qi = query[i], kr = key[r], ki = key[i];
            dot += qi * ki;
            cross = qi * kr - qr * ki;
        }
        if (FASTEQ_ROTARY_ROLE == 2) {
            real_sum += dot; imag_sum += cross;
        } else {
            real_sum += phases[2 * order] * dot + phases[2 * order + 1] * cross;
        }
    }
    real_sum = warp_reduce_sum(real_sum);
    imag_sum = warp_reduce_sum(imag_sum);
    if (lane == 0) { real_partial[warp] = real_sum; imag_partial[warp] = imag_sum; }
    __syncthreads();
    if (warp == 0) {
        real_sum = warp_reduce_sum(lane < 4 ? real_partial[lane] : RotaryT(0));
        imag_sum = warp_reduce_sum(lane < 4 ? imag_partial[lane] : RotaryT(0));
        if (lane == 0) {
            if (FASTEQ_ROTARY_ROLE == 2) {
                const long long offset = ((edge_id * (RotaryM + 1) + selected_order) * RotaryH + head) * 2;
                const RotaryT multiplier = score[edge_id * RotaryH + head];
                result[offset] = real_sum * multiplier;
                result[offset + 1] = imag_sum * multiplier;
            } else result[edge_id * RotaryH + head] = real_sum;
        }
    }
#endif
}
#endif

#endif
