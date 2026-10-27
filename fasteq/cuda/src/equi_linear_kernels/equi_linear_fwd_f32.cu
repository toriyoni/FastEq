#include <cuda.h>
#include <cuda/barrier>
#include <cuda/ptx>
#include <cudaTypedefs.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#include <stdexcept>
#include <utility>

#include "./device_cuda_helper.cuh"
#include "./impl.h"
#include "./ptx_inst.cuh"

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

// Optional native mul_ir access for the existing WGMMA pipeline. These paths
// have one input/output irrep each; the general packed API remains unchanged.
struct LinearNativePath { int sx, sw, sy, ci, dim; };
struct LinearNativeAccess {
    const float *x = nullptr, *w = nullptr;
    float *y = nullptr;
    const int64_t *ids = nullptr;
    int din = 0, dout = 0;
    LinearNativePath paths[16]{};
};

/* ===================== Device Helper Functions ===================== */
__device__ __forceinline__ void wg_trans_16x32f32tf32_sw128B_sync(
    float *out_smem_ptr,            // TF32 high tile
    float *low_smem_ptr,            // TF32 residual tile
    float *tmp_smem_ptr,            // tmp buffer ptr containning input 16x32 data
    const float &cur_cg_val,        // cur_cg_val
    const uint32_t &base_sw_x0_tag, // 0 or 1 (indicate swizzle 16B_x starts from 0 or 4)
    const uint32_t &in_wg_tid       // in warpgroup tid
)
{
    // sw128B means the transposed matrix leading dimention must be 32(floats).
    // so each 16x32 input matrix will be transposed to a 32x16 matirx,
    // and then make 128B swizzle with a neighbor 32x16 matirx (i.e. 32x32 matirx sw128B)
    uint32_t ld_col = in_wg_tid % 32;
    uint32_t ld_row0 = (in_wg_tid / 32) * 4;
    float tmp_reg[4], low_reg[4];
#pragma unroll
    for (uint32_t cnt = 0; cnt < 4; ++cnt)
    {
        uint32_t ld_row = ld_row0 + cnt;
        float fp32_data = tmp_smem_ptr[ld_row * 32 + ld_col] * cur_cg_val;
        uint32_t tf32_data;
        asm_cvt_tf32_f32(tf32_data, fp32_data);
        tmp_reg[cnt] = __uint_as_float(tf32_data);
        asm_cvt_tf32_f32(tf32_data, fp32_data - tmp_reg[cnt]);
        low_reg[cnt] = __uint_as_float(tf32_data);
    }

    // store to out smem sync
    uint32_t st_sw_y = ld_col;
    uint32_t st_unsw_16B_x = base_sw_x0_tag * 4 + (in_wg_tid / 32); // 0~7
    uint32_t st_sw_x = ((st_sw_y % 8) ^ st_unsw_16B_x) * 4;
    FETCH_16B(out_smem_ptr[st_sw_y * 32 + st_sw_x]) = FETCH_16B(tmp_reg[0]);
    FETCH_16B(low_smem_ptr[st_sw_y * 32 + st_sw_x]) = FETCH_16B(low_reg[0]);
}

__device__ __forceinline__ void warp_ldmatirx_8x32f32_sw128B_sync(
    float d_reg[16],         // out
    float *smem_ptr,         // smem_ptr
    const uint32_t &cnt0,    // out base offset (0 or 1)
    const uint32_t &smem_y0, // smem_y base offset (in view of '32bit_T smem[y][32]')
    const uint32_t &warp_tid // warpgroup tid
)
{
#pragma unroll
    for (uint32_t cnt = 0; cnt < 2; ++cnt)
    {
        uint32_t sw_y = warp_tid % 8;
        uint32_t unsw_x_16B = (warp_tid / 8) + cnt * 4;
        uint32_t sw_x_16B = (sw_y ^ unsw_x_16B) % 8;
        uint32_t smem_offset = (sw_y + smem_y0) * 32 + sw_x_16B * 4;
        uint32_t real_smem_ptr =
            static_cast<uint32_t>(__cvta_generic_to_shared(reinterpret_cast<void *>(smem_ptr + smem_offset)));
        asm_ldmatrix_x4(d_reg[cnt * 8 + cnt0 + 0], // r0
                        d_reg[cnt * 8 + cnt0 + 2], // r1
                        d_reg[cnt * 8 + cnt0 + 4], // r2
                        d_reg[cnt * 8 + cnt0 + 6], // r3
                        real_smem_ptr              // smem ptr
        );
    }
}

__device__ __forceinline__ void warpgroup_ldmatirx_64x32f32_sw128B_sync(
    float d_reg[16],          // out
    float *smem_ptr,          // smem_ptr
    const uint32_t &smem_y0,  // smem_y base offset (in view of '32bit_T smem[y][32]')
    const uint32_t &in_wg_tid // warpgroup tid
)
{
    uint32_t warp_tid = in_wg_tid % WARP_SIZE;
    uint32_t in_wg_wid = in_wg_tid / WARP_SIZE;
    uint32_t warp_smem_y0 = in_wg_wid * 16 + smem_y0;
#pragma unroll
    for (uint32_t cnt = 0; cnt < 2; ++cnt)
    {
        uint32_t warp_smem_y = warp_smem_y0 + 8 * cnt;
        warp_ldmatirx_8x32f32_sw128B_sync(d_reg, smem_ptr, cnt, warp_smem_y, warp_tid);
    }
}

__device__ __forceinline__ uint64_t make_smem_desc(float *ptr)
{
    // k-major
    uint32_t addr = static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
    uint64_t desc = 0x0000000000000000;
    desc |= matrix_descriptor_encode(addr);
    // Swizzled layouts: not used, assumed to be 1.
    desc |= (1llu) << 16;
    // The offset from the first 8 rows to the next 8 rows.
    desc |= matrix_descriptor_encode(1024) << 32;
    desc |= (1llu) << 62; // swizzle 128B
    return desc;
}

template <uint32_t TILE_N>
__device__ void wg_reg_swap_swizzle4_m64n32f32(float accu[TILE_N / 8][4], // accu
                                               const uint32_t &in_wg_tid  // in_wg_tid
)
{
    static_assert(TILE_N % 32 == 0);
    /* TILE_N must be divisable by 32 !!
     * input matrix m64nN shape: (same as wgmma.m64nNk8 defines)
     * output matrix m64nN - tilem4n32 shape:(8float, swizzle4)
     *    | T0~3{d0, d1}     |  T0~3{d4, d5}     |  T0~3{d8, d9}   | T0~3{d12, d13}
     *    | T4~7{d4, d5}     |  T4~7{d0, d1}     |  T4~7{d12, d13} | T4~7{d8, d9}
     *    | T8~11{d8, d9}    |  T8~11{d12, d13}  |  T8~11{d0, d1}  | T8~11{d4, d5}
     *    | T12~15{d12, d13} |  T12~15{d8, d9}   |  T12~15{d4, d5} | T12~15{d0, d1}
     * No bank-conflict when threads in a warp store float2 of same tags(e.g.{d0, d1}),
     * and the Matrix in SMEM matrix CU_TENSOR_MAP_SWIZZLE_NONE.
     */
    if (in_wg_tid % 8 >= 4)
    {
#pragma unroll
        for (uint32_t _cnt = 0; _cnt < (TILE_N / 32); ++_cnt)
        {
            swap_Nregs_f32<4>(accu[_cnt * 4 + 0], accu[_cnt * 4 + 1]);
            swap_Nregs_f32<4>(accu[_cnt * 4 + 2], accu[_cnt * 4 + 3]);
        }
    }
    if (in_wg_tid % 16 >= 8)
    {
#pragma unroll
        for (uint32_t _cnt = 0; _cnt < (TILE_N / 32); ++_cnt)
        {
            swap_Nregs_f32<4>(accu[_cnt * 4 + 0], accu[_cnt * 4 + 2]);
            swap_Nregs_f32<4>(accu[_cnt * 4 + 1], accu[_cnt * 4 + 3]);
        }
    }
}

template <uint32_t TILE_N>
__device__ void wg_R2S_swizzle4_m64n32f32(float *out_smem0,          // out smem base ptr
                                          float accu[TILE_N / 8][4], // accu
                                          const uint32_t &in_wg_tid  // in_wg_tid
)
{
    static_assert(TILE_N % 32 == 0);
    constexpr uint32_t SMEM_Y_STRIDE = TILE_N / 32; // N方向每32个数一行

    uint32_t smem_y0 = ((in_wg_tid / 32) * 16) + ((in_wg_tid % 32) / 4);
    uint32_t smem_in_swg_x0 = (in_wg_tid % 4) * 2;

    // We make swizzle4 within swg(swizzle group)
    uint32_t smem_unswg_x0 = (in_wg_tid / 4) % 4;
#pragma unroll
    for (uint32_t _y = 0; _y < SMEM_Y_STRIDE; ++_y)
    {
        uint32_t smem_y1 = smem_y0 + _y;
#pragma unroll
        for (uint32_t _cnt = 0; _cnt < 4; ++_cnt)
        {
            uint32_t smem_swg_x0 = _cnt ^ smem_unswg_x0; // (0~3)
            uint32_t smem_x = smem_swg_x0 * 8 + smem_in_swg_x0;
#pragma unroll
            for (uint32_t _idx = 0; _idx < 2; ++_idx) // accu[0,1] | accu[2, 3]
            {
                uint32_t smem_y = smem_y1 + _idx * 8;
                uint32_t smem_offset = smem_y * 32 + smem_x;
                FETCH_8B(out_smem0[smem_offset]) = FETCH_8B(accu[_y * 4 + _cnt][_idx * 2]);
            }
        }
    }
}

/* ========================================================== */
/* =================== P&C Step Functions =================== */

template <uint32_t TILE_N, uint32_t TILE_K, uint32_t TOTAL_STAGES>
__device__ void preloop_producer(barrier bars_ready[],         // buffer finish consuming barrier
                                 barrier bars_filled[],        // buffer finish filling barrier
                                 float *tmp_smem,              //
                                 const CUtensorMap *w_map,     //
                                 const uint32_t &b_ntile_id,   //
                                 const uint32_t &b_in_path_id, //
                                 const uint32_t &U,            //
                                 const uint32_t &in_wg_tid     //
)
{
    if (in_wg_tid == 0)
    {
        constexpr uint32_t PRELOOP_BUFFER_SIZE_PER_STAGE = TILE_N * TILE_K;
        constexpr size_t TX_BYTES_PER_STAGE = PRELOOP_BUFFER_SIZE_PER_STAGE * sizeof(float);
        uint32_t cur_buffer_id = 0;
        uint32_t coord_n0 = b_ntile_id * TILE_N;
        uint32_t coord_n2 = b_in_path_id * 1;

        // prefetch loop 0 data to L2
        asm_cp_async_bulk_prefetch_tensor_3d_l2(w_map, coord_n0, 0, coord_n2);

        // loop
        uint32_t k_count;
        for (k_count = 0; k_count < (CEIL_DIV(U, TILE_K) - 1); ++k_count)
        {
            // wait buffer finish consuming
            bars_ready[cur_buffer_id].arrive_and_wait();
            // load w tile and prefech next tile
            uint32_t coord_n1 = k_count * TILE_K;
            uint32_t next_coord_n1 = (k_count + 1) * TILE_K;
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (tmp_smem + cur_buffer_id * PRELOOP_BUFFER_SIZE_PER_STAGE), // smem_ptr base
                w_map,                                                      // tensormap
                coord_n0, coord_n1, coord_n2,                               // coords
                bars_filled[cur_buffer_id]                                  // barrier
            );
            asm_cp_async_bulk_prefetch_tensor_3d_l2(w_map, coord_n0, next_coord_n1, coord_n2);
            cuda::device::barrier_arrive_tx(bars_filled[cur_buffer_id], 1, TX_BYTES_PER_STAGE);
            // switch to next buffer
            cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
        }
        // loop epilogue (no prefetch)
        {
            // wait buffer finish consuming
            bars_ready[cur_buffer_id].arrive_and_wait();
            // load w tile and prefech next tile
            uint32_t coord_n1 = k_count * TILE_K;
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (tmp_smem + cur_buffer_id * PRELOOP_BUFFER_SIZE_PER_STAGE), // smem_ptr base
                w_map,                                                      // tensormap
                coord_n0, coord_n1, coord_n2,                               // coords
                bars_filled[cur_buffer_id]                                  // barrier
            );
            cuda::device::barrier_arrive_tx(bars_filled[cur_buffer_id], 1, TX_BYTES_PER_STAGE);
            // switch to next buffer
            cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
        }

        // 循环结束时，bars_ready由于缺少producer wg的arrive，而没有被重置；
        // 需要等待所有buffer的tag闲置之后才能进行下一步操作。
#pragma unroll
        for (uint32_t i = 0; i < TOTAL_STAGES; ++i)
        {
            bars_ready[(cur_buffer_id + i) % TOTAL_STAGES].arrive_and_wait();
        }
    }
    // All warp groups must finish this phase before reused barriers/buffers
    // can be initialized for the next one. Arrive alone does not wait.
    __syncthreads();
}

template <uint32_t TILE_N, uint32_t TILE_K, uint32_t B_SMEM_SIZE_PER_TILE, uint32_t CONSUMER_WG_NUM,
          uint32_t TOTAL_STAGES>
__device__ void preloop_consumer(barrier bars_ready[],           // buffer finish consuming barrier
                                 barrier bars_filled[],          // buffer finish filling barrier
                                 float *B_smem,                  //
                                 float *tmp_smem,                //
                                 const float &cur_cg_val,        //
                                 const uint32_t &U,              //
                                 const uint32_t &consumer_wg_id, //
                                 const uint32_t &in_wg_tid       //
)
{
    static_assert(TILE_N == 32 && TILE_K == 32);
    constexpr uint32_t PRELOOP_BUFFER_SIZE_PER_STAGE = TILE_N * TILE_K;
    // init (All buffers are empty)
#pragma unroll
    for (uint32_t i = 0; i < TOTAL_STAGES; ++i)
    {
        bars_ready[i].arrive(); // buffer_0 is ready for initial fill
    }

    uint32_t cur_buffer_id = 0;
    // loop
    for (uint32_t k_count = 0; k_count < CEIL_DIV(U, TILE_K); ++k_count)
    {
        // Wait for the data to have arrived.
        bars_filled[cur_buffer_id].arrive_and_wait();
        // consume (TILE_N = 32)
        uint32_t base_sw_x0_tag = consumer_wg_id;
        wg_trans_16x32f32tf32_sw128B_sync(
            (B_smem + k_count * B_SMEM_SIZE_PER_TILE), // TF32 high
            (B_smem + U * TILE_N + k_count * B_SMEM_SIZE_PER_TILE), // residual
            (tmp_smem + cur_buffer_id * PRELOOP_BUFFER_SIZE_PER_STAGE +
             base_sw_x0_tag * 512), // tmp buffer ptr containning input 16x32 data
            cur_cg_val,             // cur_cg_val
            base_sw_x0_tag,         // 0 or 1
            in_wg_tid               // in warpgroup tid
        );
        // tell producer that consuming finished
        bars_ready[cur_buffer_id].arrive();
        // switch to next buffer
        cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
    }
    // All warp groups must finish this phase before reused barriers/buffers
    // can be initialized for the next one. Arrive alone does not wait.
    __syncthreads();
}

template <uint32_t TILE_M, uint32_t TILE_K, uint32_t A_SMEM_SIZE_PER_STAGE, uint32_t TOTAL_STAGES>
__device__ void mainloop_producer(barrier bars_ready[],       // buffer finish consuming barrier
                                  barrier bars_filled[],      // buffer finish filling barrier
                                  float *A_smem,              //
                                  const CUtensorMap *x_map,   //
                                  const uint32_t &b_in_i_id,  //
                                  const uint32_t &b_mtile_id, //
                                  const uint32_t &B,          //
                                  const uint32_t &in_total_i, //
                                  const uint32_t &U,          //
                                  const uint32_t &in_wg_tid   //
)
{
    if (in_wg_tid == 0)
    {
        constexpr size_t TX_BYTES_PER_STAGE = A_SMEM_SIZE_PER_STAGE * sizeof(float);
        uint32_t cur_buffer_id = 0;
        uint32_t coord_n1 = b_in_i_id * 1;
        uint32_t coord_n2 = b_mtile_id * TILE_M;
        // prefetch
        asm_cp_async_bulk_prefetch_tensor_3d_l2(x_map, 0, coord_n1, coord_n2);
        // loop
        uint32_t k_count;
        for (k_count = 0; k_count < (CEIL_DIV(U, TILE_K) - 1); ++k_count)
        {
            // wait buffer finish consuming
            bars_ready[cur_buffer_id].arrive_and_wait();
            // load x tile
            uint32_t coord_n0 = k_count * TILE_K;
            uint32_t next_coord_n0 = (k_count + 1) * TILE_K;
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (A_smem + cur_buffer_id * A_SMEM_SIZE_PER_STAGE), // smem_ptr base
                x_map,                                            // tensormap
                coord_n0, coord_n1, coord_n2,                     // coords
                bars_filled[cur_buffer_id]                        // barrier
            );
            asm_cp_async_bulk_prefetch_tensor_3d_l2(x_map, next_coord_n0, coord_n1, coord_n2);
            cuda::device::barrier_arrive_tx(bars_filled[cur_buffer_id], 1, TX_BYTES_PER_STAGE);
            // switch to next buffer
            cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
        }
        // loop epilogue (no prefetch)
        {
            // wait buffer finish consuming
            bars_ready[cur_buffer_id].arrive_and_wait();
            // load x tile
            uint32_t coord_n0 = k_count * TILE_K;
            cde::cp_async_bulk_tensor_3d_global_to_shared(
                (A_smem + cur_buffer_id * A_SMEM_SIZE_PER_STAGE), // smem_ptr base
                x_map,                                            // tensormap
                coord_n0, coord_n1, coord_n2,                     // coords
                bars_filled[cur_buffer_id]                        // barrier
            );
            cuda::device::barrier_arrive_tx(bars_filled[cur_buffer_id], 1, TX_BYTES_PER_STAGE);
            // switch to next buffer
            cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
        }
        // 循环结束时，bars_ready由于缺少producer wg的arrive，而没有被重置；
        // 需要等待所有buffer的tag闲置之后才能进行下一步操作。
#pragma unroll
        for (uint32_t i = 0; i < TOTAL_STAGES; ++i)
        {
            bars_ready[(cur_buffer_id + i) % TOTAL_STAGES].arrive_and_wait();
        }
    }
    // All warp groups must finish this phase before reused barriers/buffers
    // can be initialized for the next one. Arrive alone does not wait.
    __syncthreads();
}

// All producer lanes gather directly into the same shared tiles used by the
// original consumers. No global packed input or padded weight is materialized.
// Barrier 3 belongs exclusively to the 128-lane producer warpgroup.
template <bool WEIGHT, uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K,
          uint32_t TOTAL_STAGES>
__device__ void native_linear_producer(barrier bars_ready[], barrier bars_filled[],
    float *buffer, const LinearNativeAccess &access, const LinearNativePath &p,
    uint32_t angle, uint32_t mtile, uint32_t ntile, uint32_t nodes,
    uint32_t u, uint32_t v, uint32_t lane, const int64_t *row_ids) {
    constexpr uint32_t SIZE = WEIGHT ? TILE_K * TILE_N : TILE_M * TILE_K;
    uint32_t stage = 0;
    for (uint32_t k = 0; k < u; k += TILE_K) {
        if (lane == 0) bars_ready[stage].arrive_and_wait();
        asm volatile("bar.sync 3, 128;" : : : "memory");
        #pragma unroll 4
        for (uint32_t i = lane; i < SIZE; i += 128) {
            const float *source = access.x;
            uint32_t offset, valid = 0;
            if constexpr (WEIGHT) {
                const uint32_t ci = k + i / TILE_N, co = ntile * TILE_N + i % TILE_N;
                source = access.w;
                if (ci < p.ci && co < v) {
                    source += p.sw + uint64_t(ci) * v + co;
                    valid = 4;
                }
                offset = i;
            } else {
                const uint32_t row = i / TILE_K, col = i % TILE_K;
                const uint32_t n = mtile * TILE_M + row, ci = k + col;
                if (n < nodes && ci < p.ci) {
                    const int64_t original = row_ids[row];
                    source += original * access.din + p.sx + ci * p.dim + angle;
                    valid = 4;
                }
                offset = row * TILE_K + (((col / 4) ^ (row % 8)) * 4 + col % 4);
            }
            const uint32_t destination = static_cast<uint32_t>(__cvta_generic_to_shared(buffer + stage * SIZE + offset));
            asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;" ::
                "r"(destination), "l"(source), "r"(valid) : "memory");
        }
        asm volatile("cp.async.commit_group; cp.async.wait_group 0;" ::: "memory");
        asm volatile("bar.sync 3, 128;" : : : "memory");
        if (lane == 0) bars_filled[stage].arrive();
        stage = (stage + 1) % TOTAL_STAGES;
    }
    if (lane == 0)
        for (uint32_t i = 0; i < TOTAL_STAGES; ++i)
            bars_ready[(stage + i) % TOTAL_STAGES].arrive_and_wait();
    __syncthreads();
}

// Reuse the existing conflict-free register/shared transpose, then scatter full
// channel rows. Each warp writes one node, reducing fragmented transactions.
template <uint32_t TILE_M, uint32_t TILE_N, uint32_t CONSUMER_WG_NUM>
__device__ void native_linear_store(float *output_smem,
    float accu[(TILE_M / CONSUMER_WG_NUM) / 64][TILE_N / 8][4],
    const LinearNativeAccess &access, const LinearNativePath &p,
    uint32_t angle, uint32_t mtile, uint32_t ntile, uint32_t nodes,
    uint32_t consumer, uint32_t lane, const int64_t *row_ids) {
    constexpr uint32_t ROWS = TILE_M / CONSUMER_WG_NUM;
    float *tile = output_smem + consumer * ROWS * TILE_N;
    for (uint32_t m = 0; m < ROWS / 64; ++m) {
        wg_reg_swap_swizzle4_m64n32f32<TILE_N>(accu[m], lane);
        wg_R2S_swizzle4_m64n32f32<TILE_N>(tile + m * 64 * TILE_N, accu[m], lane);
    }
    if (consumer == 0) asm volatile("bar.sync 1, 128;" ::: "memory");
    else asm volatile("bar.sync 2, 128;" ::: "memory");
    for (uint32_t i = lane; i < ROWS * TILE_N; i += 128) {
        const uint32_t n = mtile * TILE_M + consumer * ROWS + i / TILE_N;
        if (n < nodes) {
            const int64_t original = row_ids[consumer * ROWS + i / TILE_N];
            const uint32_t col = ntile * TILE_N + i % TILE_N;
            access.y[original * access.dout + p.sy + col * p.dim + angle] = tile[i];
        }
    }
}

template <uint32_t TILE_M, uint32_t TILE_N, uint32_t TILE_K, uint32_t A_SMEM_SIZE_PER_STAGE,
          uint32_t B_SMEM_SIZE_PER_TILE, uint32_t PRODUCER_WG_NUM, uint32_t CONSUMER_WG_NUM, uint32_t TOTAL_STAGES>
__device__ void mainloop_consumer(barrier bars_ready[],  // buffer finish consuming barrier
                                  barrier bars_filled[], // buffer finish filling barrier
                                  float t_accu[(TILE_M / CONSUMER_WG_NUM) / 64][(TILE_N / 8)][4], //
                                  float *A_smem,                                                  //
                                  float *B_smem,                                                  //
                                  const uint32_t &U,                                              //
                                  const uint32_t &consumer_wg_id,                                 //
                                  const uint32_t &in_wg_tid                                       //
)
{
    // define wgmma(wgmma.mma_async.m64nNk8)
    constexpr uint32_t WGMMA_M = 64;
    constexpr uint32_t WGMMA_N = TILE_N;
    constexpr uint32_t WGMMA_K = 8;

    uint32_t cur_buffer_id = 0;
    // init (All buffers are empty)
#pragma unroll
    for (uint32_t i = 0; i < TOTAL_STAGES; ++i)
    {
        bars_ready[i].arrive(); // buffer_0 is ready for initial fill
    }
    // loop
    for (uint32_t k_count = 0; k_count < CEIL_DIV(U, TILE_K); ++k_count)
    {
        // Wait for the data to have arrived.
        bars_filled[cur_buffer_id].arrive_and_wait();

        // load A data from SMEM
        float a_reg[(TILE_M / CONSUMER_WG_NUM) / 64][16];
        float a_low[(TILE_M / CONSUMER_WG_NUM) / 64][16];
#pragma unroll
        for (uint32_t _m = 0; _m < ((TILE_M / CONSUMER_WG_NUM) / 64); ++_m)
        {
            warpgroup_ldmatirx_64x32f32_sw128B_sync(
                a_reg[_m],                                                      // out
                (A_smem + cur_buffer_id * A_SMEM_SIZE_PER_STAGE),               // smem_ptr
                (_m + consumer_wg_id * ((TILE_M / CONSUMER_WG_NUM) / 64)) * 64, // smem_y base offset
                in_wg_tid                                                       // warpgroup tid
            );
#pragma unroll
            for (uint32_t _l = 0; _l < 16; ++_l)
            {
                uint32_t tmp;
                asm_cvt_tf32_f32(tmp, a_reg[_m][_l]);
                const float value = a_reg[_m][_l];
                a_reg[_m][_l] = __uint_as_float(tmp);
                asm_cvt_tf32_f32(tmp, value - a_reg[_m][_l]);
                a_low[_m][_l] = __uint_as_float(tmp);
            }
        }

        // do mma
        asm_warpgroup_arrive();
#pragma unroll
        for (uint32_t _k = 0; _k < (TILE_K / WGMMA_K); ++_k)
        {
#pragma unroll
            for (uint32_t _m = 0; _m < ((TILE_M / CONSUMER_WG_NUM) / WGMMA_M); ++_m)
            {
                uint64_t b_desc = make_smem_desc(&B_smem[k_count * B_SMEM_SIZE_PER_TILE + WGMMA_K * _k]);
                asm_wgmma_m64n32k8_tf32<1, 1, 1>(t_accu[_m], a_reg[_m][4 * _k], a_reg[_m][4 * _k + 1],
                                                 a_reg[_m][4 * _k + 2], a_reg[_m][4 * _k + 3], b_desc);
                // Preserve FP32 accuracy using the existing WGMMA primitive:
                // Ahi*Bhi + Alo*Bhi + Ahi*Blo; omitted Alo*Blo is second order.
                asm_wgmma_m64n32k8_tf32<1, 1, 1>(t_accu[_m], a_low[_m][4 * _k], a_low[_m][4 * _k + 1],
                                                 a_low[_m][4 * _k + 2], a_low[_m][4 * _k + 3], b_desc);
                const uint64_t low_desc = make_smem_desc(&B_smem[U * TILE_N + k_count * B_SMEM_SIZE_PER_TILE + WGMMA_K * _k]);
                asm_wgmma_m64n32k8_tf32<1, 1, 1>(t_accu[_m], a_reg[_m][4 * _k], a_reg[_m][4 * _k + 1],
                                                 a_reg[_m][4 * _k + 2], a_reg[_m][4 * _k + 3], low_desc);
            }
            asm_warpgroup_commit_batch();
            asm_warpgroup_wait(0);
        }

        // tell producer that consuming finished
        bars_ready[cur_buffer_id].arrive();

        // switch to next buffer
        cur_buffer_id = (cur_buffer_id + 1) % TOTAL_STAGES;
    }
    // All warp groups must finish this phase before reused barriers/buffers
    // can be initialized for the next one. Arrive alone does not wait.
    __syncthreads();
}

template <uint32_t TILE_M, uint32_t TILE_N, uint32_t CONSUMER_WG_NUM>
__device__ void store_re_async_consumer(
    float *O_smem,                                                  // output smem buffer base addr
    float t_accu[(TILE_M / CONSUMER_WG_NUM) / 64][(TILE_N / 8)][4], // accumulator regs
    const CUtensorMap *out_map,                                     // out tensor map
    const uint32_t &b_mtile_id,                                     // b_mtile_id
    const uint32_t &b_ntile_id,                                     // b_ntile_id
    const uint32_t &b_out_i_id,                                     // b_out_i_id
    const uint32_t &consumer_wg_id,                                 // consumer warpgroup id
    const uint32_t &in_wg_tid                                       // in warpgroup tid
)
{
    constexpr uint32_t SMEM_SIZE_PER_M_TILE = 64 * TILE_N;
    constexpr uint32_t OUT_SMEM_SIZE_PER_CONS_WG = (TILE_M / CONSUMER_WG_NUM) * TILE_N;
    // Swizzle regs within each thread to avoid SMEM bank-conflict.
    // Target layout (Layout from reg to SMEM): swizzle 4 (each element is 32B);
    // Target SMEM layout (Layout from SMEM to GLOBAL): row major, swizzle NONE;
    for (uint32_t _m = 0; _m < (TILE_M / CONSUMER_WG_NUM) / 64; ++_m)
    {
        wg_reg_swap_swizzle4_m64n32f32<TILE_N>(t_accu[_m], in_wg_tid);
        wg_R2S_swizzle4_m64n32f32<TILE_N>(
            (O_smem + consumer_wg_id * OUT_SMEM_SIZE_PER_CONS_WG + _m * SMEM_SIZE_PER_M_TILE), //
            t_accu[_m],                                                                        //
            in_wg_tid                                                                          //
        );
    }

    // Wait for every writer in this consumer warp group before its leader
    // issues the TMA store. Each group owns a separate shared output tile.
    if (consumer_wg_id == 0)
        asm volatile("bar.sync 1, 128;" : : : "memory");
    else
        asm volatile("bar.sync 2, 128;" : : : "memory");

    // Wait for shared memory writes to be visible to TMA engine.
    cde::fence_proxy_async_shared_cta();

    // Send TMA store request
    if (in_wg_tid == 0)
    {
        uint32_t coord_n0 = b_ntile_id * TILE_N;                                               // V
        uint32_t coord_n1 = b_out_i_id;                                                        // in_total_i
        uint32_t coord_n2 = b_mtile_id * TILE_M + consumer_wg_id * (TILE_M / CONSUMER_WG_NUM); // B
        // We transfer a (TILE_M x TIME_N) tile each time
        cde::cp_async_bulk_tensor_3d_shared_to_global(
            out_map,                                              // tensor map
            coord_n0, coord_n1, coord_n2,                         // coord
            (O_smem + consumer_wg_id * OUT_SMEM_SIZE_PER_CONS_WG) // smem start ptr
        );
        // Wait for TMA transfer to have finished reading shared memory.
        // Create a "bulk async-group" out of the previous bulk copy operation.
        cde::cp_async_bulk_commit_group();
    }
}

/* ============================================================= */
/* =================== Main Kernel Functions =================== */
template <uint32_t IN_NUM_PATHS,        // in path数量
          uint32_t OUT_NUM_PATHS,       // out path数量
          uint32_t TILE_M,              // tile m大小
          uint32_t TILE_N,              // tile n大小
          uint32_t TILE_K = 32,         // tile k大小
          uint32_t PRODUCER_WG_NUM = 1, // producer warpgroup数量
          uint32_t CONSUMER_WG_NUM = 2, // consumer warpgroup数量
          uint32_t FUSION = 0          // bit 0: input/weights; bit 1: output
          >
__global__ void mutipath_equi_linear_fwd_f32_tf32_kernel(const __grid_constant__ CUtensorMap x_map,   // x tensor maps
                                                         const __grid_constant__ CUtensorMap w_map,   // w tensor maps
                                                         const __grid_constant__ CUtensorMap out_map, // out tensor maps
                                                         const __grid_constant__ cg_T<IN_NUM_PATHS> cg_vals,                  // val
                                                         const __grid_constant__ idim_T<IN_NUM_PATHS> in_prefex_i_sum,        // [IN_NUM_PATHS]
                                                         const __grid_constant__ idim_T<OUT_NUM_PATHS> out_prefex_i_sum,      // [OUT_NUM_PATHS]
                                                         const __grid_constant__ idim_T<OUT_NUM_PATHS> out_path_count,        // [OUT_NUM_PATHS]
                                                         const __grid_constant__ idim_T<OUT_NUM_PATHS> out_path_in_istart,    // [OUT_NUM_PATHS]
                                                         uint32_t B,                                  // batch size
                                                         uint32_t U,                                  // U
                                                         uint32_t V,                                  // V
                                                         uint32_t in_total_i,                         // input i的总数
                                                         uint32_t out_total_i,                        // output i的总数
                                                         const __grid_constant__ LinearNativeAccess native
)
{
    /* tensor shape:
     * x: [B, in_total_i, U]
     * w: [IN_NUM_PATHS, U, V]
     * out: [B, out_total_i, V]
     */
    // basic info
    constexpr uint32_t WARPGROUP_NUM = PRODUCER_WG_NUM + CONSUMER_WG_NUM;
    constexpr uint32_t PRODUCER_WG_ID = 0;
    uint32_t in_wg_tid = threadIdx.x;
    uint32_t wg_id = threadIdx.y;
    uint32_t b_ntile_id = blockIdx.x;
    uint32_t b_mtile_id = blockIdx.y;
    uint32_t b_out_i_id = blockIdx.z;
    if constexpr (FUSION) {
        b_ntile_id = blockIdx.x % (V / TILE_N);
        b_out_i_id = blockIdx.x / (V / TILE_N);
    }
    // Reuse node addresses throughout all K tiles and the output epilogue.
    __shared__ int64_t row_ids[TILE_M];
    if constexpr (FUSION) {
        if (wg_id == 0) {
            const uint32_t n = b_mtile_id * TILE_M + in_wg_tid;
            row_ids[in_wg_tid] = n < B ? (native.ids ? native.ids[n] : n) : 0;
        }
    }

    // get path info
    uint32_t b_out_path_id = 0;
#pragma unroll
    for (uint32_t n = 0; n < OUT_NUM_PATHS; ++n)
    {
        b_out_path_id = b_out_i_id >= out_prefex_i_sum._i[n] ? n : b_out_path_id;
    }
    LinearNativePath native_path{};
    uint32_t native_angle = 0;
    if constexpr (FUSION) {
        native_path = native.paths[b_out_path_id];
        native_angle = b_out_i_id - out_prefex_i_sum._i[b_out_path_id];
        if constexpr (FUSION & 1) U = CEIL_DIV(native_path.ci, TILE_K) * TILE_K;
    }
    uint32_t b_out_reduce_loop_count = out_path_count._i[b_out_path_id];

    uint32_t b_in_path_id0 = 0;
    for (uint32_t p = 0; p < b_out_path_id; ++p)
    {
        b_in_path_id0 += out_path_count._i[p];
    }

    // define SMEM-buffer
    extern __shared__ __align__(1024) float smem[];
    constexpr uint32_t TOTAL_STAGES = 2; // 流水级
    static_assert((TILE_M * TILE_N) % CONSUMER_WG_NUM == 0);
    constexpr uint32_t OUT_SMEM_SIZE_PER_CONS_WG = (TILE_M * TILE_N) / CONSUMER_WG_NUM;
    constexpr uint32_t A_SMEM_SIZE_PER_STAGE = TILE_M * TILE_K;
    constexpr uint32_t B_SMEM_SIZE_PER_TILE = TILE_N * TILE_K;
    constexpr uint32_t OUT_SMEM_SIZE = OUT_SMEM_SIZE_PER_CONS_WG * CONSUMER_WG_NUM;
    constexpr uint32_t A_SMEM_SIZE = A_SMEM_SIZE_PER_STAGE * TOTAL_STAGES;
    float *A_smem = smem; // must align 1024
    // Weight staging, input staging, and output staging have disjoint
    // lifetimes, separated by the existing CTA barriers. Share their storage.
    static_assert(A_SMEM_SIZE >= OUT_SMEM_SIZE);
    float *O_smem = A_smem;
    float *B_smem = A_smem + A_SMEM_SIZE;

    // define & init barrier
    // for buffer id=s:
    //      bars[0~OUTER_BARRIER_NUMS]: 保留备用
    //      bars[OUTER_BARRIER_NUMS + s]: buffer 's' ready to be filled barrier; (bars_ready)
    //      bars[OUTER_BARRIER_NUMS + TOTAL_STAGES + s]: buffer 's' are filled-in barrier; (bars_filled)
    constexpr uint32_t OUTER_BARRIER_NUMS = 0;
    constexpr uint32_t TOTAL_BARRIER_NUMS = OUTER_BARRIER_NUMS + TOTAL_STAGES * 2;
#pragma nv_diag_suppress static_var_with_dynamic_init
    __shared__ barrier bars[TOTAL_BARRIER_NUMS];
    static_assert(TOTAL_BARRIER_NUMS <= 128); // 不同thread并行初始化barriers
    if (in_wg_tid < TOTAL_BARRIER_NUMS && wg_id == 0)
    {
        // Initialize barrier.
        init(&bars[in_wg_tid], (CONSUMER_WG_NUM * 128 + 1));
        // Make initialized barrier visible in async proxy.
        cde::fence_proxy_async_shared_cta();
    }
    barrier *bars_ready = bars + OUTER_BARRIER_NUMS;
    barrier *bars_filled = bars + OUTER_BARRIER_NUMS + TOTAL_STAGES;
    // Syncthreads so initialized barrier is visible to all threads.
    __syncthreads();

    if (wg_id == PRODUCER_WG_ID)
    {
        // Get b_in_i_id, b_in_path_id
        uint32_t b_in_i_id0 = out_path_in_istart._i[b_out_path_id] + (b_out_i_id - out_prefex_i_sum._i[b_out_path_id]);
        uint32_t b_in_i_stride = (b_out_path_id < OUT_NUM_PATHS - 1)
                                     ? (out_prefex_i_sum._i[b_out_path_id + 1] - out_prefex_i_sum._i[b_out_path_id])
                                     : (out_total_i - out_prefex_i_sum._i[b_out_path_id]);

        // Out reduce loop
        for (uint32_t cnt = 0; cnt < b_out_reduce_loop_count; ++cnt)
        {
            uint32_t b_in_i_id = b_in_i_id0 + b_in_i_stride * cnt;
            uint32_t b_in_path_id = b_in_path_id0 + cnt;

            /* STEP1 (prepare loop): Fetch W tile and transpose (prepare loop)*/
            // We use O_smem as a tmp buffer (buffer size: TILE_M * TILE_N)
            if constexpr (FUSION & 1) {
                native_linear_producer<true, TILE_M, TILE_N, TILE_K, TOTAL_STAGES>(
                    bars_ready, bars_filled, O_smem, native, native_path,
                    native_angle, b_mtile_id, b_ntile_id, B, U, V, in_wg_tid, row_ids);
            } else preloop_producer<TILE_N, TILE_K, TOTAL_STAGES>(bars_ready,   //
                                                           bars_filled,  //
                                                           O_smem,       //
                                                           &w_map,       //
                                                           b_ntile_id,   //
                                                           b_in_path_id, //
                                                           U,            //
                                                           in_wg_tid     //
            );

            /* STEP2: Fetch X tile and do wgmma (main loop) */
            if constexpr (FUSION & 1) {
                native_linear_producer<false, TILE_M, TILE_N, TILE_K, TOTAL_STAGES>(
                    bars_ready, bars_filled, A_smem, native, native_path,
                    native_angle, b_mtile_id, b_ntile_id, B, U, V, in_wg_tid, row_ids);
            } else mainloop_producer<TILE_M, TILE_K, A_SMEM_SIZE_PER_STAGE, TOTAL_STAGES>(bars_ready,  //
                                                                                   bars_filled, //
                                                                                   A_smem,      //
                                                                                   &x_map,      //
                                                                                   b_in_i_id,   //
                                                                                   b_mtile_id,  //
                                                                                   B,           //
                                                                                   in_total_i,  //
                                                                                   U,           //
                                                                                   in_wg_tid    //
            );
        }

        /* STEP3: Store accumulator */
        // Producer do nothing
    }
    else
    {
        uint32_t consumer_wg_id = wg_id - PRODUCER_WG_NUM;
        // define accumulators
        float t_accu[(TILE_M / CONSUMER_WG_NUM) / 64][(TILE_N / 8)][4] = {0.0f};

        // Out reduce loop
        for (uint32_t cnt = 0; cnt < b_out_reduce_loop_count; ++cnt)
        {
            /* STEP1 (prepare loop): Fetch W tile and transpose (prepare loop)*/
            // We use O_smem as a tmp buffer (buffer size: TILE_M * TILE_N)
            uint32_t b_in_path_id = b_in_path_id0 + cnt;
            float cur_cg_val = cg_vals._v[b_in_path_id];
            preloop_consumer<TILE_N, TILE_K, B_SMEM_SIZE_PER_TILE, CONSUMER_WG_NUM, TOTAL_STAGES>(bars_ready,     //
                                                                                                  bars_filled,    //
                                                                                                  B_smem,         //
                                                                                                  O_smem,         //
                                                                                                  cur_cg_val,     //
                                                                                                  U,              //
                                                                                                  consumer_wg_id, //
                                                                                                  in_wg_tid       //
            );

            cde::fence_proxy_async_shared_cta();
            /* STEP2: Fetch X tile and do wgmma (main loop) */
            mainloop_consumer<TILE_M, TILE_N, TILE_K, A_SMEM_SIZE_PER_STAGE, B_SMEM_SIZE_PER_TILE, PRODUCER_WG_NUM,
                              CONSUMER_WG_NUM, TOTAL_STAGES>(bars_ready,     //
                                                             bars_filled,    //
                                                             t_accu,         //
                                                             A_smem,         //
                                                             B_smem,         //
                                                             U,              //
                                                             consumer_wg_id, //
                                                             in_wg_tid       //
            );
        }

        /* STEP3: Store accumulator */
        if constexpr (FUSION & 2) {
            native_linear_store<TILE_M, TILE_N, CONSUMER_WG_NUM>(O_smem, t_accu, native,
                native_path, native_angle, b_mtile_id, b_ntile_id, B, consumer_wg_id, in_wg_tid, row_ids);
        } else {
        store_re_async_consumer<TILE_M, TILE_N, CONSUMER_WG_NUM>(O_smem,         // output smem buffer base addr
                                                                 t_accu,         // accumulator regs
                                                                 &out_map,       // out tensor map
                                                                 b_mtile_id,     // b_mtile_id
                                                                 b_ntile_id,     // b_ntile_id
                                                                 b_out_i_id,     // b_out_i_id
                                                                 consumer_wg_id, // consumer warpgroup id
                                                                 in_wg_tid       // in warpgroup tid
        );
        // Wait for accumulator finishing storing.
        if (in_wg_tid == 0)
        {
            cde::cp_async_bulk_wait_group_read<0>();
        }
        }
    }
}

template <uint32_t IN_NUM_PATHS, uint32_t OUT_NUM_PATHS = 4, uint32_t FUSION = 0>
void mutipath_equi_linear_fwd_f32(float *out,                                 // [B, out_total_i, V]
                                  float *x,                                   // [B, in_total_i, U]
                                  float *w,                                   // [IN_NUM_PATHS, U, V]
                                  const std::vector<int64_t> &in_i_dims_vec,  // in i dims
                                  const std::vector<int64_t> &out_i_dims_vec, // out i dims
                                  const uint32_t &B,                          // batch
                                  const uint32_t &in_total_i,                 // in_total_i
                                  const uint32_t &out_total_i,                // out_total_i
                                  const uint32_t &U,                          // U
                                  const uint32_t &V,                          // V
                                  const std::vector<double> &cg_val_vec,      // cg_val
                                  const cudaStream_t &cur_stream,             // current stream
                                  LinearNativeAccess native = {}
)
{
    constexpr uint32_t PRODUCER_WG_NUM = 1; // producer warpgroup数量
    constexpr uint32_t CONSUMER_WG_NUM = 2; // consumer warpgroup数量
    constexpr uint32_t WARPGROUP_NUM = PRODUCER_WG_NUM + CONSUMER_WG_NUM;
    constexpr uint32_t TILE_K = 32;
    constexpr uint32_t TILE_M = 128;
    constexpr uint32_t TILE_N = 32;

    static_assert(TILE_K == 32);
    if (U == 0 || U % TILE_K || V == 0 || V % TILE_N)
        throw std::invalid_argument("WGMMA Linear channels must be positive multiples of 32");
    if (in_i_dims_vec.empty() || in_i_dims_vec.size() > IN_NUM_PATHS ||
        out_i_dims_vec.empty() || out_i_dims_vec.size() > OUT_NUM_PATHS ||
        cg_val_vec.size() != in_i_dims_vec.size())
        throw std::invalid_argument("Invalid WGMMA Linear path metadata");
    std::vector<int64_t> groups;
    uint32_t in_sum = 0, out_sum = 0;
    for (auto dim : in_i_dims_vec) {
        if (dim <= 0) throw std::invalid_argument("Angular dimensions must be positive");
        in_sum += dim;
        if (groups.empty() || groups.back() != dim) groups.push_back(dim);
    }
    for (auto dim : out_i_dims_vec) out_sum += dim;
    if (groups != out_i_dims_vec || in_sum != in_total_i || out_sum != out_total_i)
        throw std::invalid_argument("WGMMA Linear paths must be grouped by output angular dimension");
    if (B == 0) return;

    idim_T<IN_NUM_PATHS> in_prefex_i_sum{};
    for (uint32_t p = 1; p < in_i_dims_vec.size(); ++p)
        in_prefex_i_sum._i[p] = in_prefex_i_sum._i[p-1] + in_i_dims_vec[p-1];

    idim_T<OUT_NUM_PATHS> out_prefex_i_sum{};
    for (uint32_t p = 1; p < OUT_NUM_PATHS; ++p)
        out_prefex_i_sum._i[p] = p < out_i_dims_vec.size()
            ? out_prefex_i_sum._i[p-1] + out_i_dims_vec[p-1] : out_total_i;

    idim_T<OUT_NUM_PATHS> out_path_count{};
    uint32_t _cur_cnt = 0;
    uint32_t _cur_op = 0;
    uint32_t _last_i_dim = in_i_dims_vec[0];
    for (uint32_t p = 0; p < in_i_dims_vec.size(); ++p)
    {
        if (in_i_dims_vec[p] == _last_i_dim)
        {
            ++_cur_cnt;
        }
        else
        {
            out_path_count._i[_cur_op] = _cur_cnt;
            _cur_cnt = 1;
            ++_cur_op;
            _last_i_dim = in_i_dims_vec[p];
        }
    }
    out_path_count._i[_cur_op] = _cur_cnt;

    idim_T<OUT_NUM_PATHS> out_path_in_istart{};
    uint32_t _cur_i_path = 0;
    for (uint32_t op = 0; op < out_i_dims_vec.size(); ++op)
    {
        out_path_in_istart._i[op] = in_prefex_i_sum._i[_cur_i_path];
        _cur_i_path += out_path_count._i[op];
    }

    cg_T<IN_NUM_PATHS> cg_vals(cg_val_vec);

    CUtensorMap x_map{};
    CUtensorMap w_map{};
    CUtensorMap out_map{};
    if constexpr (!(FUSION & 1)) {
    init_3d_tensormap(&x_map,                                        // tensor_map_ptr
                      static_cast<void *>(x),                        // tensor_ptr
                      B,                                             // B
                      in_total_i,                                    // M
                      U,                                             // N
                      in_total_i * U,                                // stride B
                      U,                                             // stride M
                      TILE_M,                                        // box B
                      1,                                             // box M
                      TILE_K,                                        // box N
                      CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_128B // swizzle 128B
    );
    init_3d_tensormap(&w_map,                 // tensor_map_ptr
                      static_cast<void *>(w), // tensor_ptr
                      in_i_dims_vec.size(),   // B
                      U,                      // M
                      V,                      // N
                      U * V,                  // stride B
                      V,                      // stride M
                      1,                      // box B
                      TILE_K,                 // box M
                      TILE_N                  // box N
    );
    }
    if constexpr (!(FUSION & 2)) {
    init_3d_tensormap(&out_map,                   // tensor_map_ptr
                      static_cast<void *>(out),   // tensor_ptr
                      B,                          // B
                      out_total_i,                // M
                      V,                          // N
                      out_total_i * V,            // stride B
                      V,                          // stride M
                      (TILE_M / CONSUMER_WG_NUM), // box B
                      1,                          // box M
                      TILE_N                      // box N
    );
    }
    dim3 grid(CEIL_DIV(V, TILE_N), CEIL_DIV(B, TILE_M), out_total_i);
    if constexpr (FUSION) grid = dim3(CEIL_DIV(V, TILE_N) * out_total_i, CEIL_DIV(B, TILE_M), 1);
    dim3 block(128, WARPGROUP_NUM);

    constexpr uint32_t STAGES = 2;
    size_t SMEM_SIZE = sizeof(float) * ((TILE_M * TILE_K) * STAGES // A_smem
                                        + (2 * TILE_N * U)         // B high + residual
                                       );

    auto cuda_kernel = mutipath_equi_linear_fwd_f32_tf32_kernel<IN_NUM_PATHS, OUT_NUM_PATHS, TILE_M, TILE_N, TILE_K,
                                                                PRODUCER_WG_NUM, CONSUMER_WG_NUM, FUSION>;
    auto status = cudaFuncSetAttribute(cuda_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, SMEM_SIZE);
    if (status != cudaSuccess) throw std::runtime_error(cudaGetErrorString(status));
    cuda_kernel<<<grid, block, SMEM_SIZE, cur_stream>>>(x_map, w_map, out_map, cg_vals, in_prefex_i_sum,
                                                        out_prefex_i_sum, out_path_count, out_path_in_istart, B, U, V,
                                                        in_total_i, out_total_i, native);
}

// wrapper
void mutipath_equi_linear_fwd_f32_impl(const uint32_t &IN_NUM_PATHS,           // path nums
                                       float *out,                             // [B, out_total_i, V]
                                       float *x,                               // [B, in_total_i, U]
                                       float *w,                               // [IN_NUM_PATHS, U, V]
                                       const uint32_t &B,                      // batch
                                       const uint32_t &total_i,                // in_total_i
                                       const std::vector<int64_t> &i_dims_vec, // i dims
                                       const uint32_t &U,                      // U
                                       const uint32_t &V,                      // V
                                       const std::vector<double> &cg_val_vec,  // cg_val
                                       const cudaStream_t &cur_stream          // current stream
)
{
    const std::vector<int64_t> out_i_dims_vec = {1, 3, 5, 7};
    const uint32_t out_total_i = 16;
    auto call_impl = [&](auto &&...forwarded_args) {
        switch (IN_NUM_PATHS)
        {
        case 4: // small
            mutipath_equi_linear_fwd_f32<4>(std::forward<decltype(forwarded_args)>(forwarded_args)...);
            break;
        case 10: // medium
            mutipath_equi_linear_fwd_f32<10>(std::forward<decltype(forwarded_args)>(forwarded_args)...);
            break;
        case 17: // large
            mutipath_equi_linear_fwd_f32<17>(std::forward<decltype(forwarded_args)>(forwarded_args)...);
            break;
        default:
            throw std::invalid_argument("Unsupported number of paths: " + std::to_string(IN_NUM_PATHS) +
                                        ". Supported values are 4, 10, and 17.");
        }
    };
    // 调用lambda，完美转发参数
    call_impl(out, x, w, i_dims_vec, out_i_dims_vec, B, total_i, out_total_i, U, V, cg_val_vec, cur_stream);
}

// General metadata entry reuses the original TMA/warp-specialized WGMMA kernel.
// Capacities bound launch metadata only; unused output groups are never selected.
void multipath_equi_linear_fwd_f32_general(float *out, float *x, float *w,
    const std::vector<int64_t>& in_dims, const std::vector<int64_t>& out_dims,
    uint32_t batch, uint32_t in_angles, uint32_t out_angles, uint32_t u, uint32_t v,
    const std::vector<double>& coefficients, cudaStream_t stream) {
    mutipath_equi_linear_fwd_f32<32,16>(out,x,w,in_dims,out_dims,batch,
        in_angles,out_angles,u,v,coefficients,stream);
}

void multipath_equi_linear_fwd_f32_native(float *out, float *x, float *w,
    const std::vector<int64_t>& dims, uint32_t batch, uint32_t angles,
    uint32_t u, uint32_t v, const std::vector<double>& coefficients,
    cudaStream_t stream, LinearNativeAccess access, int fusion) {
    switch (fusion) {
    case 1: mutipath_equi_linear_fwd_f32<32,16,1>(out,x,w,dims,dims,batch,angles,angles,u,v,coefficients,stream,access); break;
    case 2: mutipath_equi_linear_fwd_f32<32,16,2>(out,x,w,dims,dims,batch,angles,angles,u,v,coefficients,stream,access); break;
    case 3: mutipath_equi_linear_fwd_f32<32,16,3>(out,x,w,dims,dims,batch,angles,angles,u,v,coefficients,stream,access); break;
    default: throw std::invalid_argument("Invalid WGMMA fusion mode");
    }
}
