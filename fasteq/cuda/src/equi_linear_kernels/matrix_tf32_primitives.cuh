#pragma once

// Shared compensated-TF32 primitives extending FastEq TF32 rounding.
// m16n8k8 fragment layout follows the NVIDIA PTX ISA; no external kernel.
__device__ __forceinline__ unsigned matrix_tf32_round(float value) {
    unsigned result;
    asm_cvt_tf32_f32(result, value);
    return result;
}

__device__ __forceinline__ void matrix_tf32_mma(
    float* acc, const unsigned* a, const unsigned* b) {
    asm volatile(
        "mma.sync.aligned.m16n8k8.row.col.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};\n"
        : "+f"(acc[0]), "+f"(acc[1]), "+f"(acc[2]), "+f"(acc[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

