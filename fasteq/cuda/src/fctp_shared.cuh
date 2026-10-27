#pragma once

// Extracted from FastEq's fctp_fwd.cu/fctp_bwd.cu shared weight transpose.
// Both the legacy one-hot FCTP and the general UVW adjoints use this loader.
namespace fasteq_fctp {
template <typename T> struct Vector4;
template <> struct Vector4<float> { using type = float4; };
template <> struct Vector4<double> { using type = double4; };

template <typename T>
__device__ __forceinline__ void transpose_four(
    T* shared, int pitch, int row, int column, const T* source) {
    using Vec4 = typename Vector4<T>::type;
    const Vec4 value = *reinterpret_cast<const Vec4*>(source);
    shared[(row + 0) * pitch + column] = value.x;
    shared[(row + 1) * pitch + column] = value.y;
    shared[(row + 2) * pitch + column] = value.z;
    shared[(row + 3) * pitch + column] = value.w;
}

// Contiguity alone does not imply vector alignment (e.g. storage_offset=1).
template <typename T>
__device__ __forceinline__ void transpose_four_safe(
    T* shared, int pitch, int row, int column, const T* source) {
    if ((reinterpret_cast<unsigned long long>(source) % alignof(typename Vector4<T>::type)) == 0) {
        transpose_four(shared,pitch,row,column,source);
    } else {
        #pragma unroll
        for (int i=0;i<4;++i) shared[(row+i)*pitch+column]=source[i];
    }
}
}  // namespace fasteq_fctp
