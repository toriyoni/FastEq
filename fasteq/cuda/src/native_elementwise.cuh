#pragma once

// Extend FastEq's original 16-byte FETCH_16B packet schedule to elementwise
// addition and the scalar-irrep bias map. Strided/tail readers are supplied by
// the owning plan; no tensor library executes the arithmetic.
using ElementT = FASTEQ_ELEMENT_TYPE;
using ElementIndex = long long;
#if FASTEQ_ELEMENT_FP64
using ElementPacket = double2;
constexpr int ElementItems = 2;
#else
using ElementPacket = float4;
constexpr int ElementItems = 4;
#endif

__device__ __forceinline__ ElementT element_add(ElementT a, ElementT b) {
#if FASTEQ_ELEMENT_FP64
    return __dadd_rn(a, b);
#else
    return __fadd_rn(a, b);
#endif
}

extern "C" __global__ void FASTEQ_ELEMENT_ENTRY(
    const ElementT* a, const ElementT* b, ElementT* output, ElementIndex count) {
    const ElementIndex packets = (count + ElementItems - 1) / ElementItems;
    for (ElementIndex packet = ElementIndex(blockIdx.x) * blockDim.x + threadIdx.x;
         packet < packets; packet += ElementIndex(gridDim.x) * blockDim.x) {
        const ElementIndex base = packet * ElementItems;
        ElementPacket result, left, right;
        ElementT* values = reinterpret_cast<ElementT*>(&result);
        const ElementT* lvalues = reinterpret_cast<const ElementT*>(&left);
        const ElementT* rvalues = reinterpret_cast<const ElementT*>(&right);
#if FASTEQ_ELEMENT_LEFT_CONTIGUOUS
        const bool vector_left = base + ElementItems <= count &&
            (reinterpret_cast<unsigned long long>(a) & 15) == 0;
        if (vector_left) left = reinterpret_cast<const ElementPacket*>(a)[packet];
#endif
#if FASTEQ_ELEMENT_RIGHT_CONTIGUOUS
        const bool vector_right = base + ElementItems <= count &&
            (reinterpret_cast<unsigned long long>(b) & 15) == 0;
        if (vector_right) right = reinterpret_cast<const ElementPacket*>(b)[packet];
#endif
        #pragma unroll
        for (int item = 0; item < ElementItems; ++item) {
            const ElementIndex index = base + item;
            if (index < count) {
                ElementT av = ElementT(0), bv = ElementT(0);
#if FASTEQ_ELEMENT_HAS_LEFT
#if FASTEQ_ELEMENT_LEFT_CONTIGUOUS
                av = vector_left ? lvalues[item] : fasteq_element_left(a, index);
#else
                av = fasteq_element_left(a, index);
#endif
#endif
#if FASTEQ_ELEMENT_HAS_RIGHT
#if FASTEQ_ELEMENT_RIGHT_CONTIGUOUS
                bv = vector_right ? rvalues[item] : fasteq_element_right(b, index);
#else
                bv = fasteq_element_right(b, index);
#endif
#endif
                values[item] = element_add(av, bv);
            }
        }
        if (base + ElementItems <= count)
            reinterpret_cast<ElementPacket*>(output)[packet] = result;
        else {
            #pragma unroll
            for (int item = 0; item < ElementItems; ++item)
                if (base + item < count) output[base + item] = values[item];
        }
    }
}
