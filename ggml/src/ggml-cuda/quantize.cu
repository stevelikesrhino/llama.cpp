#include "quantize.cuh"
#include "unary.cuh"
#include <cstdint>

#if defined(BLACKWELL_MMA_AVAILABLE)
struct __builtin_align__(32) ggml_cuda_float8 {
    float x0;
    float x1;
    float x2;
    float x3;
    float x4;
    float x5;
    float x6;
    float x7;
};

template <bool use_aligned_float8>
static __device__ __forceinline__ void ggml_cuda_load_nvfp4_values(
        const float * __restrict__ x, const int64_t i0, const int64_t ne00,
        float (&vals0)[8], float (&vals1)[8]) {
    if constexpr (use_aligned_float8) {
        const ggml_cuda_float8 z = {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f};
        const ggml_cuda_float8 v0 = i0 +  7 < ne00 ? reinterpret_cast<const ggml_cuda_float8 *>(x + i0)[0] : z;
        const ggml_cuda_float8 v1 = i0 + 15 < ne00 ? reinterpret_cast<const ggml_cuda_float8 *>(x + i0 + 8)[0] : z;
        vals0[0] = v0.x0; vals0[1] = v0.x1; vals0[2] = v0.x2; vals0[3] = v0.x3;
        vals0[4] = v0.x4; vals0[5] = v0.x5; vals0[6] = v0.x6; vals0[7] = v0.x7;
        vals1[0] = v1.x0; vals1[1] = v1.x1; vals1[2] = v1.x2; vals1[3] = v1.x3;
        vals1[4] = v1.x4; vals1[5] = v1.x5; vals1[6] = v1.x6; vals1[7] = v1.x7;
    } else {
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            vals0[k] = i0 + k     < ne00 ? x[i0 + k]     : 0.0f;
            vals1[k] = i0 + k + 8 < ne00 ? x[i0 + k + 8] : 0.0f;
        }
    }
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)

template <bool has_scale>
__launch_bounds__(CUDA_QUANTIZE_BLOCK_SIZE, 1)
static __global__ void quantize_q8_1(
        const float * x_ptr, void * vy_ptr, const float * scale_activation,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const uint32_t ne1, const uint3 ne2) {
    ggml_cuda_pdl_lc();
    const float * GGML_CUDA_RESTRICT x  = x_ptr;
    void        * GGML_CUDA_RESTRICT vy = vy_ptr;
    const int64_t i0 = (int64_t)blockDim.x*blockIdx.x + threadIdx.x;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i3 = fastdiv(blockIdx.z, ne2);
    const int64_t i2 = blockIdx.z - i3*ne2.z;
    const int64_t i1 = blockIdx.y;

    const int64_t & i00 = i0;
    const int64_t & i01 = i1;
    const int64_t & i02 = i2;
    const int64_t & i03 = i3;

    const int64_t i_cont = ((i3*ne2.z + i2) * ne1 + i1) * ne0 + i0;

    block_q8_1 * y = (block_q8_1 *) vy;

    const int64_t ib  = i_cont / QK8_1; // block index
    const int64_t iqs = i_cont % QK8_1; // quant index

    float inv_input_scale = 1.0f;
    if constexpr (has_scale) {
        float input_scale = scale_activation[0];
        if (!(input_scale != 0.0f) || !isfinite(input_scale)) {
            input_scale = 1.0f;
        }
        inv_input_scale = 1.0f / input_scale;
    }

    ggml_cuda_pdl_sync();
    const float xi = i0 < ne00 ? x[i03*s03 + i02*s02 + i01*s01 + i00] * inv_input_scale : 0.0f;
    float amax = fabsf(xi);
    float sum = xi;

    amax = warp_reduce_max<QK8_1>(amax);
    sum  = warp_reduce_sum<QK8_1>(sum);

    const float  d = amax / 127.0f;
    const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

    y[ib].qs[iqs] = q;

    if (iqs > 0) {
        return;
    }

    y[ib].ds = make_half2(d, sum);
}

__device__ __forceinline__ uint8_t compute_e8m0_scale(float amax) {
    if (!(amax > 0.0f)) {
        return 0;
    }

    // FP4 E2M1: max exponent (unbiased) is 2.
    constexpr int FP4_E2M1_EMAX = 2;

    const float e = log2f(amax);

    // "even" -> round-to-nearest integer, ties-to-even
    const int e_int = __float2int_rn(e);

    const int shared_exp = e_int - FP4_E2M1_EMAX;

    int biased = shared_exp + 127;

    biased = max(biased, 0);
    biased = min(biased, 254);

    return static_cast<uint8_t>(biased);
}

static __device__ __forceinline__ float ggml_cuda_fp4x2_mse(
        const uint32_t fp16x2, const float scale, const float x0, const float x1) {
    const float q0 = __half2float(__ushort_as_half((uint16_t) (fp16x2 & 0xFFFFu))) * scale;
    const float q1 = __half2float(__ushort_as_half((uint16_t) (fp16x2 >> 16))) * scale;
    const float e0 = q0 - x0;
    const float e1 = q1 - x1;
    return e0 * e0 + e1 * e1;
}

static __device__ __forceinline__ uint32_t ggml_cuda_fp32x8_to_fp4_e2m1_mse(
        const float (&x)[8], const float inv_scale, const float dequant_scale, float & err) {
    float v[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        v[k] = x[k] * inv_scale;
    }

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= GGML_CUDA_CC_BLACKWELL
    uint32_t packed;
    uint32_t fp16x2_0;
    uint32_t fp16x2_1;
    uint32_t fp16x2_2;
    uint32_t fp16x2_3;
    asm volatile(
        "{\n"
        ".reg .b8 byte0;\n"
        ".reg .b8 byte1;\n"
        ".reg .b8 byte2;\n"
        ".reg .b8 byte3;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte0, %6, %5;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte1, %8, %7;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte2, %10, %9;\n"
        "cvt.rn.satfinite.e2m1x2.f32   byte3, %12, %11;\n"
        "mov.b32 %0, {byte0, byte1, byte2, byte3};\n"
        "cvt.rn.f16x2.e2m1x2 %1, byte0;\n"
        "cvt.rn.f16x2.e2m1x2 %2, byte1;\n"
        "cvt.rn.f16x2.e2m1x2 %3, byte2;\n"
        "cvt.rn.f16x2.e2m1x2 %4, byte3;\n"
        "}"
        : "=r"(packed), "=r"(fp16x2_0), "=r"(fp16x2_1), "=r"(fp16x2_2), "=r"(fp16x2_3)
        : "f"(v[0]), "f"(v[1]), "f"(v[2]), "f"(v[3]), "f"(v[4]), "f"(v[5]), "f"(v[6]), "f"(v[7]));
    err += ggml_cuda_fp4x2_mse(fp16x2_0, dequant_scale, x[0], x[1]);
    err += ggml_cuda_fp4x2_mse(fp16x2_1, dequant_scale, x[2], x[3]);
    err += ggml_cuda_fp4x2_mse(fp16x2_2, dequant_scale, x[4], x[5]);
    err += ggml_cuda_fp4x2_mse(fp16x2_3, dequant_scale, x[6], x[7]);
    return packed;
#else
    uint32_t packed = 0;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const uint8_t q = ggml_cuda_float_to_fp4_e2m1(v[k], 1.0f);
        const float dq = 0.5f * dequant_scale * (float) kvalues_mxfp4[q];
        const float e = dq - x[k];
        err += e * e;
        packed |= (uint32_t) q << (4 * k);
    }
    return packed;
#endif
}

static __device__ __forceinline__ float ggml_cuda_nvfp4_quantize_code_mse(
        const float (&vals0)[8], const float (&vals1)[8], const uint8_t fp8_code,
        uint32_t & qs0, uint32_t & qs1) {
    const float subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);
    const float inv_scale      = subblock_scale > 0.0f ? 0.5f / subblock_scale : 0.0f;
    const float dequant_scale  = 2.0f * subblock_scale;

    float err = 0.0f;
    qs0 = ggml_cuda_fp32x8_to_fp4_e2m1_mse(vals0, inv_scale, dequant_scale, err);
    qs1 = ggml_cuda_fp32x8_to_fp4_e2m1_mse(vals1, inv_scale, dequant_scale, err);
    return err;
}

static __device__ __forceinline__ void ggml_cuda_nvfp4_quantize_4o6_mse(
        const float (&vals0)[8], const float (&vals1)[8], const float sub_max,
        uint32_t & qs0, uint32_t & qs1, uint8_t & fp8_code) {
    constexpr float SCALE_EPS = 0.001953125f;

    const float scale6_f = fmaxf(sub_max * (1.0f / 6.0f), SCALE_EPS);
    const uint8_t fp8_code6 = ggml_cuda_fp32_to_ue4m3(scale6_f);
    uint32_t qs6_0;
    uint32_t qs6_1;
    const float err6 = ggml_cuda_nvfp4_quantize_code_mse(vals0, vals1, fp8_code6, qs6_0, qs6_1);

    const float scale4_f = fmaxf(sub_max * 0.25f, SCALE_EPS);
    const uint8_t fp8_code4 = ggml_cuda_fp32_to_ue4m3(scale4_f);
    uint32_t qs4_0;
    uint32_t qs4_1;
    const float err4 = ggml_cuda_nvfp4_quantize_code_mse(vals0, vals1, fp8_code4, qs4_0, qs4_1);

    const bool use4 = err4 < err6;
    qs0 = use4 ? qs4_0 : qs6_0;
    qs1 = use4 ? qs4_1 : qs6_1;
    fp8_code = use4 ? fp8_code4 : fp8_code6;
}

static __device__ __forceinline__ void ggml_cuda_nvfp4_quantize_4o6_residual(
        const float (&vals0)[8], const float (&vals1)[8], const float sub_max,
        uint32_t & plane0_0, uint32_t & plane0_1,
        uint32_t & plane1_0, uint32_t & plane1_1,
        uint8_t & fp8_code) {
    ggml_cuda_nvfp4_quantize_4o6_mse(vals0, vals1, sub_max, plane0_0, plane0_1, fp8_code);

    const float subblock_scale = ggml_cuda_ue4m3_to_fp32(fp8_code);
    const float dequant_scale  = 2.0f * subblock_scale;
    float residual0[8];
    float residual1[8];
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float q0 = 0.5f * (float) kvalues_mxfp4[(plane0_0 >> (4 * k)) & 0x0Fu];
        const float q1 = 0.5f * (float) kvalues_mxfp4[(plane0_1 >> (4 * k)) & 0x0Fu];
        residual0[k] = 12.0f * (vals0[k] / dequant_scale - q0);
        residual1[k] = 12.0f * (vals1[k] / dequant_scale - q1);
    }

    float residual_err = 0.0f;
    plane1_0 = ggml_cuda_fp32x8_to_fp4_e2m1_mse(residual0, 1.0f, 1.0f, residual_err);
    plane1_1 = ggml_cuda_fp32x8_to_fp4_e2m1_mse(residual1, 1.0f, 1.0f, residual_err);
}

bool ggml_cuda_can_quantize_nvfp4_glu(const ggml_tensor * src) {
    if (src == nullptr || src->op != GGML_OP_GLU || ggml_get_glu_op(src) != GGML_GLU_OP_SWIGLU) {
        return false;
    }

    const ggml_tensor * gate = src->src[0];
    const ggml_tensor * up   = src->src[1];
    if (gate == nullptr || up == nullptr) {
        return false;
    }

    if (src->type != GGML_TYPE_F32 || gate->type != GGML_TYPE_F32 || up->type != GGML_TYPE_F32) {
        return false;
    }

    if (ggml_get_op_params_i32(src, 1) != 0) {
        return false;
    }

    if (!ggml_are_same_shape(src, gate) || !ggml_are_same_shape(src, up)) {
        return false;
    }

    if (!ggml_is_contiguous_1(gate) || !ggml_is_contiguous_1(up)) {
        return false;
    }

    return gate->nb[0] == ggml_element_size(gate) && up->nb[0] == ggml_element_size(up);
}

template <bool has_ids, bool has_scale, bool use_aligned_float8>
static __global__ void quantize_mmq_nvfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          const int32_t * __restrict__ ids_expert,
                                          void * __restrict__ vy,
                                          const float * __restrict__ scale_activation,
                                          const int64_t scale_activation_ne,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int64_t ne1,
                                          const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * QK_NVFP4_SUB;
    if (i0_base >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;
    float inv_input_scale = 1.0f;
    if constexpr (has_scale) {
        const int64_t scale_idx = scale_activation_ne <= 1 ? 0 : (has_ids && ids_expert ? ids_expert[i1] : i01);
        float input_scale = scale_activation[scale_idx];
        if (!(input_scale != 0.0f) || !isfinite(input_scale)) {
            input_scale = 1.0f;
        }
        inv_input_scale = 1.0f / input_scale;
    }

    const int64_t k_block = i0_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) {
        return;
    }
    const int64_t batch_offset = (int64_t) blockIdx.z * (blocks_per_col * ne1);
    block_nvfp4_mmq * yb = (block_nvfp4_mmq *) vy + batch_offset + k_block * ne1 + blockIdx.x;
    const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;

    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    float vals0[8];
    float vals1[8];
    ggml_cuda_load_nvfp4_values<use_aligned_float8>(x + base_idx, i0_base, ne00, vals0, vals1);

    float sub_max = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        vals0[k] *= inv_input_scale;
        sub_max = fmaxf(sub_max, fabsf(vals0[k]));
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        vals1[k] *= inv_input_scale;
        sub_max = fmaxf(sub_max, fabsf(vals1[k]));
    }

    uint32_t qs0;
    uint32_t qs1;
    uint8_t fp8_code;
    ggml_cuda_nvfp4_quantize_4o6_mse(vals0, vals1, sub_max, qs0, qs1, fp8_code);
    yb->qs_u32[2 * sub + 0] = qs0;
    yb->qs_u32[2 * sub + 1] = qs1;
    reinterpret_cast<uint8_t *>(yb->sc4_u32)[sub] = fp8_code;
#else
    NO_DEVICE_CODE; // This is for Blackwell NVFP4 activations only.
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <bool has_ids, bool use_aligned_float8>
static __global__ void quantize_mmq_nvfp4_dynamic(const float * __restrict__ x,
                                                  const int32_t * __restrict__ ids,
                                                  void * __restrict__ vy,
                                                  float * __restrict__ scale_dynamic,
                                                  const int64_t ne00,
                                                  const int64_t s01,
                                                  const int64_t s02,
                                                  const int64_t s03,
                                                  const int64_t ne0,
                                                  const int64_t ne1,
                                                  const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y % ne2;
    const int64_t i3 = blockIdx.y / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;
    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
    const float * __restrict__ x_row = x + base_idx;

    float amax = 0.0f;
    if constexpr (use_aligned_float8) {
        for (int64_t i0 = 8 * threadIdx.x; i0 < ne00; i0 += 8 * blockDim.x) {
            const ggml_cuda_float8 v = reinterpret_cast<const ggml_cuda_float8 *>(x_row + i0)[0];
            amax = fmaxf(amax, fabsf(v.x0));
            amax = fmaxf(amax, fabsf(v.x1));
            amax = fmaxf(amax, fabsf(v.x2));
            amax = fmaxf(amax, fabsf(v.x3));
            amax = fmaxf(amax, fabsf(v.x4));
            amax = fmaxf(amax, fabsf(v.x5));
            amax = fmaxf(amax, fabsf(v.x6));
            amax = fmaxf(amax, fabsf(v.x7));
        }
    } else {
        for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
            amax = fmaxf(amax, fabsf(x_row[i0]));
        }
    }

    amax = warp_reduce_max<WARP_SIZE>(amax);
    __shared__ float warp_amax[CUDA_QUANTIZE_BLOCK_SIZE / WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_amax[warp] = amax;
    }
    __syncthreads();

    if (warp == 0) {
        amax = lane < CUDA_QUANTIZE_BLOCK_SIZE / WARP_SIZE ? warp_amax[lane] : 0.0f;
        amax = warp_reduce_max<WARP_SIZE>(amax);
        if (lane == 0) {
            warp_amax[0] = amax / (6.0f * 448.0f);
            scale_dynamic[(int64_t) blockIdx.y * ne1 + i1] = warp_amax[0];
        }
    }
    __syncthreads();

    const float row_scale = warp_amax[0];
    const float inv_row_scale = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    const int64_t n_subblocks = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;
    block_nvfp4_mmq * y = (block_nvfp4_mmq *) vy;

    for (int64_t isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
        const int64_t i0_base = isb * QK_NVFP4_SUB;
        const int64_t k_block = i0_base / QK_K;
        const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;
        const int64_t batch_offset = (int64_t) blockIdx.y * (blocks_per_col * ne1);
        block_nvfp4_mmq * yb = y + batch_offset + k_block * ne1 + i1;

        float vals0[8];
        float vals1[8];
        ggml_cuda_load_nvfp4_values<use_aligned_float8>(x_row, i0_base, ne00, vals0, vals1);
        float sub_max = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            vals0[k] *= inv_row_scale;
            vals1[k] *= inv_row_scale;
            sub_max = fmaxf(sub_max, fabsf(vals0[k]));
            sub_max = fmaxf(sub_max, fabsf(vals1[k]));
        }

        uint32_t qs0;
        uint32_t qs1;
        uint8_t fp8_code;
        ggml_cuda_nvfp4_quantize_4o6_mse(vals0, vals1, sub_max, qs0, qs1, fp8_code);
        yb->qs_u32[2 * sub + 0] = qs0;
        yb->qs_u32[2 * sub + 1] = qs1;
        reinterpret_cast<uint8_t *>(yb->sc4_u32)[sub] = fp8_code;
    }
#else
    GGML_UNUSED_VARS(x, ids, vy, scale_dynamic, ne00, s01, s02, s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <bool has_scale, bool dense_2d, bool use_aligned_float8>
static __global__ void quantize_mmq_nvfp4_glu(const float * __restrict__ gate,
                                              const float * __restrict__ up,
                                              void * __restrict__ vy,
                                              const float * __restrict__ scale_activation,
                                              const int64_t scale_activation_ne,
                                              const int64_t ne00,
                                              const int64_t gate_s01,
                                              const int64_t gate_s02,
                                              const int64_t gate_s03,
                                              const int64_t up_s01,
                                              const int64_t up_s02,
                                              const int64_t up_s03,
                                              const int64_t ne0,
                                              const int64_t ne1,
                                              const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i0_base = ((int64_t) blockDim.x * blockIdx.y + threadIdx.x) * QK_NVFP4_SUB;
    if (i0_base >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    int64_t gate_base_idx = i1 * gate_s01;
    int64_t up_base_idx   = i1 * up_s01;
    if constexpr (!dense_2d) {
        const int64_t i2 = blockIdx.z % ne2;
        const int64_t i3 = blockIdx.z / ne2;
        gate_base_idx += i3 * gate_s03 + i2 * gate_s02;
        up_base_idx   += i3 * up_s03   + i2 * up_s02;
    }
    float inv_input_scale = 1.0f;
    if constexpr (has_scale) {
        const int64_t scale_idx = scale_activation_ne <= 1 ? 0 : i1;
        float input_scale = scale_activation[scale_idx];
        if (!(input_scale != 0.0f) || !isfinite(input_scale)) {
            input_scale = 1.0f;
        }
        inv_input_scale = 1.0f / input_scale;
    }

    const int64_t k_block = i0_base / QK_K;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    if (k_block >= blocks_per_col) {
        return;
    }

    int64_t batch_offset = 0;
    if constexpr (!dense_2d) {
        batch_offset = (int64_t) blockIdx.z * (blocks_per_col * ne1);
    }
    block_nvfp4_mmq * yb = (block_nvfp4_mmq *) vy + batch_offset + k_block * ne1 + blockIdx.x;
    const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;

    ggml_cuda_pdl_sync();
    float gate0[8];
    float gate1[8];
    float up0[8];
    float up1[8];
    ggml_cuda_load_nvfp4_values<use_aligned_float8>(gate + gate_base_idx, i0_base, ne00, gate0, gate1);
    ggml_cuda_load_nvfp4_values<use_aligned_float8>(up   + up_base_idx,   i0_base, ne00, up0,   up1);

    float vals0[8];
    float vals1[8];
    float sub_max = 0.0f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float v = ggml_cuda_op_silu_single(gate0[k]) * up0[k] * inv_input_scale;
        vals0[k] = v;
        sub_max = fmaxf(sub_max, fabsf(v));
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float v = ggml_cuda_op_silu_single(gate1[k]) * up1[k] * inv_input_scale;
        vals1[k] = v;
        sub_max = fmaxf(sub_max, fabsf(v));
    }

    uint32_t qs0;
    uint32_t qs1;
    uint8_t fp8_code;
    ggml_cuda_nvfp4_quantize_4o6_mse(vals0, vals1, sub_max, qs0, qs1, fp8_code);
    yb->qs_u32[2 * sub + 0] = qs0;
    yb->qs_u32[2 * sub + 1] = qs1;
    reinterpret_cast<uint8_t *>(yb->sc4_u32)[sub] = fp8_code;
#else
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <bool dense_2d, bool use_aligned_float8>
static __global__ void quantize_mmq_nvfp4_glu_dynamic(const float * __restrict__ gate,
                                                      const float * __restrict__ up,
                                                      void * __restrict__ vy,
                                                      float * __restrict__ scale_dynamic,
                                                      const int64_t ne00,
                                                      const int64_t gate_s01,
                                                      const int64_t gate_s02,
                                                      const int64_t gate_s03,
                                                      const int64_t up_s01,
                                                      const int64_t up_s02,
                                                      const int64_t up_s03,
                                                      const int64_t ne0,
                                                      const int64_t ne1,
                                                      const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i1 = blockIdx.x;
    int64_t gate_base_idx = i1 * gate_s01;
    int64_t up_base_idx   = i1 * up_s01;
    if constexpr (!dense_2d) {
        const int64_t i2 = blockIdx.y % ne2;
        const int64_t i3 = blockIdx.y / ne2;
        gate_base_idx += i3 * gate_s03 + i2 * gate_s02;
        up_base_idx   += i3 * up_s03   + i2 * up_s02;
    }
    const float * __restrict__ gate_row = gate + gate_base_idx;
    const float * __restrict__ up_row   = up   + up_base_idx;

    float amax = 0.0f;
    if constexpr (use_aligned_float8) {
        for (int64_t i0 = 8 * threadIdx.x; i0 < ne00; i0 += 8 * blockDim.x) {
            const ggml_cuda_float8 g = reinterpret_cast<const ggml_cuda_float8 *>(gate_row + i0)[0];
            const ggml_cuda_float8 u = reinterpret_cast<const ggml_cuda_float8 *>(up_row   + i0)[0];
            amax = fmaxf(amax, fabsf(ggml_cuda_op_silu_single(g.x0) * u.x0));
            amax = fmaxf(amax, fabsf(ggml_cuda_op_silu_single(g.x1) * u.x1));
            amax = fmaxf(amax, fabsf(ggml_cuda_op_silu_single(g.x2) * u.x2));
            amax = fmaxf(amax, fabsf(ggml_cuda_op_silu_single(g.x3) * u.x3));
            amax = fmaxf(amax, fabsf(ggml_cuda_op_silu_single(g.x4) * u.x4));
            amax = fmaxf(amax, fabsf(ggml_cuda_op_silu_single(g.x5) * u.x5));
            amax = fmaxf(amax, fabsf(ggml_cuda_op_silu_single(g.x6) * u.x6));
            amax = fmaxf(amax, fabsf(ggml_cuda_op_silu_single(g.x7) * u.x7));
        }
    } else {
        for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
            amax = fmaxf(amax, fabsf(ggml_cuda_op_silu_single(gate_row[i0]) * up_row[i0]));
        }
    }

    amax = warp_reduce_max<WARP_SIZE>(amax);
    __shared__ float warp_amax[CUDA_QUANTIZE_BLOCK_SIZE / WARP_SIZE];
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    if (lane == 0) {
        warp_amax[warp] = amax;
    }
    __syncthreads();
    if (warp == 0) {
        amax = lane < CUDA_QUANTIZE_BLOCK_SIZE / WARP_SIZE ? warp_amax[lane] : 0.0f;
        amax = warp_reduce_max<WARP_SIZE>(amax);
        if (lane == 0) {
            warp_amax[0] = amax / (6.0f * 448.0f);
            scale_dynamic[(int64_t) blockIdx.y * ne1 + i1] = warp_amax[0];
        }
    }
    __syncthreads();

    const float row_scale = warp_amax[0];
    const float inv_row_scale = row_scale > 0.0f ? 1.0f / row_scale : 0.0f;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    const int64_t n_subblocks = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;
    block_nvfp4_mmq * y = (block_nvfp4_mmq *) vy;

    for (int64_t isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
        const int64_t i0_base = isb * QK_NVFP4_SUB;
        const int64_t k_block = i0_base / QK_K;
        const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;
        const int64_t batch_offset = (int64_t) blockIdx.y * (blocks_per_col * ne1);
        block_nvfp4_mmq * yb = y + batch_offset + k_block * ne1 + i1;

        float gate0[8];
        float gate1[8];
        float up0[8];
        float up1[8];
        ggml_cuda_load_nvfp4_values<use_aligned_float8>(gate_row, i0_base, ne00, gate0, gate1);
        ggml_cuda_load_nvfp4_values<use_aligned_float8>(up_row,   i0_base, ne00, up0,   up1);
        float vals0[8];
        float vals1[8];
        float sub_max = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            vals0[k] = ggml_cuda_op_silu_single(gate0[k]) * up0[k] * inv_row_scale;
            vals1[k] = ggml_cuda_op_silu_single(gate1[k]) * up1[k] * inv_row_scale;
            sub_max = fmaxf(sub_max, fabsf(vals0[k]));
            sub_max = fmaxf(sub_max, fabsf(vals1[k]));
        }

        uint32_t qs0;
        uint32_t qs1;
        uint8_t fp8_code;
        ggml_cuda_nvfp4_quantize_4o6_mse(vals0, vals1, sub_max, qs0, qs1, fp8_code);
        yb->qs_u32[2 * sub + 0] = qs0;
        yb->qs_u32[2 * sub + 1] = qs1;
        reinterpret_cast<uint8_t *>(yb->sc4_u32)[sub] = fp8_code;
    }
#else
    GGML_UNUSED_VARS(gate, up, vy, scale_dynamic, ne00, gate_s01, gate_s02, gate_s03,
                     up_s01, up_s02, up_s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <bool has_ids, bool has_scale, bool dynamic_scale>
static __global__ void quantize_mmq_nvfp4_w4a8(
        const float * __restrict__ x,
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ ids_expert,
        block_nvfp4_w4a8_mmq * __restrict__ y,
        const float * __restrict__ scale_activation,
        float * __restrict__ scale_dynamic,
        const int64_t scale_activation_ne,
        const int64_t ne00,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i1  = blockIdx.x;
    const int64_t i2  = blockIdx.y % ne2;
    const int64_t i3  = blockIdx.y / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;
    const float * __restrict__ x_row = x + i3 * s03 + i2 * s02 + i01 * s01;

    GGML_UNUSED(scale_dynamic);
    float inv_input_scale = 1.0f;
    if constexpr (has_scale) {
        const int64_t scale_idx = scale_activation_ne <= 1 ? 0 :
            (has_ids && ids_expert ? ids_expert[i1] : i01);
        float input_scale = scale_activation[scale_idx];
        if (!(input_scale > 0.0f) || !isfinite(input_scale)) {
            input_scale = 1.0f;
        }
        inv_input_scale = 1.0f / input_scale;
    }

    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    const int64_t batch_offset = (int64_t) blockIdx.y * blocks_per_col * ne1;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;
    const int64_t nfrags = (ne0 + 31) / 32;
    const int64_t frag_stride = (int64_t) gridDim.z * nwarps;
    for (int64_t frag = (int64_t) blockIdx.z * nwarps + warp; frag < nfrags; frag += frag_stride) {
        const int64_t i0 = 32 * frag + 2 * lane;
        const float2 values = lane < 16 ? make_float2(
            i0     < ne00 ? x_row[i0]     * inv_input_scale : 0.0f,
            i0 + 1 < ne00 ? x_row[i0 + 1] * inv_input_scale : 0.0f) : make_float2(0.0f, 0.0f);
        const float amax = warp_reduce_max<WARP_SIZE>(fmaxf(fabsf(values.x), fabsf(values.y)));
        uint32_t scale_code = lane == 0 ?
            (uint32_t) __nv_cvt_float_to_e8m0(amax / 448.0f, __NV_SATFINITE, cudaRoundPosInf) : 0;
        scale_code = __shfl_sync(0xFFFFFFFFu, scale_code, 0);
        const float inv_scale = scale_code > 0 ? ldexpf(1.0f, 127 - (int) scale_code) : 0.0f;

        const int64_t k_block = frag / 8;
        const int frag_idx = frag % 8;
        block_nvfp4_w4a8_mmq * __restrict__ yb = y + batch_offset + k_block * ne1 + i1;
        if (lane == 0) {
            reinterpret_cast<uint8_t *>(yb->sc8_u32)[frag_idx] = (uint8_t) scale_code;
        }
        if (lane < 16 && i0 < ne0) {
            const __nv_fp8x2_storage_t packed = __nv_cvt_float2_to_fp8x2(
                make_float2(values.x * inv_scale, values.y * inv_scale), __NV_SATFINITE, __NV_E4M3);
            uint8_t * __restrict__ dst = yb->qs + frag_idx * 32 + 2 * lane;
            if (i0 + 1 < ne0) {
                *reinterpret_cast<__nv_fp8x2_storage_t *>(dst) = packed;
            } else {
                dst[0] = static_cast<uint8_t>(packed);
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, ids, ids_expert, y, scale_activation, scale_dynamic, scale_activation_ne,
                     ne00, s01, s02, s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <bool has_scale, bool dynamic_scale>
static __global__ void quantize_mmq_nvfp4_w4a8_glu(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        block_nvfp4_w4a8_mmq * __restrict__ y,
        const float * __restrict__ scale_activation,
        float * __restrict__ scale_dynamic,
        const int64_t scale_activation_ne,
        const int64_t ne00,
        const int64_t gate_s01,
        const int64_t gate_s02,
        const int64_t gate_s03,
        const int64_t up_s01,
        const int64_t up_s02,
        const int64_t up_s03,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y % ne2;
    const int64_t i3 = blockIdx.y / ne2;
    const float * __restrict__ gate_row = gate + i3 * gate_s03 + i2 * gate_s02 + i1 * gate_s01;
    const float * __restrict__ up_row   = up   + i3 * up_s03   + i2 * up_s02   + i1 * up_s01;

    GGML_UNUSED(scale_dynamic);
    float inv_input_scale = 1.0f;
    if constexpr (has_scale) {
        const int64_t scale_idx = scale_activation_ne <= 1 ? 0 : i1;
        float input_scale = scale_activation[scale_idx];
        if (!(input_scale > 0.0f) || !isfinite(input_scale)) {
            input_scale = 1.0f;
        }
        inv_input_scale = 1.0f / input_scale;
    }

    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    const int64_t batch_offset = (int64_t) blockIdx.y * blocks_per_col * ne1;
    const int lane = threadIdx.x % WARP_SIZE;
    const int warp = threadIdx.x / WARP_SIZE;
    const int nwarps = blockDim.x / WARP_SIZE;
    const int64_t nfrags = (ne0 + 31) / 32;
    const int64_t frag_stride = (int64_t) gridDim.z * nwarps;
    for (int64_t frag = (int64_t) blockIdx.z * nwarps + warp; frag < nfrags; frag += frag_stride) {
        const int64_t i0 = 32 * frag + 2 * lane;
        const float2 values = lane < 16 ? make_float2(
            i0     < ne00 ? ggml_cuda_op_silu_single(gate_row[i0])     * up_row[i0]     * inv_input_scale : 0.0f,
            i0 + 1 < ne00 ? ggml_cuda_op_silu_single(gate_row[i0 + 1]) * up_row[i0 + 1] * inv_input_scale : 0.0f) :
            make_float2(0.0f, 0.0f);
        const float amax = warp_reduce_max<WARP_SIZE>(fmaxf(fabsf(values.x), fabsf(values.y)));
        uint32_t scale_code = lane == 0 ?
            (uint32_t) __nv_cvt_float_to_e8m0(amax / 448.0f, __NV_SATFINITE, cudaRoundPosInf) : 0;
        scale_code = __shfl_sync(0xFFFFFFFFu, scale_code, 0);
        const float inv_scale = scale_code > 0 ? ldexpf(1.0f, 127 - (int) scale_code) : 0.0f;

        const int64_t k_block = frag / 8;
        const int frag_idx = frag % 8;
        block_nvfp4_w4a8_mmq * __restrict__ yb = y + batch_offset + k_block * ne1 + i1;
        if (lane == 0) {
            reinterpret_cast<uint8_t *>(yb->sc8_u32)[frag_idx] = (uint8_t) scale_code;
        }
        if (lane < 16 && i0 < ne0) {
            const __nv_fp8x2_storage_t packed = __nv_cvt_float2_to_fp8x2(
                make_float2(values.x * inv_scale, values.y * inv_scale), __NV_SATFINITE, __NV_E4M3);
            uint8_t * __restrict__ dst = yb->qs + frag_idx * 32 + 2 * lane;
            if (i0 + 1 < ne0) {
                *reinterpret_cast<__nv_fp8x2_storage_t *>(dst) = packed;
            } else {
                dst[0] = static_cast<uint8_t>(packed);
            }
        }
    }
#else
    GGML_UNUSED_VARS(gate, up, y, scale_activation, scale_dynamic, scale_activation_ne, ne00,
                     gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <bool has_ids, bool has_scale, bool dynamic_scale>
static __global__ void quantize_mmq_nvfp4_w4a44(
        const float * __restrict__ x,
        const int32_t * __restrict__ ids,
        const int32_t * __restrict__ ids_expert,
        block_nvfp4_w4a44_mmq * __restrict__ y,
        const float * __restrict__ scale_activation,
        float * __restrict__ scale_dynamic,
        const int64_t scale_activation_ne,
        const int64_t ne00,
        const int64_t s01,
        const int64_t s02,
        const int64_t s03,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i1  = blockIdx.x;
    const int64_t i2  = blockIdx.y % ne2;
    const int64_t i3  = blockIdx.y / ne2;
    const int64_t i01 = has_ids ? ids[i1] : i1;
    const float * __restrict__ x_row = x + i3 * s03 + i2 * s02 + i01 * s01;

    __shared__ float row_scale_shared;
    if constexpr (dynamic_scale) {
        float amax = 0.0f;
        for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
            amax = fmaxf(amax, fabsf(x_row[i0]));
        }
        amax = warp_reduce_max<WARP_SIZE>(amax);

        __shared__ float warp_amax[2 * CUDA_QUANTIZE_BLOCK_SIZE / WARP_SIZE];
        const int lane = threadIdx.x % WARP_SIZE;
        const int warp = threadIdx.x / WARP_SIZE;
        if (lane == 0) {
            warp_amax[warp] = amax;
        }
        __syncthreads();
        if (warp == 0) {
            amax = lane < blockDim.x / WARP_SIZE ? warp_amax[lane] : 0.0f;
            amax = warp_reduce_max<WARP_SIZE>(amax);
            if (lane == 0) {
                row_scale_shared = amax / (6.0f * 448.0f);
                scale_dynamic[(int64_t) blockIdx.y * ne1 + i1] = row_scale_shared;
            }
        }
    } else if constexpr (has_scale) {
        if (threadIdx.x == 0) {
            const int64_t scale_idx = scale_activation_ne <= 1 ? 0 :
                (has_ids && ids_expert ? ids_expert[i1] : i01);
            float input_scale = scale_activation[scale_idx];
            if (!(input_scale > 0.0f) || !isfinite(input_scale)) {
                input_scale = 1.0f;
            }
            row_scale_shared = input_scale;
        }
    } else if (threadIdx.x == 0) {
        row_scale_shared = 1.0f;
    }
    __syncthreads();

    const float inv_scale = row_scale_shared > 0.0f ? 1.0f / row_scale_shared : 0.0f;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    const int64_t n_subblocks = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;
    const int64_t batch_offset = (int64_t) blockIdx.y * (blocks_per_col * ne1);
    for (int64_t isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
        const int64_t i0_base = isb * QK_NVFP4_SUB;
        const int64_t k_block = i0_base / QK_K;
        const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;
        block_nvfp4_w4a44_mmq * yb = y + batch_offset + k_block * ne1 + i1;

        float vals0[8];
        float vals1[8];
        ggml_cuda_load_nvfp4_values<false>(x_row, i0_base, ne00, vals0, vals1);
        float sub_max = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            vals0[k] *= inv_scale;
            vals1[k] *= inv_scale;
            sub_max = fmaxf(sub_max, fabsf(vals0[k]));
            sub_max = fmaxf(sub_max, fabsf(vals1[k]));
        }

        uint32_t plane0_0;
        uint32_t plane0_1;
        uint32_t plane1_0;
        uint32_t plane1_1;
        uint8_t fp8_code;
        ggml_cuda_nvfp4_quantize_4o6_residual(
            vals0, vals1, sub_max, plane0_0, plane0_1, plane1_0, plane1_1, fp8_code);
        yb->qs0_u32[2 * sub + 0] = plane0_0;
        yb->qs0_u32[2 * sub + 1] = plane0_1;
        yb->qs1_u32[2 * sub + 0] = plane1_0;
        yb->qs1_u32[2 * sub + 1] = plane1_1;
        reinterpret_cast<uint8_t *>(yb->sc4_u32)[sub] = fp8_code;
    }
#else
    GGML_UNUSED_VARS(x, ids, ids_expert, y, scale_activation, scale_dynamic, scale_activation_ne,
                     ne00, s01, s02, s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <bool has_scale, bool dynamic_scale>
static __global__ void quantize_mmq_nvfp4_w4a44_glu(
        const float * __restrict__ gate,
        const float * __restrict__ up,
        block_nvfp4_w4a44_mmq * __restrict__ y,
        const float * __restrict__ scale_activation,
        float * __restrict__ scale_dynamic,
        const int64_t scale_activation_ne,
        const int64_t ne00,
        const int64_t gate_s01,
        const int64_t gate_s02,
        const int64_t gate_s03,
        const int64_t up_s01,
        const int64_t up_s02,
        const int64_t up_s03,
        const int64_t ne0,
        const int64_t ne1,
        const int64_t ne2) {
#if defined(BLACKWELL_MMA_AVAILABLE)
    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.y % ne2;
    const int64_t i3 = blockIdx.y / ne2;
    const float * __restrict__ gate_row = gate + i3 * gate_s03 + i2 * gate_s02 + i1 * gate_s01;
    const float * __restrict__ up_row   = up   + i3 * up_s03   + i2 * up_s02   + i1 * up_s01;

    __shared__ float row_scale_shared;
    if constexpr (dynamic_scale) {
        float amax = 0.0f;
        for (int64_t i0 = threadIdx.x; i0 < ne00; i0 += blockDim.x) {
            const float value = ggml_cuda_op_silu_single(gate_row[i0]) * up_row[i0];
            amax = fmaxf(amax, fabsf(value));
        }
        amax = warp_reduce_max<WARP_SIZE>(amax);

        __shared__ float warp_amax[CUDA_QUANTIZE_BLOCK_SIZE / WARP_SIZE];
        const int lane = threadIdx.x % WARP_SIZE;
        const int warp = threadIdx.x / WARP_SIZE;
        if (lane == 0) {
            warp_amax[warp] = amax;
        }
        __syncthreads();
        if (warp == 0) {
            amax = lane < CUDA_QUANTIZE_BLOCK_SIZE / WARP_SIZE ? warp_amax[lane] : 0.0f;
            amax = warp_reduce_max<WARP_SIZE>(amax);
            if (lane == 0) {
                row_scale_shared = amax / (6.0f * 448.0f);
                scale_dynamic[(int64_t) blockIdx.y * ne1 + i1] = row_scale_shared;
            }
        }
    } else if constexpr (has_scale) {
        if (threadIdx.x == 0) {
            const int64_t scale_idx = scale_activation_ne <= 1 ? 0 : i1;
            float input_scale = scale_activation[scale_idx];
            if (!(input_scale > 0.0f) || !isfinite(input_scale)) {
                input_scale = 1.0f;
            }
            row_scale_shared = input_scale;
        }
    } else if (threadIdx.x == 0) {
        row_scale_shared = 1.0f;
    }
    __syncthreads();

    const float inv_scale = row_scale_shared > 0.0f ? 1.0f / row_scale_shared : 0.0f;
    const int64_t blocks_per_col = (ne0 + QK_K - 1) / QK_K;
    const int64_t n_subblocks = (ne0 + QK_NVFP4_SUB - 1) / QK_NVFP4_SUB;
    const int64_t batch_offset = (int64_t) blockIdx.y * (blocks_per_col * ne1);
    for (int64_t isb = threadIdx.x; isb < n_subblocks; isb += blockDim.x) {
        const int64_t i0_base = isb * QK_NVFP4_SUB;
        const int64_t k_block = i0_base / QK_K;
        const int sub = (i0_base % QK_K) / QK_NVFP4_SUB;
        block_nvfp4_w4a44_mmq * yb = y + batch_offset + k_block * ne1 + i1;

        float gate0[8];
        float gate1[8];
        float up0[8];
        float up1[8];
        ggml_cuda_load_nvfp4_values<false>(gate_row, i0_base, ne00, gate0, gate1);
        ggml_cuda_load_nvfp4_values<false>(up_row,   i0_base, ne00, up0,   up1);
        float vals0[8];
        float vals1[8];
        float sub_max = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k) {
            vals0[k] = ggml_cuda_op_silu_single(gate0[k]) * up0[k] * inv_scale;
            vals1[k] = ggml_cuda_op_silu_single(gate1[k]) * up1[k] * inv_scale;
            sub_max = fmaxf(sub_max, fabsf(vals0[k]));
            sub_max = fmaxf(sub_max, fabsf(vals1[k]));
        }

        uint32_t plane0_0;
        uint32_t plane0_1;
        uint32_t plane1_0;
        uint32_t plane1_1;
        uint8_t fp8_code;
        ggml_cuda_nvfp4_quantize_4o6_residual(
            vals0, vals1, sub_max, plane0_0, plane0_1, plane1_0, plane1_1, fp8_code);
        yb->qs0_u32[2 * sub + 0] = plane0_0;
        yb->qs0_u32[2 * sub + 1] = plane0_1;
        yb->qs1_u32[2 * sub + 0] = plane1_0;
        yb->qs1_u32[2 * sub + 1] = plane1_1;
        reinterpret_cast<uint8_t *>(yb->sc4_u32)[sub] = fp8_code;
    }
#else
    GGML_UNUSED_VARS(gate, up, y, scale_activation, scale_dynamic, scale_activation_ne, ne00,
                     gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
    NO_DEVICE_CODE;
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <bool scatter>
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2,
                                          const int     n_expert_used) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t block_fp4_mmq_size = QK_FP4_MMQ;
    const int64_t k_block            = warp_start_offset / block_fp4_mmq_size;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;

    ggml_cuda_pdl_sync();
    int64_t base_pos;
    if constexpr (scatter) {
        base_pos = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_pos = i3 * s03 + i2 * s02 + i01 * s01;
    }

    uint8_t scales[2];
    char2   packed[2];

#pragma unroll
    for (int b = 0; b < 2; ++b) {
        const int64_t i0 = warp_start_offset + b * vals_per_scale + lane_id_32;
        const float xi = (i0 < ne00) ? x[base_pos + i0] : 0.0f;

        float amax = fabsf(xi);
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) {
            amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, mask, WARP_SIZE));
        }

        const uint8_t e = compute_e8m0_scale(amax);
        scales[b] = e;
        const float inv_s = (amax == 0.0f) ? 0.0f : __frcp_rn(ggml_cuda_e8m0_to_fp32(e));

#if CUDART_VERSION >= 12080
        const float scaled_val = xi * inv_s;

        const float val0 = __shfl_sync(0xFFFFFFFF, scaled_val, base, WARP_SIZE);
        const float val1 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 16, WARP_SIZE);
        const float val2 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 1, WARP_SIZE);
        const float val3 = __shfl_sync(0xFFFFFFFF, scaled_val, base + 17, WARP_SIZE);

        __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));
        packed[b] = *(char2 *) &fp4_packed;
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        char2 q;
        q.x = (q_hi_0 << 4) | q_lo_0;
        q.y = (q_hi_1 << 4) | q_lo_1;
        packed[b] = q;
#endif // CUDART_VERSION >= 12080
    }

    block_fp4_mmq * y = (block_fp4_mmq *) vy;
    if constexpr (scatter) {
#pragma unroll
        for (int slot = 0; slot < n_expert_used; ++slot) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            block_fp4_mmq * yb = y + (k_block * ne1 + i);
            char2 * yqs2 = (char2 *) yb->qs;
            if (lane_in_group == 0) {
                yqs2[quad_idx_in_block * 16 + 0 * 8 + group_id] = packed[0];
                yqs2[quad_idx_in_block * 16 + 1 * 8 + group_id] = packed[1];
            }
            if (lane_id_32 == 0) {
                yb->d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
            }
        }
    } else {
        const int64_t ib0 = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
        block_fp4_mmq * yb = y + (ib0 + k_block * ne1 + blockIdx.x);
        char2 * yqs2 = (char2 *) yb->qs;
        if (lane_in_group == 0) {
            yqs2[quad_idx_in_block * 16 + 0 * 8 + group_id] = packed[0];
            yqs2[quad_idx_in_block * 16 + 1 * 8 + group_id] = packed[1];
        }
        if (lane_id_32 == 0) {
            yb->d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
        }
    }
    GGML_UNUSED(n_expert_used);
}

// scatter: grid over tokens, quantize once, write to all the token's compact rows
template <mmq_q8_1_ds_layout ds_layout, bool scatter>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2, const int n_expert_used) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();

    int64_t base_idx;
    if constexpr (scatter) {
        base_idx = (int64_t) blockIdx.x * s02; // one physical row per token
    } else {
        const int64_t i2  = blockIdx.z % ne2;
        const int64_t i3  = blockIdx.z / ne2;
        const int64_t i01 = ids ? ids[blockIdx.x] : blockIdx.x;
        base_idx = i3*s03 + i2*s02 + i01*s01;
    }

    const float4 * x4 = (const float4 *) x;
    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t k_block = i0 / QK8_1_MMQ; // column block in the channel
    const int64_t iqs     = i0 % QK8_1_MMQ; // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(base_idx + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    float amax = fabsf(xi.x);
    amax = fmaxf(amax, fabsf(xi.y));
    amax = fmaxf(amax, fabsf(xi.z));
    amax = fmaxf(amax, fabsf(xi.w));

    // Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
    for (int offset = vals_per_scale/8; offset > 0; offset >>= 1) {
        amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
    }

    float sum;
    if (ds_layout != MMQ_Q8_1_DS_LAYOUT_D4) {
        sum = xi.x + xi.y + xi.z + xi.w;

        // Calculate sums across vals_per_sum/4 threads.
#pragma unroll
        for (int offset = vals_per_sum/8; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
        }
    }

    const float d_inv = 127.0f / amax;
    char4 q;
    q.x = roundf(xi.x*d_inv);
    q.y = roundf(xi.y*d_inv);
    q.z = roundf(xi.z*d_inv);
    q.w = roundf(xi.w*d_inv);
    const float d = 1.0f / d_inv;

    // write the block once (normal) or to each of the token's compact rows (scatter)
    const int nwrite = scatter ? n_expert_used : 1;
#pragma unroll
    for (int slot = 0; slot < nwrite; ++slot) {
        int64_t ib;
        if constexpr (scatter) {
            const int64_t i = ids[(int64_t) blockIdx.x * n_expert_used + slot];
            ib = k_block*ne1 + i;
        } else {
            const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
            ib = ib0 + k_block*ne1 + blockIdx.x;
        }

        // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
        char4 * yqs4 = (char4 *) y[ib].qs;
        yqs4[iqs/4] = q;

        if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
            if (iqs % 16 == 0 && iqs < 96) {
                y[ib].d2s6[2 + iqs/16] = sum;
                if (iqs % 64 == 0) {
                    y[ib].d2s6[iqs/64] = d;
                }
            }
        } else if (iqs % 32 == 0) {
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
                y[ib].ds4[iqs/32] = make_half2(d, sum);
            } else {
                y[ib].d4[iqs/32]  = d;
            }
        }
    }
    GGML_UNUSED(n_expert_used);
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    quantize_row_q8_1_cuda(x, ids, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, nullptr, 0, stream);
}

void quantize_row_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    GGML_ASSERT(!ids);
    GGML_ASSERT(ne0 % QK8_1 == 0);
    GGML_ASSERT(scale_activation == nullptr || scale_activation_ne <= 1);

    const uint3 ne2_fastdiv = init_fastdiv_values(ne2);

    const int64_t block_num_x = (ne0 + CUDA_QUANTIZE_BLOCK_SIZE - 1) / CUDA_QUANTIZE_BLOCK_SIZE;
    const dim3 num_blocks(block_num_x, ne1, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params(num_blocks, block_size, 0, stream);
    if (scale_activation != nullptr) {
        ggml_cuda_kernel_launch(quantize_q8_1<true>, launch_params, x, vy, scale_activation, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    } else {
        ggml_cuda_kernel_launch(quantize_q8_1<false>, launch_params, x, vy, nullptr, ne00, s01, s02, s03, ne0, ne1, ne2_fastdiv);
    }
    GGML_UNUSED(type_src0);
}

void quantize_mmq_q8_1_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, false>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, false>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, false>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, /*n_expert_used=*/0);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_scatter_mmq_q8_1_cuda(
        const float * x, const int32_t * ids_src1_inv, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t stride_token, const int64_t ne0,
        const int64_t n_tokens, const int64_t nrows_dst, const int n_expert_used, cudaStream_t stream) {
    GGML_ASSERT(ne00 % 4 == 0);
    GGML_ASSERT(ne0 % QK8_1_MMQ == 0);

    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(n_tokens, block_num_y, 1);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, 0, stride_token, 0, ne0, (int) nrows_dst, 1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, 0, stride_token, 0, ne0, (int) nrows_dst, 1, n_expert_used);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids_src1_inv, vy, ne00, 0, stride_token, 0, ne0, (int) nrows_dst, 1, n_expert_used);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
}

void quantize_scatter_mmq_fp4_cuda(
        const float * x, const int32_t * ids_src1_inv, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t stride_token, const int64_t ne0,
        const int64_t n_tokens, const int64_t nrows_dst, const int n_expert_used, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_MXFP4);
    GGML_ASSERT(ne0 > 0);
    GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

    constexpr int nwarps = 8;
    constexpr int vals_per_block = nwarps * 2 * QK_MXFP4;
    const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
    const dim3 block_size(WARP_SIZE, nwarps, 1);
    const dim3 num_blocks(n_tokens, block_num_y, 1);
    quantize_mmq_mxfp4<true><<<num_blocks, block_size, 0, stream>>>(
        x, ids_src1_inv, vy, ne00, 0, stride_token, 0, ne0, (int) nrows_dst, 1, n_expert_used);
}

void quantize_mmq_nvfp4_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3, cudaStream_t stream) {
    quantize_mmq_nvfp4_cuda(x, ids, nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, nullptr, 0, stream);
}

void quantize_mmq_nvfp4_cuda(
        const float * x, const int32_t * ids, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    quantize_mmq_nvfp4_cuda(x, ids, nullptr, vy, type_src0, ne00, s01, s02, s03, ne0, ne1, ne2, ne3, scale_activation, scale_activation_ne, stream);
}

void quantize_mmq_nvfp4_cuda(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    quantize_mmq_nvfp4_cuda(x, ids, ids_expert, vy, type_src0, ne00, s01, s02, s03,
                            ne0, ne1, ne2, ne3, scale_activation, scale_activation_ne,
                            nullptr, false, stream);
}

void quantize_mmq_nvfp4_cuda(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy, const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne,
        float * scale_dynamic, const bool use_aligned_float8, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ne0 > 0);
    GGML_ASSERT(!scale_activation || !scale_dynamic);

    if (scale_dynamic) {
        const dim3 dynamic_blocks(ne1, ne2 * ne3, 1);
        const dim3 dynamic_threads(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
        if (ids) {
            if (use_aligned_float8) {
                quantize_mmq_nvfp4_dynamic<true, true><<<dynamic_blocks, dynamic_threads, 0, stream>>>(
                    x, ids, vy, scale_dynamic, ne00, s01, s02, s03, ne0, ne1, ne2);
            } else {
                quantize_mmq_nvfp4_dynamic<true, false><<<dynamic_blocks, dynamic_threads, 0, stream>>>(
                    x, ids, vy, scale_dynamic, ne00, s01, s02, s03, ne0, ne1, ne2);
            }
        } else if (use_aligned_float8) {
            quantize_mmq_nvfp4_dynamic<false, true><<<dynamic_blocks, dynamic_threads, 0, stream>>>(
                x, nullptr, vy, scale_dynamic, ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4_dynamic<false, false><<<dynamic_blocks, dynamic_threads, 0, stream>>>(
                x, nullptr, vy, scale_dynamic, ne00, s01, s02, s03, ne0, ne1, ne2);
        }
        return;
    }

    constexpr int nvfp4_block_size = 256;
    const int64_t block_num_y = (ne0 + QK_NVFP4_SUB * nvfp4_block_size - 1) / (QK_NVFP4_SUB * nvfp4_block_size);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    if (ids) {
        if (scale_activation) {
            if (use_aligned_float8) {
                quantize_mmq_nvfp4<true, true, true><<<num_blocks, block_size, 0, stream>>>(
                    x, ids, ids_expert, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
            } else {
                quantize_mmq_nvfp4<true, true, false><<<num_blocks, block_size, 0, stream>>>(
                    x, ids, ids_expert, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
            }
        } else if (use_aligned_float8) {
            quantize_mmq_nvfp4<true, false, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4<true, false, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (scale_activation) {
        if (use_aligned_float8) {
            quantize_mmq_nvfp4<false, true, true><<<num_blocks, block_size, 0, stream>>>(
                x, nullptr, nullptr, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4<false, true, false><<<num_blocks, block_size, 0, stream>>>(
                x, nullptr, nullptr, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (use_aligned_float8) {
        quantize_mmq_nvfp4<false, false, true><<<num_blocks, block_size, 0, stream>>>(
            x, nullptr, nullptr, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_nvfp4<false, false, false><<<num_blocks, block_size, 0, stream>>>(
            x, nullptr, nullptr, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_nvfp4_w4a8_cuda(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy,
        const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne,
        float * scale_dynamic, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne0 > 0);
    GGML_ASSERT(!scale_activation || !scale_dynamic);

    const dim3 threads(ne1 <= 8 ? 2 * CUDA_QUANTIZE_BLOCK_SIZE : CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    const int64_t nfrags = (ne0 + 31) / 32;
    const int64_t nwarps = threads.x / WARP_SIZE;
    const int64_t frag_blocks = ne1 <= 8 ? (nfrags + nwarps - 1) / nwarps : 1;
    const dim3 blocks(ne1, ne2 * ne3, frag_blocks);
    if (scale_dynamic) {
        if (ids) {
            quantize_mmq_nvfp4_w4a8<true, false, true><<<blocks, threads, 0, stream>>>(
                x, ids, ids_expert, (block_nvfp4_w4a8_mmq *) vy, nullptr, scale_dynamic, 0,
                ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4_w4a8<false, false, true><<<blocks, threads, 0, stream>>>(
                x, nullptr, nullptr, (block_nvfp4_w4a8_mmq *) vy, nullptr, scale_dynamic, 0,
                ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (scale_activation) {
        if (ids) {
            quantize_mmq_nvfp4_w4a8<true, true, false><<<blocks, threads, 0, stream>>>(
                x, ids, ids_expert, (block_nvfp4_w4a8_mmq *) vy, scale_activation, nullptr, scale_activation_ne,
                ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4_w4a8<false, true, false><<<blocks, threads, 0, stream>>>(
                x, nullptr, nullptr, (block_nvfp4_w4a8_mmq *) vy, scale_activation, nullptr, scale_activation_ne,
                ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (ids) {
        quantize_mmq_nvfp4_w4a8<true, false, false><<<blocks, threads, 0, stream>>>(
            x, ids, ids_expert, (block_nvfp4_w4a8_mmq *) vy, nullptr, nullptr, 0,
            ne00, s01, s02, s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_nvfp4_w4a8<false, false, false><<<blocks, threads, 0, stream>>>(
            x, nullptr, nullptr, (block_nvfp4_w4a8_mmq *) vy, nullptr, nullptr, 0,
            ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_nvfp4_w4a8_glu_cuda(
        const float * gate, const float * up, void * vy, const ggml_type type_src0,
        const int64_t ne00,
        const int64_t gate_s01, const int64_t gate_s02, const int64_t gate_s03,
        const int64_t up_s01, const int64_t up_s02, const int64_t up_s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne,
        float * scale_dynamic, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne0 > 0);
    GGML_ASSERT(!scale_activation || !scale_dynamic);

    const dim3 threads(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    const int64_t nfrags = (ne0 + 31) / 32;
    const int64_t nwarps = threads.x / WARP_SIZE;
    const int64_t frag_blocks = ne1 <= 8 ? (nfrags + nwarps - 1) / nwarps : 1;
    const dim3 blocks(ne1, ne2 * ne3, frag_blocks);
    if (scale_dynamic) {
        quantize_mmq_nvfp4_w4a8_glu<false, true><<<blocks, threads, 0, stream>>>(
            gate, up, (block_nvfp4_w4a8_mmq *) vy, nullptr, scale_dynamic, 0, ne00,
            gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
    } else if (scale_activation) {
        quantize_mmq_nvfp4_w4a8_glu<true, false><<<blocks, threads, 0, stream>>>(
            gate, up, (block_nvfp4_w4a8_mmq *) vy, scale_activation, nullptr, scale_activation_ne, ne00,
            gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_nvfp4_w4a8_glu<false, false><<<blocks, threads, 0, stream>>>(
            gate, up, (block_nvfp4_w4a8_mmq *) vy, nullptr, nullptr, 0, ne00,
            gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_nvfp4_w4a44_cuda(
        const float * x, const int32_t * ids, const int32_t * ids_expert, void * vy,
        const ggml_type type_src0,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne,
        float * scale_dynamic, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne0 > 0);
    GGML_ASSERT(!scale_activation || !scale_dynamic);

    const dim3 blocks(ne1, ne2 * ne3, 1);
    const dim3 threads(ne1 <= 8 ? 2 * CUDA_QUANTIZE_BLOCK_SIZE : CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    if (scale_dynamic) {
        if (ids) {
            quantize_mmq_nvfp4_w4a44<true, false, true><<<blocks, threads, 0, stream>>>(
                x, ids, ids_expert, (block_nvfp4_w4a44_mmq *) vy, nullptr, scale_dynamic, 0,
                ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4_w4a44<false, false, true><<<blocks, threads, 0, stream>>>(
                x, nullptr, nullptr, (block_nvfp4_w4a44_mmq *) vy, nullptr, scale_dynamic, 0,
                ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (scale_activation) {
        if (ids) {
            quantize_mmq_nvfp4_w4a44<true, true, false><<<blocks, threads, 0, stream>>>(
                x, ids, ids_expert, (block_nvfp4_w4a44_mmq *) vy, scale_activation, nullptr, scale_activation_ne,
                ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4_w4a44<false, true, false><<<blocks, threads, 0, stream>>>(
                x, nullptr, nullptr, (block_nvfp4_w4a44_mmq *) vy, scale_activation, nullptr, scale_activation_ne,
                ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else {
        quantize_mmq_nvfp4_w4a44<false, false, false><<<blocks, threads, 0, stream>>>(
            x, nullptr, nullptr, (block_nvfp4_w4a44_mmq *) vy, nullptr, nullptr, 0,
            ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_nvfp4_w4a44_glu_cuda(
        const float * gate, const float * up, void * vy, const ggml_type type_src0,
        const int64_t ne00,
        const int64_t gate_s01, const int64_t gate_s02, const int64_t gate_s03,
        const int64_t up_s01, const int64_t up_s02, const int64_t up_s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne,
        float * scale_dynamic, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne0 > 0);
    GGML_ASSERT(!scale_activation || !scale_dynamic);

    const dim3 blocks(ne1, ne2 * ne3, 1);
    const dim3 threads(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
    if (scale_dynamic) {
        quantize_mmq_nvfp4_w4a44_glu<false, true><<<blocks, threads, 0, stream>>>(
            gate, up, (block_nvfp4_w4a44_mmq *) vy, nullptr, scale_dynamic, 0, ne00,
            gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
    } else if (scale_activation) {
        quantize_mmq_nvfp4_w4a44_glu<true, false><<<blocks, threads, 0, stream>>>(
            gate, up, (block_nvfp4_w4a44_mmq *) vy, scale_activation, nullptr, scale_activation_ne, ne00,
            gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_nvfp4_w4a44_glu<false, false><<<blocks, threads, 0, stream>>>(
            gate, up, (block_nvfp4_w4a44_mmq *) vy, nullptr, nullptr, 0, ne00,
            gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_nvfp4_glu_cuda(
        const float * gate, const float * up, void * vy, const ggml_type type_src0,
        const int64_t ne00,
        const int64_t gate_s01, const int64_t gate_s02, const int64_t gate_s03,
        const int64_t up_s01, const int64_t up_s02, const int64_t up_s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    quantize_mmq_nvfp4_glu_cuda(gate, up, vy, type_src0, ne00,
                                gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03,
                                ne0, ne1, ne2, ne3, scale_activation, scale_activation_ne,
                                nullptr, false, stream);
}

void quantize_mmq_nvfp4_glu_cuda(
        const float * gate, const float * up, void * vy, const ggml_type type_src0,
        const int64_t ne00,
        const int64_t gate_s01, const int64_t gate_s02, const int64_t gate_s03,
        const int64_t up_s01, const int64_t up_s02, const int64_t up_s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne,
        float * scale_dynamic, const bool use_aligned_float8, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ne0 > 0);
    GGML_ASSERT(!scale_activation || !scale_dynamic);

    const bool dense_2d = ne2 == 1 && ne3 == 1;
    if (scale_dynamic) {
        const dim3 dynamic_blocks(ne1, ne2 * ne3, 1);
        const dim3 dynamic_threads(CUDA_QUANTIZE_BLOCK_SIZE, 1, 1);
        if (dense_2d) {
            if (use_aligned_float8) {
                quantize_mmq_nvfp4_glu_dynamic<true, true><<<dynamic_blocks, dynamic_threads, 0, stream>>>(
                    gate, up, vy, scale_dynamic, ne00, gate_s01, gate_s02, gate_s03,
                    up_s01, up_s02, up_s03, ne0, ne1, ne2);
            } else {
                quantize_mmq_nvfp4_glu_dynamic<true, false><<<dynamic_blocks, dynamic_threads, 0, stream>>>(
                    gate, up, vy, scale_dynamic, ne00, gate_s01, gate_s02, gate_s03,
                    up_s01, up_s02, up_s03, ne0, ne1, ne2);
            }
        } else if (use_aligned_float8) {
            quantize_mmq_nvfp4_glu_dynamic<false, true><<<dynamic_blocks, dynamic_threads, 0, stream>>>(
                gate, up, vy, scale_dynamic, ne00, gate_s01, gate_s02, gate_s03,
                up_s01, up_s02, up_s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4_glu_dynamic<false, false><<<dynamic_blocks, dynamic_threads, 0, stream>>>(
                gate, up, vy, scale_dynamic, ne00, gate_s01, gate_s02, gate_s03,
                up_s01, up_s02, up_s03, ne0, ne1, ne2);
        }
        return;
    }

    constexpr int nvfp4_block_size = 64;
    const int64_t block_num_y = (ne0 + QK_NVFP4_SUB * nvfp4_block_size - 1) / (QK_NVFP4_SUB * nvfp4_block_size);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    if (scale_activation) {
        if (dense_2d) {
            if (use_aligned_float8) {
                quantize_mmq_nvfp4_glu<true, true, true><<<num_blocks, block_size, 0, stream>>>(
                    gate, up, vy, scale_activation, scale_activation_ne, ne00,
                    gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
            } else {
                quantize_mmq_nvfp4_glu<true, true, false><<<num_blocks, block_size, 0, stream>>>(
                    gate, up, vy, scale_activation, scale_activation_ne, ne00,
                    gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
            }
        } else {
            if (use_aligned_float8) {
                quantize_mmq_nvfp4_glu<true, false, true><<<num_blocks, block_size, 0, stream>>>(
                    gate, up, vy, scale_activation, scale_activation_ne, ne00,
                    gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
            } else {
                quantize_mmq_nvfp4_glu<true, false, false><<<num_blocks, block_size, 0, stream>>>(
                    gate, up, vy, scale_activation, scale_activation_ne, ne00,
                    gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
            }
        }
    } else {
        if (dense_2d) {
            if (use_aligned_float8) {
                quantize_mmq_nvfp4_glu<false, true, true><<<num_blocks, block_size, 0, stream>>>(
                    gate, up, vy, nullptr, 0, ne00,
                    gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
            } else {
                quantize_mmq_nvfp4_glu<false, true, false><<<num_blocks, block_size, 0, stream>>>(
                    gate, up, vy, nullptr, 0, ne00,
                    gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
            }
        } else {
            if (use_aligned_float8) {
                quantize_mmq_nvfp4_glu<false, false, true><<<num_blocks, block_size, 0, stream>>>(
                    gate, up, vy, nullptr, 0, ne00,
                    gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
            } else {
                quantize_mmq_nvfp4_glu<false, false, false><<<num_blocks, block_size, 0, stream>>>(
                    gate, up, vy, nullptr, 0, ne00,
                    gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
            }
        }
    }
}

void quantize_mmq_mxfp4_cuda(const float *                    x,
                             const int32_t *                  ids,
                             void *                           vy,
                             [[maybe_unused]] const ggml_type type_src0,
                             const int64_t                    ne00,
                             const int64_t                    s01,
                             const int64_t                    s02,
                             const int64_t                    s03,
                             const int64_t                    ne0,
                             const int64_t                    ne1,
                             const int64_t                    ne2,
                             const int64_t                    ne3,
                             cudaStream_t                     stream) {
    GGML_ASSERT(ne0 % (2 * QK_MXFP4) == 0);

    constexpr int nwarps = 8;
    constexpr int vals_per_warp  = 2 * QK_MXFP4;
    constexpr int vals_per_block = nwarps * vals_per_warp;

    const int64_t block_num_y = (ne0 + vals_per_block - 1) / vals_per_block;
    const dim3    num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3    block_size(WARP_SIZE, nwarps, 1);

    quantize_mmq_mxfp4<false><<<num_blocks, block_size, 0, stream>>>(
        x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2, 0);
}
