#include "quantize.cuh"
#include "unary.cuh"
#include <cstdint>

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

template <bool has_ids, bool has_scale>
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

    float vals0[8];
    float vals1[8];
    float sub_max = 0.0f;
    const int64_t base_idx = i3 * s03 + i2 * s02 + i01 * s01;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int64_t i00 = i0_base + k;
        const float v = i00 < ne00 ? x[base_idx + i00] * inv_input_scale : 0.0f;
        vals0[k] = v;
        sub_max = fmaxf(sub_max, fabsf(v));
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int64_t i00 = i0_base + 8 + k;
        const float v = i00 < ne00 ? x[base_idx + i00] * inv_input_scale : 0.0f;
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
    NO_DEVICE_CODE; // This is for Blackwell NVFP4 activations only.
#endif // defined(BLACKWELL_MMA_AVAILABLE)
}

template <bool has_scale, bool dense_2d>
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

    float vals0[8];
    float vals1[8];
    float sub_max = 0.0f;

    ggml_cuda_pdl_sync();
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int64_t i00 = i0_base + k;
        const float v = i00 < ne00 ? ggml_cuda_op_silu_single(gate[gate_base_idx + i00]) * up[up_base_idx + i00] * inv_input_scale : 0.0f;
        vals0[k] = v;
        sub_max = fmaxf(sub_max, fabsf(v));
    }
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const int64_t i00 = i0_base + 8 + k;
        const float v = i00 < ne00 ? ggml_cuda_op_silu_single(gate[gate_base_idx + i00]) * up[up_base_idx + i00] * inv_input_scale : 0.0f;
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

// quantize values in the format mxfp4 is stored which is interleaved nibbles
// i.e. a block a0-a31 is represented as a0a16,a1a17 ...a15a31
static __global__ void quantize_mmq_mxfp4(const float * __restrict__ x,
                                          const int32_t * __restrict__ ids,
                                          void * __restrict__ vy,
                                          const int64_t ne00,
                                          const int64_t s01,
                                          const int64_t s02,
                                          const int64_t s03,
                                          const int64_t ne0,
                                          const int     ne1,
                                          const int     ne2) {
    constexpr int vals_per_scale = 32;
    constexpr int vals_per_warp  = 2 * vals_per_scale;  // Each warp processes 2 blocks of 32 = 64 values

    const int warp_id = threadIdx.y;
    const int lane_id_32 = threadIdx.x;

    const int nwarps = blockDim.y;

    const int64_t warp_start_offset = (blockIdx.y * nwarps + warp_id) * vals_per_warp;

    if (warp_start_offset >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    ggml_cuda_pdl_sync();
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    block_fp4_mmq * y = (block_fp4_mmq *) vy;

    const int64_t block_fp4_mmq_size = 8 * QK_MXFP4;  // 256 values
    const int64_t ib0                = blockIdx.z * ((int64_t) ne1 * (ne0 / block_fp4_mmq_size));
    const int64_t ib = ib0 + (warp_start_offset / block_fp4_mmq_size) * ne1 + blockIdx.x;
    const int64_t quad_idx_in_block  = (warp_start_offset % block_fp4_mmq_size) / vals_per_warp;

    const int group_id = lane_id_32 / 4;
    const int lane_in_group = lane_id_32 % 4;
    const int base = group_id * 2;
    char2 * yqs2 = (char2 *) y[ib].qs;

    const int64_t base_pos = i03 * s03 + i02 * s02 + i01 * s01;

    uint8_t scales[2];

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

        if (lane_in_group == 0) {
            __nv_fp4x4_e2m1 fp4_packed(make_float4(val0, val1, val2, val3));

            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = *(char2 *) &fp4_packed;
        }
#else
        // Fallback: manual FP4 conversion using LUT
        const uint8_t q_val = ggml_cuda_float_to_fp4_e2m1(xi, inv_s);

        const uint8_t q_lo_0 = __shfl_sync(0xFFFFFFFF, q_val, base,      WARP_SIZE);
        const uint8_t q_lo_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 1,  WARP_SIZE);
        const uint8_t q_hi_0 = __shfl_sync(0xFFFFFFFF, q_val, base + 16, WARP_SIZE);
        const uint8_t q_hi_1 = __shfl_sync(0xFFFFFFFF, q_val, base + 17, WARP_SIZE);

        if (lane_in_group == 0) {
            char2 q;
            q.x = (q_hi_0 << 4) | q_lo_0;
            q.y = (q_hi_1 << 4) | q_lo_1;
            yqs2[quad_idx_in_block * 16 + b * 8 + group_id] = q;
        }
#endif // CUDART_VERSION >= 12080
    }

    if (lane_id_32 == 0) {
        // Store 2 scales packed into 1 uint32
        y[ib].d4[quad_idx_in_block] = (scales[1] << 8) | scales[0];
    }
}

template <mmq_q8_1_ds_layout ds_layout>
static __global__ void quantize_mmq_q8_1(
        const float * __restrict__ x, const int32_t * __restrict__ ids, void * __restrict__ vy,
        const int64_t ne00, const int64_t s01, const int64_t s02, const int64_t s03,
        const int64_t ne0, const int ne1, const int ne2) {

    constexpr int vals_per_scale = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 64 : 32;
    constexpr int vals_per_sum   = ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6 ? 16 : 32;

    const int64_t i0 = ((int64_t)blockDim.x*blockIdx.y + threadIdx.x)*4;

    if (i0 >= ne0) {
        return;
    }

    const int64_t i1 = blockIdx.x;
    const int64_t i2 = blockIdx.z % ne2;
    const int64_t i3 = blockIdx.z / ne2;

    const int64_t i00 = i0;
    ggml_cuda_pdl_sync();
    const int64_t i01 = ids ? ids[i1] : i1;
    const int64_t i02 = i2;
    const int64_t i03 = i3;

    const float4 * x4 = (const float4 *) x;

    block_q8_1_mmq * y = (block_q8_1_mmq *) vy;

    const int64_t ib0 = blockIdx.z*((int64_t)gridDim.x*gridDim.y*blockDim.x/QK8_1); // first block of channel
    const int64_t ib  = ib0 + (i0 / (4*QK8_1))*ne1 + blockIdx.x;                    // block index in channel
    const int64_t iqs = i0 % (4*QK8_1);                                             // quant index in block

    // Load 4 floats per thread and calculate max. abs. value between them:
    const float4 xi = i0 < ne00 ? x4[(i03*s03 + i02*s02 + i01*s01 + i00)/4] : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
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

    // Write back 4 int8 values as a single 32 bit value for better memory bandwidth:
    char4 * yqs4 = (char4 *) y[ib].qs;
    yqs4[iqs/4] = q;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D2S6) {
        if (iqs % 16 != 0 || iqs >= 96) {
            return;
        }

        y[ib].d2s6[2 + iqs/16] = sum;

        if (iqs % 64 != 0) {
            return;
        }

        const float d = 1.0f / d_inv;

        y[ib].d2s6[iqs/64] = d;

        return;
    }

    if (iqs % 32 != 0) {
        return;
    }

    const float d = 1.0f / d_inv;

    if (ds_layout == MMQ_Q8_1_DS_LAYOUT_DS4) {
        y[ib].ds4[iqs/32] = make_half2(d, sum);
    } else {
        y[ib].d4[iqs/32]  = d;
    }
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
    GGML_ASSERT(ne0 % (4*QK8_1) == 0);

    // ne1 tends to assume the highest values, therefore use it as the "x" dimension of the CUDA grid:
    const int64_t block_num_y = (ne0 + 4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ - 1) / (4*CUDA_QUANTIZE_BLOCK_SIZE_MMQ);
    const dim3 num_blocks(ne1, block_num_y, ne2*ne3);
    const dim3 block_size(CUDA_QUANTIZE_BLOCK_SIZE_MMQ, 1, 1);
    switch (mmq_get_q8_1_ds_layout(type_src0)) {
        case MMQ_Q8_1_DS_LAYOUT_D4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_DS4:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_DS4>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        case MMQ_Q8_1_DS_LAYOUT_D2S6:
            quantize_mmq_q8_1<MMQ_Q8_1_DS_LAYOUT_D2S6>
                <<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
            break;
        default:
            GGML_ABORT("fatal error");
            break;
    }
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
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ne0 > 0);

    constexpr int nvfp4_block_size = 256;
    const int64_t block_num_y = (ne0 + QK_NVFP4_SUB * nvfp4_block_size - 1) / (QK_NVFP4_SUB * nvfp4_block_size);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    if (ids) {
        if (scale_activation) {
            quantize_mmq_nvfp4<true, true><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4<true, false><<<num_blocks, block_size, 0, stream>>>(
                x, ids, ids_expert, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
        }
    } else if (scale_activation) {
        quantize_mmq_nvfp4<false, true><<<num_blocks, block_size, 0, stream>>>(
            x, ids, nullptr, vy, scale_activation, scale_activation_ne, ne00, s01, s02, s03, ne0, ne1, ne2);
    } else {
        quantize_mmq_nvfp4<false, false><<<num_blocks, block_size, 0, stream>>>(
            x, ids, nullptr, vy, nullptr, 0, ne00, s01, s02, s03, ne0, ne1, ne2);
    }
}

void quantize_mmq_nvfp4_glu_cuda(
        const float * gate, const float * up, void * vy, const ggml_type type_src0,
        const int64_t ne00,
        const int64_t gate_s01, const int64_t gate_s02, const int64_t gate_s03,
        const int64_t up_s01, const int64_t up_s02, const int64_t up_s03,
        const int64_t ne0, const int64_t ne1, const int64_t ne2, const int64_t ne3,
        const float * scale_activation, const int64_t scale_activation_ne, cudaStream_t stream) {
    GGML_ASSERT(type_src0 == GGML_TYPE_NVFP4);
    GGML_ASSERT(ne00 % 8 == 0);
    GGML_ASSERT(ne0 > 0);

    constexpr int nvfp4_block_size = 64;
    const int64_t block_num_y = (ne0 + QK_NVFP4_SUB * nvfp4_block_size - 1) / (QK_NVFP4_SUB * nvfp4_block_size);
    const dim3 num_blocks(ne1, block_num_y, ne2 * ne3);
    const dim3 block_size(nvfp4_block_size, 1, 1);
    const bool dense_2d = ne2 == 1 && ne3 == 1;
    if (scale_activation) {
        if (dense_2d) {
            quantize_mmq_nvfp4_glu<true, true><<<num_blocks, block_size, 0, stream>>>(
                gate, up, vy, scale_activation, scale_activation_ne, ne00,
                gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4_glu<true, false><<<num_blocks, block_size, 0, stream>>>(
                gate, up, vy, scale_activation, scale_activation_ne, ne00,
                gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
        }
    } else {
        if (dense_2d) {
            quantize_mmq_nvfp4_glu<false, true><<<num_blocks, block_size, 0, stream>>>(
                gate, up, vy, nullptr, 0, ne00,
                gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
        } else {
            quantize_mmq_nvfp4_glu<false, false><<<num_blocks, block_size, 0, stream>>>(
                gate, up, vy, nullptr, 0, ne00,
                gate_s01, gate_s02, gate_s03, up_s01, up_s02, up_s03, ne0, ne1, ne2);
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

    quantize_mmq_mxfp4<<<num_blocks, block_size, 0, stream>>>(x, ids, vy, ne00, s01, s02, s03, ne0, ne1, ne2);
}
