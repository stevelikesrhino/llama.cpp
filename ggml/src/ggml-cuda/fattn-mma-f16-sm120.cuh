#pragma once

#if !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && defined(CUDART_VERSION) && CUDART_VERSION >= 12080

#include <cuda.h>

namespace ggml_cuda_fattn_sm120 {

template<int DKQ> struct config;

template<> struct config<128> {
    static constexpr int nbatch_fa      = 64;
    static constexpr int nbatch_combine = 64;
};

template<> struct config<256> {
    static constexpr int nbatch_fa      = 64;
    static constexpr int nbatch_combine = 128;
};

template<> struct config<512> {
    static constexpr int nbatch_fa      = 64;
    static constexpr int nbatch_combine = 128;
};

static constexpr int nwarps       = 4;
static constexpr int depth        = 2;
static constexpr int chunk_h2     = 32;
static constexpr int chunk_ne     = 2*chunk_h2;
static constexpr int barrier_id   = 7;

struct alignas(8) circular_barriers {
    uint64_t produced[depth];
    uint64_t consumed[depth];
};

struct alignas(1024) barrier_storage {
    uint64_t q_ready;
    circular_barriers kv;
    circular_barriers mask;
};

static __device__ __forceinline__ uint32_t smem_addr(const void * ptr) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

static __device__ __forceinline__ void barrier_init(uint64_t * barrier, const uint32_t arrive_count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(smem_addr(barrier)), "r"(arrive_count) : "memory");
}

static __device__ __forceinline__ void barrier_arrive_expect_tx(uint64_t * barrier, const uint32_t bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(smem_addr(barrier)), "r"(bytes) : "memory");
}

static __device__ __forceinline__ void barrier_arrive(uint64_t * barrier) {
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(smem_addr(barrier)) : "memory");
}

static __device__ __forceinline__ void barrier_wait(uint64_t * barrier, const uint32_t phase) {
    asm volatile(
        "{\n\t"
        ".reg .pred p;\n\t"
        "L_fattn_tma_wait_%=: \n\t"
        "mbarrier.try_wait.parity.shared::cta.b64 p, [%0], %1, %2;\n\t"
        "@p bra L_fattn_tma_done_%=;\n\t"
        "bra L_fattn_tma_wait_%=;\n\t"
        "L_fattn_tma_done_%=: \n\t"
        "}"
        :: "r"(smem_addr(barrier)), "r"(phase), "r"(0x989680) : "memory");
}

static __device__ __forceinline__ void tma_load_4d(
        void * dst, const CUtensorMap * map, uint64_t * barrier,
        const int32_t c0, const int32_t c1, const int32_t c2, const int32_t c3) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL
    asm volatile(
        "cp.async.bulk.tensor.4d.shared::cta.global.tile.mbarrier::complete_tx::bytes "
        "[%0], [%1, {%2, %3, %4, %5}], [%6];"
        :: "r"(smem_addr(dst)), "l"(reinterpret_cast<uint64_t>(map)),
           "r"(c0), "r"(c1), "r"(c2), "r"(c3), "r"(smem_addr(barrier)) : "memory");
#else
    GGML_UNUSED_VARS(dst, map, barrier, c0, c1, c2, c3);
#endif
}

static __device__ __forceinline__ void consumer_sync() {
    asm volatile("bar.sync %0, %1;" :: "n"(barrier_id), "n"(nwarps*WARP_SIZE) : "memory");
}

template<data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_swizzle_128(
        tile<16, 8, half2, dl> & t, const half2 * base, const int row0, const int col0, const int stride) {
    int * xi = reinterpret_cast<int *>(t.x);
    const int row = row0 + threadIdx.x % 16;
    const int col = col0 + (threadIdx.x / 16)*4;
    const int col_swizzled = col ^ ((row & 7)*4);
    const int * xs = reinterpret_cast<const int *>(base) + row*stride + col_swizzled;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[1]), "=r"(xi[2]), "=r"(xi[3]) : "l"(xs));
}

template<data_layout dl>
static __device__ __forceinline__ void load_ldmatrix_trans_swizzle_128(
        tile<16, 8, half2, dl> & t, const half2 * base, const int row0, const int col0, const int stride) {
    int * xi = reinterpret_cast<int *>(t.x);
    const int row = row0 + threadIdx.x % 16;
    const int col = col0 + (threadIdx.x / 16)*4;
    const int col_swizzled = col ^ ((row & 7)*4);
    const int * xs = reinterpret_cast<const int *>(base) + row*stride + col_swizzled;
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.b16 {%0, %1, %2, %3}, [%4];"
        : "=r"(xi[0]), "=r"(xi[2]), "=r"(xi[1]), "=r"(xi[3]) : "l"(xs));
}

struct writer {
    circular_barriers * barriers;
    uint32_t ptr   = 0;
    uint32_t phase = 0xffffffffu;

    __device__ __forceinline__ int reserve(const bool elected) {
        if (elected) {
            barrier_wait(&barriers->consumed[ptr], (phase >> ptr) & 1u);
        }
        __syncwarp();
        return ptr;
    }

    __device__ __forceinline__ void advance() {
        phase ^= 1u << ptr;
        ptr = ptr + 1 == depth ? 0 : ptr + 1;
    }
};

struct reader {
    circular_barriers * barriers;
    uint32_t ptr   = 0;
    uint32_t phase = 0;

    __device__ __forceinline__ int wait() {
        barrier_wait(&barriers->produced[ptr], (phase >> ptr) & 1u);
        return ptr;
    }

    __device__ __forceinline__ void pop(const int slot) {
        barrier_arrive(&barriers->consumed[slot]);
        phase ^= 1u << ptr;
        ptr = ptr + 1 == depth ? 0 : ptr + 1;
    }
};

template<int DKQ, int ncols1>
struct shared_layout {
    static constexpr int ncols            = 64;
    static constexpr int nbatch_fa        = config<DKQ>::nbatch_fa;
    static constexpr int q_h2             = ncols*(DKQ/2);
    static constexpr int kv_slot_h2       = nbatch_fa*chunk_h2;
    static constexpr int mask_slot_h      = ncols1*nbatch_fa;
    static constexpr int stream_bytes     = depth*kv_slot_h2*sizeof(half2) + depth*mask_slot_h*sizeof(half);
    static constexpr int pipeline_bytes   = q_h2*sizeof(half2) + stream_bytes;
    static constexpr int combine_h2       = nwarps*16*(config<DKQ>::nbatch_combine + 4);
    static constexpr int data_bytes       = pipeline_bytes > combine_h2*int(sizeof(half2)) ? pipeline_bytes : combine_h2*int(sizeof(half2));
    static constexpr int shared_bytes     = sizeof(barrier_storage) + data_bytes;

    barrier_storage * barriers;
    half2 * data;

    __device__ explicit shared_layout(void * smem) {
        barriers = reinterpret_cast<barrier_storage *>(smem);
        data = reinterpret_cast<half2 *>(reinterpret_cast<uint8_t *>(smem) + sizeof(barrier_storage));
    }

    __device__ __forceinline__ half2 * q() const {
        return data;
    }

    __device__ __forceinline__ half2 * kv(const int slot) const {
        return data + q_h2 + slot*kv_slot_h2;
    }

    __device__ __forceinline__ half * mask(const int slot) const {
        return reinterpret_cast<half *>(data + q_h2 + depth*kv_slot_h2) + slot*mask_slot_h;
    }
};

template<int DKQ, int ncols1, int ncols2>
static __device__ __forceinline__ void producer(
        const float2 * Q_f2, const float scale,
        const int stride_Q1, const int stride_Q2,
        const int jt, const int zt_gqa, const int gqa_ratio, const uint3 ne01,
        const int sequence, const int ne33, const int z_KV, const int kb0_start, const int kb0_stop,
        const CUtensorMap * map_k, const CUtensorMap * map_v, const CUtensorMap * map_mask,
        shared_layout<DKQ, ncols1> layout) {
    constexpr int nbatch_fa = config<DKQ>::nbatch_fa;
    constexpr int nchunks   = DKQ/chunk_ne;
    const int lane = threadIdx.x;
    const bool elected = lane == 0;

    writer kvw{&layout.barriers->kv};
    writer mw{&layout.barriers->mask};
    const half2 scale_h2 = make_half2(scale, scale);
    half2 * tile_q = layout.q();

    for (int i = lane; i < 64*(DKQ/2); i += WARP_SIZE) {
        const int jc = i/(DKQ/2);
        const int k = i - jc*(DKQ/2);
        const int j = jc/ncols2;
        const int c = jc - j*ncols2;
        half2 value = make_half2(0.0f, 0.0f);
        if (jt*ncols1 + j < int(ne01.z) && zt_gqa*ncols2 + c < gqa_ratio) {
            const float2 q = Q_f2[(jt*ncols1 + j)*stride_Q1 + c*stride_Q2 + k];
            value = scale_h2*make_half2(q.x, q.y);
        }
        const int k_smem = DKQ > 256 ? k ^ ((jc & 7)*4) : k;
        tile_q[jc*(DKQ/2) + k_smem] = value;
    }
    __syncwarp();
    if (elected) {
        __threadfence_block();
        barrier_arrive(&layout.barriers->q_ready);
    }
    for (int kb0 = kb0_start; kb0 < kb0_stop; ++kb0) {
#pragma unroll
        for (int chunk = 0; chunk < nchunks; ++chunk) {
            const int slot = kvw.reserve(elected);
            if (elected) {
                barrier_arrive_expect_tx(&layout.barriers->kv.produced[slot], nbatch_fa*chunk_ne*sizeof(half));
                tma_load_4d(layout.kv(slot), map_k, &layout.barriers->kv.produced[slot],
                    chunk*chunk_ne, kb0*nbatch_fa, z_KV, sequence);
            }
            kvw.advance();
        }

        {
            const int slot = mw.reserve(elected);
            if (elected) {
                barrier_arrive_expect_tx(&layout.barriers->mask.produced[slot], ncols1*nbatch_fa*sizeof(half));
                tma_load_4d(layout.mask(slot), map_mask, &layout.barriers->mask.produced[slot],
                    kb0*nbatch_fa, jt*ncols1, 0, sequence % ne33);
            }
            mw.advance();
        }

#pragma unroll
        for (int chunk = 0; chunk < nchunks; ++chunk) {
            const int slot = kvw.reserve(elected);
            if (elected) {
                barrier_arrive_expect_tx(&layout.barriers->kv.produced[slot], nbatch_fa*chunk_ne*sizeof(half));
                tma_load_4d(layout.kv(slot), map_v, &layout.barriers->kv.produced[slot],
                    chunk*chunk_ne, kb0*nbatch_fa, z_KV, sequence);
            }
            kvw.advance();
        }
    }
}

template<int DKQ, int ncols1, int ncols2, bool use_logit_softcap, int split_k>
static __device__ __forceinline__ void consumer(
        const float * sinks_f,
        float2 * dstk,
        half2 * dst_parts,
        float2 * dst_meta,
        const float slope,
        const float logit_softcap,
        const uint3 ne01,
        const int ne02,
        const int gqa_ratio,
        const int jt,
        const int zt_gqa,
        const int kb0_stop,
        shared_layout<DKQ, ncols1> layout) {
#if defined(TURING_MMA_AVAILABLE)
    constexpr int DV             = DKQ;
    constexpr int ncols          = ncols1*ncols2;
    constexpr int nbatch_fa      = config<DKQ>::nbatch_fa;
    constexpr int nbatch_combine = config<DKQ>::nbatch_combine;
    using T_A_KQ  = typename mma_tile_sizes<DV, ncols>::T_A_KQ;
    using T_B_KQ  = typename mma_tile_sizes<DV, ncols>::T_B_KQ;
    using T_C_KQ  = typename mma_tile_sizes<DV, ncols>::T_C_KQ;
    using T_A_VKQ = typename mma_tile_sizes<DV, ncols>::T_A_VKQ;
    using T_B_VKQ = typename mma_tile_sizes<DV, ncols>::T_B_VKQ;
    using T_C_VKQ = typename mma_tile_sizes<DV, ncols>::T_C_VKQ;
    constexpr int cols_per_warp   = T_B_KQ::I;
    constexpr int cols_per_thread = get_cols_per_thread();
    constexpr int np              = nwarps*cols_per_warp/ncols;
    constexpr bool q_in_reg       = DKQ <= 256;
    static_assert(ncols == 64 && cols_per_warp == 16 && np == 1, "bad SM120 attention shape");

    T_B_KQ Q_B[q_in_reg ? DKQ/(2*T_B_KQ::J) : 1];
    T_C_VKQ VKQ_C[DV/(2*T_C_VKQ::J)];
    float KQ_rowsum[cols_per_thread] = {0.0f};
    float KQ_max[cols_per_thread];
#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_max[col] = -FLT_MAX/2.0f;
    }

    barrier_wait(&layout.barriers->q_ready, 0);
    half2 * tile_q = layout.q();
    const int j0 = threadIdx.y*cols_per_warp;
    if constexpr (q_in_reg) {
#pragma unroll
        for (int k0 = 0; k0 < DKQ/2; k0 += T_B_KQ::J) {
            load_ldmatrix(Q_B[k0/T_B_KQ::J], tile_q + j0*(DKQ/2) + k0, DKQ/2);
        }
    }

    reader kvr{&layout.barriers->kv};
    reader mr{&layout.barriers->mask};

    for (int kb0 = 0; kb0 < kb0_stop; ++kb0) {
        T_C_KQ KQ_C[nbatch_fa/T_C_KQ::J];

#pragma unroll
        for (int chunk = 0; chunk < DKQ/chunk_ne; ++chunk) {
            const int slot = kvr.wait();
            half2 * tile_k = layout.kv(slot);
            T_B_KQ Q_fragment;

#pragma unroll
            for (int i00 = 0; i00 < nbatch_fa; i00 += T_A_KQ::I) {
#pragma unroll
                for (int k0 = 0; k0 < chunk_h2; k0 += T_A_KQ::J) {
                    if constexpr (q_in_reg) {
                        Q_fragment = Q_B[(chunk*chunk_h2 + k0)/T_B_KQ::J];
                    } else {
                        load_ldmatrix_swizzle_128(Q_fragment, tile_q, j0, chunk*chunk_h2 + k0, DKQ/2);
                    }
                    T_A_KQ K_A;
                    load_ldmatrix_swizzle_128(K_A, tile_k, i00, k0, chunk_h2);
                    mma(KQ_C[i00/T_A_KQ::I], Q_fragment, K_A);
                }
            }
            kvr.pop(slot);
        }

        if constexpr (use_logit_softcap) {
#pragma unroll
            for (int i = 0; i < nbatch_fa/T_C_KQ::J; ++i) {
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    KQ_C[i].x[l] = logit_softcap*tanhf(KQ_C[i].x[l]);
                }
            }
        }

        const int mask_slot = mr.wait();
        const half * tile_mask = layout.mask(mask_slot);
        float KQ_max_new[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            KQ_max_new[col] = KQ_max[col];
        }
        float KQ_rowsum_add[cols_per_thread] = {0.0f};

#pragma unroll
        for (int i00 = 0; i00 < nbatch_fa; i00 += T_C_KQ::J) {
#pragma unroll
            for (int l0 = 0; l0 < T_C_KQ::ne; l0 += 2) {
                const int i = (i00 + T_C_KQ::get_j(l0))/2;
                const int j = (threadIdx.y*cols_per_warp + T_C_KQ::get_i(l0))/ncols2;
                const float2 tmp = __half22float2(reinterpret_cast<const half2 *>(tile_mask)[j*(nbatch_fa/2) + i]);
                KQ_C[i00/T_C_KQ::J].x[l0 + 0] += slope*tmp.x;
                KQ_C[i00/T_C_KQ::J].x[l0 + 1] += slope*tmp.y;
            }
        }
        mr.pop(mask_slot);

        if constexpr (DKQ == 256) {
            constexpr int nbatch_phase = 32;
            static_assert(nbatch_fa == 2*nbatch_phase, "bad D256 logical attention tile");
            float KQ_max_scale_phase_1[cols_per_thread];

#pragma unroll
            for (int phase = 0; phase < 2; ++phase) {
#pragma unroll
                for (int col = 0; col < cols_per_thread; ++col) {
                    KQ_max_new[col] = KQ_max[col];
                    KQ_rowsum_add[col] = 0.0f;
                }
#pragma unroll
                for (int k0 = phase*nbatch_phase; k0 < (phase + 1)*nbatch_phase; k0 += T_C_KQ::J) {
#pragma unroll
                    for (int l = 0; l < T_C_KQ::ne; ++l) {
                        const int KQ_idx = (l/2) % 2;
                        KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/T_C_KQ::J].x[l] + FATTN_KQ_MAX_OFFSET);
                    }
                }
#pragma unroll
                for (int col = 0; col < cols_per_thread; ++col) {
                    KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xffffffffu, KQ_max_new[col], 2));
                    KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xffffffffu, KQ_max_new[col], 1));
                }
#pragma unroll
                for (int k0 = phase*nbatch_phase; k0 < (phase + 1)*nbatch_phase; k0 += T_C_KQ::J) {
#pragma unroll
                    for (int l = 0; l < T_C_KQ::ne; ++l) {
                        const int KQ_idx = (l/2) % 2;
                        KQ_C[k0/T_C_KQ::J].x[l] = expf(KQ_C[k0/T_C_KQ::J].x[l] - KQ_max_new[KQ_idx]);
                        KQ_rowsum_add[KQ_idx] += KQ_C[k0/T_C_KQ::J].x[l];
                    }
                }

                float KQ_max_scale[cols_per_thread];
#pragma unroll
                for (int col = 0; col < cols_per_thread; ++col) {
                    const float diff = KQ_max[col] - KQ_max_new[col];
                    KQ_max_scale[col] = expf(diff);
                    KQ_max[col] = KQ_max_new[col];
                    *reinterpret_cast<uint32_t *>(&KQ_max_scale[col]) *= diff >= SOFTMAX_FTZ_THRESHOLD;
                    KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
                }
                if (phase == 0) {
#pragma unroll
                    for (int col = 0; col < cols_per_thread; ++col) {
                        const half2 scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                        for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                            for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                                VKQ_C[i].x[l0 + col] *= scale_h2;
                            }
                        }
                    }
                } else {
#pragma unroll
                    for (int col = 0; col < cols_per_thread; ++col) {
                        KQ_max_scale_phase_1[col] = KQ_max_scale[col];
                    }
                }
            }

            T_B_VKQ B[nbatch_fa/(2*T_B_VKQ::J)];
#pragma unroll
            for (int k = 0; k < nbatch_fa/(2*T_B_VKQ::J); ++k) {
                B[k] = get_half2(KQ_C[k]);
            }

#pragma unroll
            for (int chunk = 0; chunk < DV/chunk_ne; ++chunk) {
                const int slot = kvr.wait();
                half2 * tile_v = layout.kv(slot);
                const int i0_start = chunk*chunk_ne;
#pragma unroll
                for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_start + chunk_ne; i_VKQ_0 += T_A_VKQ::I) {
#pragma unroll
                    for (int k00 = 0; k00 < nbatch_phase/2; k00 += T_A_VKQ::J) {
                        T_A_VKQ A;
                        load_ldmatrix_trans_swizzle_128(A, tile_v, 2*k00, (i_VKQ_0 - i0_start)/2, chunk_h2);
                        mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/T_A_VKQ::J], A);
                    }
#pragma unroll
                    for (int col = 0; col < cols_per_thread; ++col) {
                        const half2 scale_h2 = make_half2(KQ_max_scale_phase_1[col], KQ_max_scale_phase_1[col]);
#pragma unroll
                        for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                            VKQ_C[i_VKQ_0/T_A_VKQ::I].x[l0 + col] *= scale_h2;
                        }
                    }
#pragma unroll
                    for (int k00 = nbatch_phase/2; k00 < nbatch_fa/2; k00 += T_A_VKQ::J) {
                        T_A_VKQ A;
                        load_ldmatrix_trans_swizzle_128(A, tile_v, 2*k00, (i_VKQ_0 - i0_start)/2, chunk_h2);
                        mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/T_A_VKQ::J], A);
                    }
                }
                kvr.pop(slot);
            }
        } else {
#pragma unroll
            for (int k0 = 0; k0 < nbatch_fa; k0 += T_C_KQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int KQ_idx = (l/2) % 2;
                    KQ_max_new[KQ_idx] = fmaxf(KQ_max_new[KQ_idx], KQ_C[k0/T_C_KQ::J].x[l] + FATTN_KQ_MAX_OFFSET);
                }
            }
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xffffffffu, KQ_max_new[col], 2));
                KQ_max_new[col] = fmaxf(KQ_max_new[col], __shfl_xor_sync(0xffffffffu, KQ_max_new[col], 1));
            }
#pragma unroll
            for (int k0 = 0; k0 < nbatch_fa; k0 += T_C_KQ::J) {
#pragma unroll
                for (int l = 0; l < T_C_KQ::ne; ++l) {
                    const int KQ_idx = (l/2) % 2;
                    KQ_C[k0/T_C_KQ::J].x[l] = expf(KQ_C[k0/T_C_KQ::J].x[l] - KQ_max_new[KQ_idx]);
                    KQ_rowsum_add[KQ_idx] += KQ_C[k0/T_C_KQ::J].x[l];
                }
            }

            float KQ_max_scale[cols_per_thread];
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const float diff = KQ_max[col] - KQ_max_new[col];
                KQ_max_scale[col] = expf(diff);
                KQ_max[col] = KQ_max_new[col];
                *reinterpret_cast<uint32_t *>(&KQ_max_scale[col]) *= diff >= SOFTMAX_FTZ_THRESHOLD;
                KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + KQ_rowsum_add[col];
            }
#pragma unroll
            for (int col = 0; col < cols_per_thread; ++col) {
                const half2 scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
                for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                    for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                        VKQ_C[i].x[l0 + col] *= scale_h2;
                    }
                }
            }

            T_B_VKQ B[nbatch_fa/(2*T_B_VKQ::J)];
#pragma unroll
            for (int k = 0; k < nbatch_fa/(2*T_B_VKQ::J); ++k) {
                B[k] = get_half2(KQ_C[k]);
            }

#pragma unroll
            for (int chunk = 0; chunk < DV/chunk_ne; ++chunk) {
                const int slot = kvr.wait();
                half2 * tile_v = layout.kv(slot);
                const int i0_start = chunk*chunk_ne;
#pragma unroll
                for (int i_VKQ_0 = i0_start; i_VKQ_0 < i0_start + chunk_ne; i_VKQ_0 += T_A_VKQ::I) {
#pragma unroll
                    for (int k00 = 0; k00 < nbatch_fa/2; k00 += T_A_VKQ::J) {
                        T_A_VKQ A;
                        load_ldmatrix_trans_swizzle_128(A, tile_v, 2*k00, (i_VKQ_0 - i0_start)/2, chunk_h2);
                        mma(VKQ_C[i_VKQ_0/T_A_VKQ::I], B[k00/T_A_VKQ::J], A);
                    }
                }
                kvr.pop(slot);
            }
        }
    }

#pragma unroll
    for (int col = 0; col < cols_per_thread; ++col) {
        KQ_rowsum[col] += __shfl_xor_sync(0xffffffffu, KQ_rowsum[col], 2);
        KQ_rowsum[col] += __shfl_xor_sync(0xffffffffu, KQ_rowsum[col], 1);
    }

    if (sinks_f) {
        float KQ_max_scale[cols_per_thread];
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const int jc = threadIdx.y*cols_per_warp + T_C_KQ::get_i(2*col);
            const float sink = sinks_f[jc % ncols2];
            const float KQ_max_new = fmaxf(KQ_max[col], sink);
            const float diff = KQ_max[col] - KQ_max_new;
            KQ_max_scale[col] = expf(diff);
            KQ_max[col] = KQ_max_new;
            *reinterpret_cast<uint32_t *>(&KQ_max_scale[col]) *= diff >= SOFTMAX_FTZ_THRESHOLD;
            KQ_rowsum[col] = KQ_max_scale[col]*KQ_rowsum[col] + expf(sink - KQ_max_new);
        }
#pragma unroll
        for (int col = 0; col < cols_per_thread; ++col) {
            const half2 scale_h2 = make_half2(KQ_max_scale[col], KQ_max_scale[col]);
#pragma unroll
            for (int i = 0; i < (DV/2)/T_C_VKQ::J; ++i) {
#pragma unroll
                for (int l0 = 0; l0 < T_C_VKQ::ne; l0 += 2) {
                    VKQ_C[i].x[l0 + col] *= scale_h2;
                }
            }
        }
    }

    half2 * tile_out = layout.data;
    constexpr int tile_stride = nbatch_combine + 4;
    const int jc_meta = threadIdx.y*cols_per_warp + T_C_VKQ::get_i(threadIdx.x % 4);
    const float2 meta = make_float2(KQ_max[threadIdx.x % cols_per_thread], KQ_rowsum[threadIdx.x % cols_per_thread]);
    if (threadIdx.x % 4 < cols_per_thread) {
        reinterpret_cast<float2 *>(tile_out)[jc_meta*(tile_stride/2) + nbatch_combine/2] = meta;
    }
    consumer_sync();

#pragma unroll
    for (int k00 = 0; k00 < DV/2; k00 += nbatch_combine) {
        const int j0 = threadIdx.y*cols_per_warp;
#pragma unroll
        for (int k1 = 0; k1 < nbatch_combine; k1 += T_C_VKQ::J) {
#pragma unroll
            for (int l = 0; l < T_C_VKQ::ne; ++l) {
                const int j = j0 + T_C_VKQ::get_i(l);
                const int k = k1 + T_C_VKQ::get_j(l);
                tile_out[j*tile_stride + k] = VKQ_C[(k00 + k1)/T_C_VKQ::J].x[l];
            }
        }
        consumer_sync();

#pragma unroll
        for (int stride_k : {WARP_SIZE, WARP_SIZE/2, WARP_SIZE/4, WARP_SIZE/8}) {
            const int k0_start  = stride_k == WARP_SIZE ? 0 : nbatch_combine - nbatch_combine % (2*stride_k);
            const int k0_stop   =                             nbatch_combine - nbatch_combine % stride_k;
            const int stride_jc = WARP_SIZE/stride_k;
            if (k0_start == k0_stop) {
                continue;
            }
#pragma unroll
            for (int jc0 = 0; jc0 < ncols; jc0 += nwarps*stride_jc) {
                const int jc = jc0 + threadIdx.y*stride_jc + (stride_k == WARP_SIZE ? 0 : threadIdx.x/stride_k);
                if (jc0 + nwarps*stride_jc > ncols && jc >= ncols) {
                    break;
                }
                const int j = jc/ncols2;
                const int c = jc - j*ncols2;
                if ((ncols1 > 1 && jt*ncols1 + j >= int(ne01.z)) || (ncols2 > 1 && zt_gqa*ncols2 + c >= gqa_ratio)) {
                    continue;
                }
                const float2 meta = reinterpret_cast<const float2 *>(tile_out)[jc*(tile_stride/2) + nbatch_combine/2];
#pragma unroll
                for (int k0 = k0_start; k0 < k0_stop; k0 += stride_k) {
                    const int k = k0 + (stride_k == WARP_SIZE ? threadIdx.x : threadIdx.x % stride_k);
                    if constexpr (split_k == 1) {
                        const float2 value = __half22float2(tile_out[jc*tile_stride + k]);
                        dstk[((jt*ncols1 + j)*ne02 + c)*(DV/2) + k00 + k] = meta.y == 0.0f ? make_float2(0.0f, 0.0f) : make_float2(value.x/meta.y, value.y/meta.y);
                    } else {
                        dst_parts[(j*ne02 + c)*split_k*(DV/2) + k00 + k] = tile_out[jc*tile_stride + k];
                        if (k00 == 0 && k == 0) {
                            dst_meta[(j*ne02 + c)*split_k] = meta;
                        }
                    }
                }
            }
        }
        consumer_sync();
    }

#else
    GGML_UNUSED_VARS(sinks_f, dstk, dst_parts, dst_meta, slope, logit_softcap, ne01, ne02, gqa_ratio, jt, zt_gqa, kb0_stop, layout);
    NO_DEVICE_CODE;
#endif
}

template<int DKQ, int ncols1, int ncols2, bool use_logit_softcap, int split_k>
__global__ __launch_bounds__((nwarps + 1)*WARP_SIZE, 1) void kernel(
        __grid_constant__ const CUtensorMap map_k,
        __grid_constant__ const CUtensorMap map_v,
        __grid_constant__ const CUtensorMap map_mask,
        const char * Q_ptr, const char * sinks_ptr,
        const int * KV_max_ptr, float * dst_ptr, half * dst_parts_ptr, float2 * dst_meta_ptr,
        const float scale, const float max_bias, const float m0, const float m1,
        const uint32_t n_head_log2, const float logit_softcap,
        const uint3 ne01, const int32_t ne02,
        const int32_t nb01, const int32_t nb02, const int32_t nb03,
        const int32_t ne11, const int32_t ne12, const int32_t ne33) {
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ == GGML_CUDA_CC_BLACKWELL && defined(TURING_MMA_AVAILABLE)
    ggml_cuda_pdl_sync();
    extern __shared__ __align__(1024) uint8_t smem[];
    shared_layout<DKQ, ncols1> layout(smem);

    if (threadIdx.x == 0 && threadIdx.y == 0) {
        barrier_init(&layout.barriers->q_ready, 1);
#pragma unroll
        for (int i = 0; i < depth; ++i) {
            barrier_init(&layout.barriers->kv.produced[i], 1);
            barrier_init(&layout.barriers->kv.consumed[i], nwarps*WARP_SIZE);
            barrier_init(&layout.barriers->mask.produced[i], 1);
            barrier_init(&layout.barriers->mask.consumed[i], nwarps*WARP_SIZE);
        }
    }
    __syncthreads();

    constexpr int nbatch_fa = config<DKQ>::nbatch_fa;
    const int jt = blockIdx.x;
    const int split = blockIdx.y % split_k;
    const int tile_y = blockIdx.y / split_k;
    const int ntiles_z_gqa = (ne02/ne12 + ncols2 - 1)/ncols2;
    const int z_KV = tile_y/ntiles_z_gqa;
    const int zt_gqa = tile_y - z_KV*ntiles_z_gqa;
    const int sequence = blockIdx.z;
    const int gqa_ratio = ne02/ne12;
    const int zt_Q = z_KV*gqa_ratio + zt_gqa*ncols2;
    int kb0_total = ne11/nbatch_fa;
    if (KV_max_ptr) {
        kb0_total = min(kb0_total, KV_max_ptr[sequence*gridDim.x + jt]/nbatch_fa);
    }
    if constexpr (split_k == 1) {
        if (kb0_total <= 0) {
            return;
        }
    }
    const int kb0_start = int64_t(kb0_total)*split/split_k;
    const int kb0_stop = int64_t(kb0_total)*(split + 1)/split_k;
    const float2 * Q_f2 = reinterpret_cast<const float2 *>(Q_ptr + int64_t(nb03)*sequence + int64_t(nb02)*zt_Q);
    const float * sinks_f = sinks_ptr && split == 0 ? reinterpret_cast<const float *>(sinks_ptr) + zt_Q : nullptr;
    float2 * dstk = reinterpret_cast<float2 *>(dst_ptr) + (int64_t(sequence)*ne01.z*ne02 + zt_Q)*(DKQ/2);
    half2 * dst_parts = nullptr;
    float2 * dst_meta = nullptr;
    if constexpr (split_k > 1) {
        const int64_t row0 = (int64_t(sequence)*ne01.z + jt*ncols1)*ne02 + zt_Q;
        dst_parts = reinterpret_cast<half2 *>(dst_parts_ptr) + (row0*split_k + split)*(DKQ/2);
        dst_meta = dst_meta_ptr + row0*split_k + split;
    }
    const float slope = get_alibi_slope(max_bias, zt_Q, n_head_log2, m0, m1);

    if (threadIdx.y == nwarps) {
        producer<DKQ, ncols1, ncols2>(Q_f2, scale, nb01/sizeof(float2), nb02/sizeof(float2),
            jt, zt_gqa, gqa_ratio, ne01, sequence, ne33, z_KV, kb0_start, kb0_stop,
            &map_k, &map_v, &map_mask, layout);
    } else {
        consumer<DKQ, ncols1, ncols2, use_logit_softcap, split_k>(sinks_f, dstk, dst_parts, dst_meta, slope, logit_softcap,
            ne01, ne02, gqa_ratio, jt, zt_gqa, kb0_stop - kb0_start, layout);
    }

#else
    GGML_UNUSED_VARS(Q_ptr, sinks_ptr, KV_max_ptr, dst_ptr, dst_parts_ptr, dst_meta_ptr, scale, max_bias, m0, m1,
        n_head_log2, logit_softcap, ne01, ne02, nb01, nb02, nb03, ne11, ne12, ne33, map_k, map_v, map_mask);
    NO_DEVICE_CODE;
#endif
}

template<int D>
__launch_bounds__(D, 1)
static __global__ void combine_results(
        const half * parts_ptr,
        const float2 * meta_ptr,
        float * dst_ptr) {
    constexpr int split_k = 3;
    ggml_cuda_pdl_lc();
    const int row = (blockIdx.z*gridDim.x + blockIdx.x)*gridDim.y + blockIdx.y;
    const half * parts = parts_ptr + row*split_k*D;
    const float2 * meta_src = meta_ptr + row*split_k;
    float * dst = dst_ptr + row*D;
    const int tid = threadIdx.x;
    __builtin_assume(tid < D);

    extern __shared__ float2 meta[];
    ggml_cuda_pdl_sync();
    if (tid < split_k) {
        meta[tid] = meta_src[tid];
    }
    __syncthreads();

    float kqmax = meta[0].x;
#pragma unroll
    for (int split = 1; split < split_k; ++split) {
        kqmax = max(kqmax, meta[split].x);
    }

    float numerator = 0.0f;
    float denominator = 0.0f;
#pragma unroll
    for (int split = 0; split < split_k; ++split) {
        const float scale = expf(meta[split].x - kqmax);
        numerator += scale*__half2float(parts[split*D + tid]);
        denominator += scale*meta[split].y;
    }
    dst[tid] = denominator == 0.0f ? 0.0f : numerator/denominator;
}

static inline CUtensorMap make_map(
        void * ptr, const uint64_t (&dims)[4], const uint64_t (&strides)[3], const uint32_t (&box)[4],
        const CUtensorMapSwizzle swizzle) {
    CUtensorMap map{};
    const uint32_t element_strides[4] = {1, 1, 1, 1};
    const CUresult result = cuTensorMapEncodeTiled(&map, CU_TENSOR_MAP_DATA_TYPE_FLOAT16, 4, ptr,
        dims, strides, box, element_strides, CU_TENSOR_MAP_INTERLEAVE_NONE, swizzle,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B, CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    GGML_ASSERT(result == CUDA_SUCCESS);
    return map;
}

template<int DKQ, int ncols1, int ncols2>
static bool supported(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    if (cc != GGML_CUDA_CC_BLACKWELL || !blackwell_mma_available(cc) || ncols1*ncols2 != 64 || Q->ne[1] <= 32) {
        return false;
    }
    if (K->type != GGML_TYPE_F16 || V->type != GGML_TYPE_F16 || !mask || mask->type != GGML_TYPE_F16) {
        return false;
    }
    if (K->ne[0] != DKQ || V->ne[0] != DKQ || K->ne[1] % FATTN_KQ_STRIDE != 0) {
        return false;
    }
    if (dst->src[4] && (Q->ne[2]/K->ne[2]) % ncols2 != 0) {
        return false;
    }
    if ((reinterpret_cast<uintptr_t>(K->data) | reinterpret_cast<uintptr_t>(V->data) |
         reinterpret_cast<uintptr_t>(mask->data)) % 16 != 0) {
        return false;
    }
    for (const ggml_tensor * tensor : {K, V, mask}) {
        for (int i = 1; i < 4; ++i) {
            if (tensor->nb[i] % 16 != 0) {
                return false;
            }
        }
    }
    float max_bias = 0.0f;
    memcpy(&max_bias, reinterpret_cast<const float *>(dst->op_params) + 1, sizeof(float));
    return max_bias == 0.0f;
}

template<int DKQ, int ncols1, int ncols2>
void launch(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * Q = dst->src[0];
    const ggml_tensor * K = dst->src[1];
    const ggml_tensor * V = dst->src[2];
    const ggml_tensor * mask = dst->src[3];
    const ggml_tensor * sinks = dst->src[4];
    constexpr int nbatch_fa = config<DKQ>::nbatch_fa;

    ggml_cuda_pool & pool = ctx.pool();
    cudaStream_t stream = ctx.stream();
    ggml_cuda_pool_alloc<int> KV_max(pool);
    ggml_cuda_pool_alloc<half> dst_parts(pool);
    ggml_cuda_pool_alloc<float2> dst_meta(pool);
    const int ntiles_x = (Q->ne[1] + ncols1 - 1)/ncols1;
    const int ne_KV_max = ntiles_x*Q->ne[3];
    KV_max.alloc(ne_KV_max);
    const dim3 blocks_KV_max(ntiles_x, Q->ne[3], 1);
    const dim3 threads_KV_max(FATTN_KQ_STRIDE/2, 1, 1);
    const ggml_cuda_kernel_launch_params params_KV_max(blocks_KV_max, threads_KV_max, 0, stream);
    ggml_cuda_kernel_launch(flash_attn_mask_to_KV_max<ncols1, DKQ>, params_KV_max,
        reinterpret_cast<const half2 *>(mask->data), KV_max.ptr, int(K->ne[1]/FATTN_KQ_STRIDE),
        int64_t(mask->nb[1]/sizeof(half2)), int64_t(mask->nb[3]/sizeof(half2)), int(mask->ne[3]),
        reinterpret_cast<float *>(dst->data), int(Q->ne[1]), int(Q->ne[2]));

    const uint64_t dims_k[4] = {uint64_t(K->ne[0]), uint64_t(K->ne[1]), uint64_t(K->ne[2]), uint64_t(K->ne[3])};
    const uint64_t dims_v[4] = {uint64_t(V->ne[0]), uint64_t(V->ne[1]), uint64_t(V->ne[2]), uint64_t(V->ne[3])};
    const uint64_t dims_m[4] = {uint64_t(mask->ne[0]), uint64_t(mask->ne[1]), uint64_t(mask->ne[2]), uint64_t(mask->ne[3])};
    const uint64_t strides_k[3] = {K->nb[1], K->nb[2], K->nb[3]};
    const uint64_t strides_v[3] = {V->nb[1], V->nb[2], V->nb[3]};
    const uint64_t strides_m[3] = {mask->nb[1], mask->nb[2], mask->nb[3]};
    const uint32_t box_kv[4] = {chunk_ne, nbatch_fa, 1, 1};
    const uint32_t box_m[4] = {nbatch_fa, ncols1, 1, 1};
    const CUtensorMap map_k = make_map(K->data, dims_k, strides_k, box_kv, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap map_v = make_map(V->data, dims_v, strides_v, box_kv, CU_TENSOR_MAP_SWIZZLE_128B);
    const CUtensorMap map_m = make_map(mask->data, dims_m, strides_m, box_m, CU_TENSOR_MAP_SWIZZLE_NONE);

    float scale = 1.0f;
    float max_bias = 0.0f;
    float logit_softcap = 0.0f;
    memcpy(&scale, reinterpret_cast<const float *>(dst->op_params) + 0, sizeof(float));
    memcpy(&max_bias, reinterpret_cast<const float *>(dst->op_params) + 1, sizeof(float));
    memcpy(&logit_softcap, reinterpret_cast<const float *>(dst->op_params) + 2, sizeof(float));
    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }
    const uint32_t n_head = Q->ne[2];
    const uint32_t n_head_log2 = 1u << uint32_t(floorf(log2f(float(n_head))));
    const float m0 = powf(2.0f, -max_bias/n_head_log2);
    const float m1 = powf(2.0f, -(max_bias/2.0f)/n_head_log2);
    const uint3 ne01 = init_fastdiv_values(Q->ne[1]);
    const int gqa_ratio = Q->ne[2]/K->ne[2];
    const int ntiles_z_gqa = (gqa_ratio + ncols2 - 1)/ncols2;
    int split_k = 1;
    if constexpr (DKQ == 256 || DKQ == 512) {
        const int ntiles_dst = ntiles_x*ntiles_z_gqa*K->ne[2]*Q->ne[3];
        const int nsm = ggml_cuda_info().devices[ggml_cuda_get_device()].nsm;
        const int nwaves = (ntiles_dst + nsm - 1)/nsm;
        const int nwaves_split3 = (3*ntiles_dst + nsm - 1)/nsm;
        const float split3_wave_ratio = float(nwaves_split3)/(3.0f*nwaves);
        if (K->ne[1] >= 1024 && split3_wave_ratio <= 0.95f) {
            split_k = 3;
        }
    }
    if (split_k == 3) {
        dst_parts.alloc(split_k*ggml_nelements(dst));
        dst_meta.alloc(split_k*ggml_nrows(dst));
    }
    const dim3 blocks(ntiles_x, split_k*ntiles_z_gqa*K->ne[2], Q->ne[3]);
    const dim3 threads(WARP_SIZE, nwarps + 1, 1);
    constexpr size_t shared_bytes = shared_layout<DKQ, ncols1>::shared_bytes;

    if (logit_softcap == 0.0f) {
        auto fn = kernel<DKQ, ncols1, ncols2, false, 1>;
        if constexpr (DKQ == 256 || DKQ == 512) {
            if (split_k == 3) {
                fn = kernel<DKQ, ncols1, ncols2, false, 3>;
            }
        }
        CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        const ggml_cuda_kernel_launch_params params(blocks, threads, shared_bytes, stream);
        ggml_cuda_kernel_launch(fn, params,
            map_k, map_v, map_m,
            reinterpret_cast<const char *>(Q->data), sinks ? reinterpret_cast<const char *>(sinks->data) : nullptr,
            KV_max.ptr, reinterpret_cast<float *>(dst->data), dst_parts.ptr, dst_meta.ptr, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            ne01, int32_t(Q->ne[2]), int32_t(Q->nb[1]), int32_t(Q->nb[2]), int32_t(Q->nb[3]),
            int32_t(K->ne[1]), int32_t(K->ne[2]), int32_t(mask->ne[3]));
    } else {
        auto fn = kernel<DKQ, ncols1, ncols2, true, 1>;
        if constexpr (DKQ == 256 || DKQ == 512) {
            if (split_k == 3) {
                fn = kernel<DKQ, ncols1, ncols2, true, 3>;
            }
        }
        CUDA_CHECK(cudaFuncSetAttribute(fn, cudaFuncAttributeMaxDynamicSharedMemorySize, shared_bytes));
        const ggml_cuda_kernel_launch_params params(blocks, threads, shared_bytes, stream);
        ggml_cuda_kernel_launch(fn, params,
            map_k, map_v, map_m,
            reinterpret_cast<const char *>(Q->data), sinks ? reinterpret_cast<const char *>(sinks->data) : nullptr,
            KV_max.ptr, reinterpret_cast<float *>(dst->data), dst_parts.ptr, dst_meta.ptr, scale, max_bias, m0, m1, n_head_log2, logit_softcap,
            ne01, int32_t(Q->ne[2]), int32_t(Q->nb[1]), int32_t(Q->nb[2]), int32_t(Q->nb[3]),
            int32_t(K->ne[1]), int32_t(K->ne[2]), int32_t(mask->ne[3]));
    }

    if (split_k == 3) {
        const dim3 blocks_combine(Q->ne[1], Q->ne[2], Q->ne[3]);
        const dim3 threads_combine(DKQ, 1, 1);
        const ggml_cuda_kernel_launch_params params_combine(blocks_combine, threads_combine, split_k*sizeof(float2), stream);
        ggml_cuda_kernel_launch(combine_results<DKQ>, params_combine,
            dst_parts.ptr, dst_meta.ptr, reinterpret_cast<float *>(dst->data));
    }
}

} // namespace ggml_cuda_fattn_sm120

#define DECL_FATTN_MMA_F16_SM120_CASE(DKQ, ncols1, ncols2) \
    template void ggml_cuda_fattn_sm120::launch<DKQ, ncols1, ncols2>(ggml_backend_cuda_context &, ggml_tensor *)

extern DECL_FATTN_MMA_F16_SM120_CASE(128,  8, 8);
extern DECL_FATTN_MMA_F16_SM120_CASE(256,  8, 8);
extern DECL_FATTN_MMA_F16_SM120_CASE(256, 32, 2);
extern DECL_FATTN_MMA_F16_SM120_CASE(512,  8, 8);

#endif
