#pragma once

#include "vecdotq.cuh"
#include "mma.cuh"

using namespace ggml_cuda_mma;

#include "mmq.cuh"

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_0_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_0, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_0*VDR_Q4_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_0_Q8_1_MMQ];

                constexpr int max_cpy = ggml_cuda_get_max_cpy_bytes();
                constexpr int mcpy_int = max_cpy / sizeof(int);
                static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

                int tmp0[4], tmp1[4];

                #pragma unroll
                for (int l0 = 0; l0 < 4 / mcpy_int; ++l0) {
                    ggml_cuda_memcpy_1<max_cpy>(tmp0 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + l0 * mcpy_int]  );
                    ggml_cuda_memcpy_1<max_cpy>(tmp1 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + QI4_0 + l0 * mcpy_int]);
                }

                u[0]=tmp0[0]; u[2]=tmp0[1]; u[4]=tmp0[2]; u[6]=tmp0[3];
                u[1]=tmp1[0]; u[3]=tmp1[1]; u[5]=tmp1[2]; u[7]=tmp1[3];

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMQ>
                    (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_0], u,
                     x_df[i*(MMQ_TILE_NE_K/QI4_0) + i/QI4_0 + k0/(QR4_0*QI4_0)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_1_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_1, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_1*VDR_Q4_1_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;
                const int kyqs = QI8_1 * ((k01/2) / (QI8_1/2)) + (k01/2) % (QI8_1/2);

                int u[2*VDR_Q4_1_Q8_1_MMQ];

                constexpr int max_cpy = ggml_cuda_get_max_cpy_bytes();
                constexpr int mcpy_int = max_cpy / sizeof(int);
                static_assert(VDR_Q4_0_Q8_1_MMQ == 4, "bad VDR_Q4_0_Q8_1_MMQ");

                int tmp0[4], tmp1[4];

                #pragma unroll
                for (int l0 = 0; l0 < 4 / mcpy_int; ++l0) {
                    ggml_cuda_memcpy_1<max_cpy>(tmp0 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + l0 * mcpy_int]  );
                    ggml_cuda_memcpy_1<max_cpy>(tmp1 + l0 * mcpy_int, &y_qs[j*MMQ_TILE_Y_K + kyqs + QI4_1 + l0 * mcpy_int]);
                }

                u[0]=tmp0[0]; u[2]=tmp0[1]; u[4]=tmp0[2]; u[6]=tmp0[3];
                u[1]=tmp1[0]; u[3]=tmp1[1]; u[5]=tmp1[2]; u[7]=tmp1[3];

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMQ>
                    (&x_qs[i*(MMQ_TILE_NE_K + 1) + k0/QR4_1], u,
                     x_dm[i*(MMQ_TILE_NE_K/QI4_1) + i/QI4_1 + k0/(QR4_1*QI4_1)], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q8_0, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_0_q8_1_impl<float, VDR_Q8_0_Q8_1_MMQ>
                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k0 % MMQ_TILE_NE_K],
                     x_df[i*(2*MMQ_TILE_NE_K/QI8_0) + i/(QI8_0/2) + k0/QI8_0], y_df[j*MMQ_TILE_Y_K + (k0/QI8_1) % (MMQ_TILE_NE_K/QI8_1)]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback, mmq_q8_1_ds_layout ds_layout>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_q8_1_mma(
    const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            float dB;
            const int j = j0 + tile_C::get_j(0);
            if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            } else {
                dB = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    const float dA = x_df[i*sram_stride + k0/QI8_0];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA*dB;
                }
            }
        }
    }
#else
    typedef tile<16, 8, int> tile_A;
    typedef tile< 8, 8, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;
    const half2 * y_ds = (const half2 *) y;

    tile_A A[ntx][MMQ_TILE_NE_K/QI8_0];
    float dA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_0];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_0], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
                const int k0 = k00 + k01;

                dA[n][l][k01/QI8_0] = x_df[i*sram_stride + k0/QI8_0];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
            tile_B B;
            float dB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                if (ds_layout == MMQ_Q8_1_DS_LAYOUT_D4) {
                    dB[l] =             y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
                } else {
                    dB[l] = __low2float(y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
                }
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_0], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l]*dA[n][l/2][k01/QI8_0]*dB[l%2];
                }
            }
        }
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}


template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_1, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += VDR_Q8_0_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_1_q8_1_impl<QR5_1*VDR_Q5_1_Q8_1_MMQ>
                    (&x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    x_dm[i*(MMQ_TILE_NE_K/QI5_1) + i/QI5_1 + k0/QI8_1], y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_1_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  8, int, input_layout>        tile_A;
    typedef tile<16,  8, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float2 dsB = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_A::I + tile_C::get_i(l);
                    float2 dmA = __half22float2(x_dm[i*sram_stride + k0/QI8_1]);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.x*dsB.x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA.y*dsB.y;
                }
            }
        }
    }
#else
    typedef tile<16,  8, int> tile_A;
    typedef tile< 8,  8, int> tile_B;
    typedef tile<16,  8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + 2*MMQ_TILE_NE_K;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_dm = (const half2 *) y;

    tile_A   A[ntx][MMQ_TILE_NE_K/QI8_1];
    float2 dmA[ntx][tile_C::ne/2][MMQ_TILE_NE_K/QI8_1];

    const int i0 = (threadIdx.y/ntx)*rows_per_warp;

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_A::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
                const int k0 = k00 + k01;

                dmA[n][l][k01/QI8_1] = __half22float2(x_dm[i*sram_stride + k0/QI8_1]);
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B   B;
            float2 dsB[tile_C::ne/2];

            load_generic(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K); // faster than load_ldmatrix

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dsB[l] = __half22float2(y_dm[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n][k01/QI8_1], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].x*dsB[l%2].x*C.x[l];
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dmA[n][l/2][k01/QI8_1].y*dsB[l%2].y;
                }
            }
        }
    }
#endif // defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
}

// Used for NVFP4, Q3_K, IQ2_S, and IQ2_XS
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(type, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_0) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q8_0_16_q8_1_impl<QI8_0>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0],
                    &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_df[i*(2*MMQ_TILE_NE_K*2/QI8_0) + i/(QI8_0/4) + k0/(QI8_0/2)],
                    y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

// Used for Q3_K, IQ2_S, and IQ2_XS:
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q8_0_16_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * x_df[i*sram_stride + k0/4] * dB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/8], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
                const int k0 = k00 + k01;

                dA[n][l][k01/4] = x_df[i*sram_stride + k0/4];
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += dB[l%2]*(C[0].x[l]*dA[n][l/2][k01/4 + 0] + C[1].x[l]*dA[n][l/2][k01/4 + 1]);
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q2_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q2_K, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    float2 y_df[J/nwarps];
#pragma unroll
    for (int j0 = 0; j0 < J; j0 += nwarps) {
        const int j = j0 + threadIdx.y;

        y_df[j0/nwarps] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
    }

#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K/2; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                constexpr int ns = 2;
                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }
        }
    }

    // Some compilers fail to unroll the loop over k01 if there is a conditional statement for ns in the inner loop.
    // As a workaround 2 separate loops are used instead.
#pragma unroll
    for (int k01 = MMQ_TILE_NE_K/2; k01 < MMQ_TILE_NE_K; k01 += QR2_K*VDR_Q2_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                constexpr int ns = 1;
                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q2_K_q8_1_impl_mmq<ns>(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01],
                    &x_dm[i*(MMQ_TILE_NE_K + 1) + k0/4], k01 < MMQ_TILE_NE_K/2 ? y_df[j0/nwarps].x : y_df[j0/nwarps].y,
                    &y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q2_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = (k01 < MMQ_TILE_NE_K/2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K]).x : __half22float2(y_ds[j*MMQ_TILE_Y_K]).y;
            const float sB = (k01 >= MMQ_TILE_NE_K * 3/4) ? 0
                                              : (((k01/4)%2) ? __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).y
                                                             : __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]).x);

            tile_C Cm;
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
#pragma unroll
                for (int l = 0; l < tile_A::ne; ++l) {
                    A1.x[l] = 0x01010101;
                }
                mma(Cm, A1, B);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd;
                mma(Cd, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const float2 dm = __half22float2(x_dm[i*sram_stride + k0/4]);
                    float tmp = Cd.x[l]*dm.x;
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm.x[l]*dm.y;
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*dB;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= dm.y*sB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile<16, 8, int> tile_A_8;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + MMQ_TILE_NE_K*2;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A  A[ntx][8];
    float  dA[ntx][tile_C::ne/2][8];
    float  mA[ntx][tile_C::ne/2][8];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            const int k0 = k00 + k01;

            load_ldmatrix(((tile_A_8 *) A[n])[k01/QI8_1], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }
    }

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

#pragma unroll
            for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1/2) {
                const int k0 = k00 + k01;

                const float2 dm = __half22float2(x_dm[i*sram_stride + k0/(QI8_1/2)]);

                dA[n][l][k01/(QI8_1/2)] = dm.x;
                mA[n][l][k01/(QI8_1/2)] = dm.y;
            }
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float2 dB[tile_C::ne/2];

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int j = j0 + tile_C::get_j(l);

            dB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K]);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QI8_1) {
            tile_B B[2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + (k01 + 0),         MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + (k01 + tile_B::J), MMQ_TILE_Y_K);

            tile_C Cm[2];
            if (k01 >= MMQ_TILE_NE_K * 3/4) {
                tile_A A1;
                A1.x[0] = 0x01010101;
                A1.x[1] = 0x01010101;
                mma(Cm[0], A1, B[0]);
                mma(Cm[1], A1, B[1]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C Cd[2];

                mma(Cd[0], A[n][k01/4 + 0], B[0]);
                mma(Cd[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    float tmp = Cd[0].x[l]*dA[n][l/2][k01/4 + 0] + Cd[1].x[l]*dA[n][l/2][k01/4 + 1];
                    if (k01 >= MMQ_TILE_NE_K * 3/4) {
                        tmp -= Cm[0].x[l]*mA[n][l/2][k01/4 + 0] + Cm[1].x[l]*mA[n][l/2][k01/4 + 1];
                    }
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp*(k01 < MMQ_TILE_NE_K/2 ? dB[l%2].x : dB[l%2].y);
                }
            }
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K * 3/4; k01 += QI8_1) {
            float2 sB[tile_C::ne/2];

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                sB[l] = __half22float2(y_ds[j*MMQ_TILE_Y_K + (1 + k01/QI8_1)]);
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 0]*sB[l%2].x;
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] -= mA[n][l/2][k01/4 + 1]*sB[l%2].y;
                }
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q3_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q3_K, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR3_K*VDR_Q3_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const int8_t * scales = ((const int8_t *) (x_sc + i*(MMQ_TILE_NE_K/8) + i/8)) + k0/4;

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q3_K_q8_1_impl_mmq(
                    &x_qs[i*(2*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], scales,
                    x_df[i], y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q4_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q4_K, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR4_K*VDR_Q4_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = (const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/32] + 2*(k01/16);

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q4_K_q8_1_impl_mmq(
                    &x_qs[i*(MMQ_TILE_NE_K + 1) + k0/2], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q5_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q5_K, I);
    const int   * x_qs = (const int   *) x;
    const half2 * x_dm = (const half2 *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_dm + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const half2 * y_ds = (const half2 *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR5_K*VDR_Q5_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const uint8_t * sc = ((const uint8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k00/32]) + 2*(k01/16);

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q5_K_q8_1_impl_mmq(
                    &x_qs[i*(QR5_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc, sc+8,
                    x_dm[i], &y_ds[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_dp4a(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
    constexpr int warp_size = ggml_cuda_get_physical_warp_size();
    constexpr int nwarps    = ggml_cuda_mmq_get_nthreads(type, J, fallback) / warp_size;
    constexpr int I         = ggml_cuda_mmq_get_I(type, J, fallback);

    constexpr tile_x_sizes txs = mmq_get_dp4a_tile_x_sizes(GGML_TYPE_Q6_K, I);
    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + txs.qs;
    const int   * x_sc = (const int   *) x_df + txs.dm;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

// #pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += QR6_K*VDR_Q6_K_Q8_1_MMQ) {
        const int k0 = k00 + k01;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += nwarps) {
            const int j = j0 + threadIdx.y;

#pragma unroll
            for (int i0 = 0; i0 < I; i0 += warp_size) {
                const int i = i0 + threadIdx.x;

                const int8_t * sc = ((const int8_t *) &x_sc[i * (MMQ_TILE_NE_K/8) + i/8 + k0/16]);

                sum[j0/nwarps*I/warp_size + i0/warp_size] += vec_dot_q6_K_q8_1_impl_mmq(
                    &x_qs[i*(QR6_K*MMQ_TILE_NE_K + 1) + k0], &y_qs[j*MMQ_TILE_Y_K + k01], sc,
                    x_df[i*(MMQ_TILE_NE_K/QI6_K) + i/QI6_K], &y_df[j*MMQ_TILE_Y_K + k01/QI8_1]);
            }
        }
    }
}

template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_q6_K_q8_1_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {
#if defined(AMD_MFMA_AVAILABLE) || defined(AMD_WMMA_AVAILABLE)
    constexpr data_layout input_layout = get_input_data_layout();
    typedef tile<16,  4, int, input_layout>        tile_A;
    typedef tile<16,  4, int, input_layout>        tile_B;
    typedef tile<16, 16, int, DATA_LAYOUT_J_MAJOR> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * rows_per_warp;

    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 4) {
        const int k0 = k00 + k01;

        tile_A A[ntx];
#pragma unroll
        for (int n = 0; n < ntx; ++n) {
            load_ldmatrix(A[n], x_qs + (i0 + n*tile_A::I)*sram_stride + k0, sram_stride);
        }

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
            tile_B B;
            load_ldmatrix(B, y_qs + j0*MMQ_TILE_Y_K + k01, MMQ_TILE_Y_K);

            const int j = j0 + tile_C::get_j(0);
            const float dB = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C;
                mma(C, A[n], B);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    const int i = i0 + n*tile_C::I + tile_C::get_i(l);
                    const int8_t * sc = (const int8_t *) (x_sc + i*sram_stride + k00/16);
                    sum[(j0/tile_C::J + n)*tile_C::ne + l] += C.x[l] * sc[k01/4] * x_df[i*sram_stride] * dB;
                }
            }
        }
    }
#elif defined(TURING_MMA_AVAILABLE)

    typedef tile<16, 4, int> tile_A;
    typedef tile< 8, 4, int> tile_B;
    typedef tile<16, 8, int> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp/tile_C::I; // Number of x minitiles per warp.

    y += (threadIdx.y % ntx) * (tile_C::J*MMQ_TILE_Y_K);

    const int   * x_qs = (const int   *) x;
    const float * x_df = (const float *) x_qs + MMQ_TILE_NE_K*2;
    const int   * x_sc = (const int   *) x_df + MMQ_TILE_NE_K/QI6_K;
    const int   * y_qs = (const int   *) y + 4;
    const float * y_df = (const float *) y;

    const int i0 = (threadIdx.y / ntx) * (ntx*tile_A::I);

    tile_A   A[ntx][8];
    int    scA[ntx][tile_C::ne/2][8];
    float   dA[ntx][tile_C::ne/2];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            const int k0 = k00 + k01;

            load_ldmatrix(A[n][k01/4 + 0], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + 0),         sram_stride);
            load_ldmatrix(A[n][k01/4 + 1], x_qs + (i0 + n*tile_A::I)*sram_stride + (k0 + tile_A::J), sram_stride);
        }

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
            const int k0 = k00 + k01;

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

                const int      sc_packed = x_sc[i*sram_stride + k0/16];
                const int8_t * sc        = (const int8_t *) &sc_packed;

#pragma unroll
                for (int ksc = 0; ksc < sizeof(int); ++ksc) {
                    scA[n][l][k01/4 + ksc] = sc[ksc];
                }
            }
        }

#pragma unroll
        for (int l = 0; l < tile_C::ne/2; ++l) {
            const int i = i0 + n*tile_C::I + tile_C::get_i(2*l);

            dA[n][l] = x_df[i*sram_stride];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx*tile_C::J) {
        float tmp[ntx][tile_C::ne] = {{0.0f}};

#pragma unroll
        for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 8) {
            tile_B B[2];
            float dB[tile_C::ne/2];

            // Here load_generic is faster than load_ldmatrix.
            load_generic(B[0], y_qs + j0*MMQ_TILE_Y_K + 0         + k01, MMQ_TILE_Y_K);
            load_generic(B[1], y_qs + j0*MMQ_TILE_Y_K + tile_B::J + k01, MMQ_TILE_Y_K);

#pragma unroll
            for (int l = 0; l < tile_C::ne/2; ++l) {
                const int j = j0 + tile_C::get_j(l);

                dB[l] = y_df[j*MMQ_TILE_Y_K + k01/QI8_1];
            }

#pragma unroll
            for (int n = 0; n < ntx; ++n) {
                tile_C C[2];
                mma(C[0], A[n][k01/4 + 0], B[0]);
                mma(C[1], A[n][k01/4 + 1], B[1]);

#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    tmp[n][l] += (C[0].x[l]*scA[n][l/2][k01/4 + 0] + C[1].x[l]*scA[n][l/2][k01/4 + 1])*dB[l%2];
                }
            }
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int l = 0; l < tile_C::ne; ++l) {
                sum[(j0/tile_C::J + n)*tile_C::ne + l] += tmp[n][l]*dA[n][l/2];
            }
        }
    }
#else
    GGML_UNUSED_VARS(x, y, sum, k00);
    NO_DEVICE_CODE;
#endif // AMD_MFMA_AVAILABLE || AMD_WMMA_AVAILABLE
}

// ---------------------------------------------------------------------------------------------

// Shared MMA kernel for MXFP4 and NVFP4 on Blackwell.
// Both quantizations encode values as e2m1 (FP4) and produce one uint32 scale per
// m16n8k64 MMA call; only the PTX kind (scale_vec::2X ue8m0 vs scale_vec::4X ue4m3)
// and the per-type stride constant differ.
template <ggml_type type, int J, bool fallback> static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_fp4_fp4_mma(
        const int * __restrict__ x, const int * __restrict__ y, float * __restrict__ sum, const int k00) {

    typedef tile<16, 8, int>   tile_A;
    typedef tile<8,  8, int>   tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int I             = ggml_cuda_mmq_get_I(type, J, fallback);
    constexpr int sram_stride   = ggml_cuda_mmq_get_sram_stride(type, J, fallback);
    constexpr int rows_per_warp = ggml_cuda_mmq_get_rows_per_warp(type, J, fallback);
    constexpr int ntx           = rows_per_warp / tile_C::I;
    constexpr int nfrags        = MMQ_TILE_NE_K / tile_A::J;

    y += (threadIdx.y % ntx) * (tile_C::J * MMQ_TILE_Y_K);

    const int *      x_qs = (const int *) x;
    const uint32_t * x_sc = (const uint32_t *) (x_qs + 2 * MMQ_TILE_NE_K);
    const int *      y_qs = (const int *) y + 4;
    const uint32_t * y_sc = (const uint32_t *) y;

    // 2 threads per quad supply the packed scale register to the block_scale MMA,
    // see https://docs.nvidia.com/cuda/parallel-thread-execution/#warp-level-block-scaling
    const int tidx_A = threadIdx.x / 4 + (threadIdx.x % 2) * 8;
    const int tidx_B = threadIdx.x / 4;
    const int i0     = (threadIdx.y / ntx) * rows_per_warp;

    tile_A   A[ntx][nfrags];
    uint32_t scaleA[ntx][nfrags];

#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = k00 + frag * tile_A::J;
            load_ldmatrix(A[n][frag], x_qs + (i0 + n * tile_A::I) * sram_stride + k0, sram_stride);
            scaleA[n][frag] = x_sc[(i0 + n * tile_A::I + tidx_A) * sram_stride + k0 / tile_A::J];
        }
    }

#pragma unroll
    for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
        tile_B   B[nfrags];
        uint32_t scaleB[nfrags];

#pragma unroll
        for (int frag = 0; frag < nfrags; ++frag) {
            const int k0 = frag * tile_B::J;
            load_generic(B[frag], y_qs + j0 * MMQ_TILE_Y_K + k0, MMQ_TILE_Y_K);
            scaleB[frag] = y_sc[(j0 + tidx_B) * MMQ_TILE_Y_K + frag];
        }

#pragma unroll
        for (int n = 0; n < ntx; ++n) {
#pragma unroll
            for (int frag = 0; frag < nfrags; ++frag) {
                tile_C C = {};
                mma_block_scaled_fp4<type>(C, A[n][frag], B[frag], scaleA[n][frag], scaleB[frag]);
#pragma unroll
                for (int l = 0; l < tile_C::ne; ++l) {
                    sum[(j0 / tile_C::J + n) * tile_C::ne + l] += C.x[l];
                }
            }
        }
    }
}

#if defined(BLACKWELL_MMA_AVAILABLE)
template <bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_load_nvfp4_tile_A(
        tile<16, 8, int> & tile_a,
        uint32_t & scale,
        const block_nvfp4_blackwell * __restrict__ x,
        const int nvfp4_blocks_per_row,
        const int row_base,
        const int frag_abs,
        const int row_max) {
    const int lane = int(threadIdx.x) & 31;
    int row_lo_abs = row_base + (lane >> 2);
    int row_hi_abs = row_lo_abs + 8;
    if constexpr (fallback) {
        row_lo_abs = min(row_lo_abs, row_max);
        row_hi_abs = min(row_hi_abs, row_max);
    }

    const int block_rel = frag_abs / 4;
    const int frag_idx  = frag_abs % 4;
    const int word_idx  = lane & 3;
    int * tx = (int *) tile_a.x;

    if constexpr (!fallback) {
        const block_nvfp4_blackwell_frag & frag =
            x[(row_base / 16) * nvfp4_blocks_per_row + block_rel].tiles[frag_idx];
        const uint4 packed = reinterpret_cast<const uint4 *>(frag.regs)[lane];
        tx[0] = (int) packed.x;
        tx[1] = (int) packed.y;
        tx[2] = (int) packed.z;
        tx[3] = (int) packed.w;
        scale = frag.scales_u32[((lane >> 2) * 2) + (lane & 1)];
        return;
    }

    const block_nvfp4_blackwell & block_lo =
        x[(row_lo_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const block_nvfp4_blackwell & block_hi =
        x[(row_hi_abs / 16) * nvfp4_blocks_per_row + block_rel];
    const int row_lo = row_lo_abs % 16;
    const int row_hi = row_hi_abs % 16;

    tx[0] = (int) ggml_cuda_nvfp4_tile_q_word(block_lo, row_lo, frag_idx, word_idx + 0);
    tx[1] = (int) ggml_cuda_nvfp4_tile_q_word(block_hi, row_hi, frag_idx, word_idx + 0);
    tx[2] = (int) ggml_cuda_nvfp4_tile_q_word(block_lo, row_lo, frag_idx, word_idx + 4);
    tx[3] = (int) ggml_cuda_nvfp4_tile_q_word(block_hi, row_hi, frag_idx, word_idx + 4);
    scale = (lane & 1) == 0
        ? ggml_cuda_nvfp4_tile_scale_word(block_lo, row_lo, frag_idx)
        : ggml_cuda_nvfp4_tile_scale_word(block_hi, row_hi, frag_idx);
}

static __device__ __forceinline__ void ggml_cuda_mmq_load_nvfp4_tile_B(
        tile<8, 8, int> & tile_b,
        uint32_t & scale,
        const block_nvfp4_mmq & y_block,
        const int frag_idx) {
    const int lane  = int(threadIdx.x) & 31;
    const int group = lane & 3;
    const uint32_t * __restrict__ y_qs = y_block.qs_u32 + frag_idx * 8;
    int * tx = (int *) tile_b.x;

    tx[0] = (int) y_qs[group + 0];
    tx[1] = (int) y_qs[group + 4];

    uint32_t scale_word = 0;
    if (group == 0) {
        scale_word = y_block.sc4_u32[frag_idx];
    }
    scale = __shfl_sync(0xFFFFFFFFu, scale_word, lane & ~3);
}

template <int J>
static __device__ __forceinline__ void ggml_cuda_mmq_load_nvfp4_tile_B(
        tile<8, 8, int> & tile_b,
        uint32_t & scale,
        const block_nvfp4_mmq * __restrict__ y_blocks_j,
        const int frag_idx) {
    const int col = (int(threadIdx.x) & 31) >> 2;
    ggml_cuda_mmq_load_nvfp4_tile_B(tile_b, scale, y_blocks_j[col], frag_idx);
}

template <int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_nvfp4_nvfp4_direct(
        const block_nvfp4_blackwell * __restrict__ x_blocks,
        const int stride_row_x,
        const block_nvfp4_mmq * __restrict__ y,
        float * __restrict__ sum,
        const int k00,
        const int i_max,
        const float tensor_scale) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8,  8, int>   tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(GGML_TYPE_NVFP4, J, fallback) /
                                    ggml_cuda_get_physical_warp_size();
    constexpr int I               = ggml_cuda_mmq_get_I(GGML_TYPE_NVFP4, J, fallback);
    constexpr int rows_per_warp   = J >= 48 ? 32 : 16;
    constexpr int ntx             = rows_per_warp / tile_C::I;
    constexpr int rows_per_slab   = nwarps * tile_C::I;
    constexpr int groups_per_slab = J / tile_C::J;

    const int ty = threadIdx.y;
    const int ty_ntx_mod = ty % ntx;
    const int ty_ntx_div = ty / ntx;
    const block_nvfp4_mmq * __restrict__ y_blocks = y + ty_ntx_mod * tile_C::J;

#pragma unroll
    for (int k01 = 0; k01 < MMQ_TILE_NE_K; k01 += 16) {
        const int frag0 = k01 / 8;
        const int frag1 = frag0 + 1;

#pragma unroll
        for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
            const block_nvfp4_mmq * __restrict__ y_blocks_j = y_blocks + j0;
            tile_B B[2];
            uint32_t scale_b[2];
            ggml_cuda_mmq_load_nvfp4_tile_B<J>(B[0], scale_b[0], y_blocks_j, frag0);
            ggml_cuda_mmq_load_nvfp4_tile_B<J>(B[1], scale_b[1], y_blocks_j, frag1);

#pragma unroll
            for (int slab_row0 = 0; slab_row0 < I; slab_row0 += rows_per_slab) {
                tile_A A[ntx][2];
                uint32_t scale_a[ntx][2];
                const int i0 = slab_row0 + ty_ntx_div * rows_per_warp;
                const int sum_j = slab_row0 / rows_per_slab * groups_per_slab + j0 / tile_C::J;

#pragma unroll
                for (int n = 0; n < ntx; ++n) {
                    const int row_base = i0 + n * tile_A::I;
                    ggml_cuda_mmq_load_nvfp4_tile_A<fallback>(
                        A[n][0], scale_a[n][0], x_blocks, stride_row_x, row_base, k00 / 8 + frag0, i_max);
                    ggml_cuda_mmq_load_nvfp4_tile_A<fallback>(
                        A[n][1], scale_a[n][1], x_blocks, stride_row_x, row_base, k00 / 8 + frag1, i_max);
                }

#pragma unroll
                for (int n = 0; n < ntx; ++n) {
                    tile_C C[2];
                    float * __restrict__ sum_n = sum + (sum_j + n) * tile_C::ne;
                    mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C[0], A[n][0], B[0], scale_a[n][0], scale_b[0]);
                    mma_block_scaled_fp4<GGML_TYPE_NVFP4>(C[1], A[n][1], B[1], scale_a[n][1], scale_b[1]);
#pragma unroll
                    for (int l = 0; l < tile_C::ne; ++l) {
                        sum_n[l] += tensor_scale * (C[0].x[l] + C[1].x[l]);
                    }
                }
            }
        }
    }
}

static __device__ __forceinline__ uint32_t ggml_cuda_w4a8_expand_e2m1x4(const uint32_t packed, const int half) {
    const uint32_t p = half ? packed >> 16 : packed;
    const uint32_t q = __byte_perm(p, p, 0x1100);
    return ((q & 0x000F000Fu) << 2) | ((q & 0xF000F000u) >> 2);
}

static __device__ __forceinline__ uint32_t ggml_cuda_w4a8_scaled_e4m3x4(
        const uint32_t expanded_e2m1, const half2 scale_h2) {
    const uint32_t nibbles = (expanded_e2m1 >> 2) & 0x0F0F0F0Fu;
    const __nv_fp4x2_storage_t fp4_lo = (nibbles & 0x0Fu) | ((nibbles >> 4) & 0xF0u);
    const __nv_fp4x2_storage_t fp4_hi = ((nibbles >> 16) & 0x0Fu) | ((nibbles >> 20) & 0xF0u);
    const half2 h2_lo = __hmul2(half2(__nv_cvt_fp4x2_to_halfraw2(fp4_lo, __NV_E2M1)), scale_h2);
    const half2 h2_hi = __hmul2(half2(__nv_cvt_fp4x2_to_halfraw2(fp4_hi, __NV_E2M1)), scale_h2);
    const uint32_t fp8_lo = __nv_cvt_halfraw2_to_fp8x2(h2_lo, __NV_SATFINITE, __NV_E4M3);
    const uint32_t fp8_hi = __nv_cvt_halfraw2_to_fp8x2(h2_hi, __NV_SATFINITE, __NV_E4M3);
    return fp8_lo | (fp8_hi << 16);
}

static __device__ __forceinline__ void ggml_cuda_mmq_w4a8_make_scaled_e4m3_tile_A(
        tile<16, 8, int> & dst,
        const tile<16, 8, int> & src,
        const uint32_t scale_word_lo,
        const uint32_t scale_word_hi,
        const int pair) {
    int * __restrict__ d = (int *) dst.x;
    const int * __restrict__ s = (const int *) src.x;
    const int lane = int(threadIdx.x) & 31;
    const int lane_in_group = lane & 3;
    const int src_reg = 2 * pair;
    const int src_lane0 = (lane & ~3) + (lane_in_group >> 1);
    const int src_lane1 = src_lane0 + 2;
    const uint32_t lo0 = ggml_cuda_w4a8_expand_e2m1x4(
        __shfl_sync(0xFFFFFFFFu, (uint32_t) s[src_reg + 0], src_lane0), lane_in_group & 1);
    const uint32_t hi0 = ggml_cuda_w4a8_expand_e2m1x4(
        __shfl_sync(0xFFFFFFFFu, (uint32_t) s[src_reg + 1], src_lane0), lane_in_group & 1);
    const uint32_t lo1 = ggml_cuda_w4a8_expand_e2m1x4(
        __shfl_sync(0xFFFFFFFFu, (uint32_t) s[src_reg + 0], src_lane1), lane_in_group & 1);
    const uint32_t hi1 = ggml_cuda_w4a8_expand_e2m1x4(
        __shfl_sync(0xFFFFFFFFu, (uint32_t) s[src_reg + 1], src_lane1), lane_in_group & 1);

    // UE4M3 block scales are normalized up to 448. Divide by 8 before folding them into
    // E4M3 weights so that max(E2M1) * max(UE4M3) remains representable (6 * 448 / 8 = 336).
    const __nv_fp8x2_storage_t packed_scales_lo = (scale_word_lo >> (16 * pair)) & 0xFFFFu;
    const __nv_fp8x2_storage_t packed_scales_hi = (scale_word_hi >> (16 * pair)) & 0xFFFFu;
    const half2 inv_eight = __float2half2_rn(0.125f);
    const half2 scales_lo = __hmul2(half2(__nv_cvt_fp8x2_to_halfraw2(packed_scales_lo, __NV_E4M3)), inv_eight);
    const half2 scales_hi = __hmul2(half2(__nv_cvt_fp8x2_to_halfraw2(packed_scales_hi, __NV_E4M3)), inv_eight);
    d[0] = (int) ggml_cuda_w4a8_scaled_e4m3x4(lo0, __half2half2(__low2half(scales_lo)));
    d[1] = (int) ggml_cuda_w4a8_scaled_e4m3x4(hi0, __half2half2(__low2half(scales_hi)));
    d[2] = (int) ggml_cuda_w4a8_scaled_e4m3x4(lo1, __half2half2(__high2half(scales_lo)));
    d[3] = (int) ggml_cuda_w4a8_scaled_e4m3x4(hi1, __half2half2(__high2half(scales_hi)));
}

template <int J>
static __device__ __forceinline__ void ggml_cuda_mmq_load_w4a8_tile_B(
        tile<8, 8, int> & tile_b,
        const block_nvfp4_w4a8_mmq * __restrict__ y_blocks_j,
        const int frag_idx) {
    const int lane  = int(threadIdx.x) & 31;
    const int col   = lane >> 2;
    const int group = lane & 3;
    const uint32_t * __restrict__ y_qs = (const uint32_t *) y_blocks_j[col].qs + frag_idx * 8;
    int * tx = (int *) tile_b.x;
    tx[0] = (int) y_qs[group + 0];
    tx[1] = (int) y_qs[group + 4];
}

template <int J, bool fallback>
static __device__ __forceinline__ void ggml_cuda_mmq_vec_dot_nvfp4_w4a8_direct(
        const block_nvfp4_blackwell * __restrict__ x_blocks,
        const int stride_row_x,
        const block_nvfp4_w4a8_mmq * __restrict__ y,
        float * __restrict__ sum,
        const int k00,
        const int i_max,
        const float tensor_scale) {
    typedef tile<16, 8, int>   tile_A;
    typedef tile<8,  8, int>   tile_B;
    typedef tile<16, 8, float> tile_C;

    constexpr int nwarps          = ggml_cuda_mmq_get_nthreads(GGML_TYPE_NVFP4, J, fallback) /
                                    ggml_cuda_get_physical_warp_size();
    constexpr int I               = ggml_cuda_mmq_get_I(GGML_TYPE_NVFP4, J, fallback);
    constexpr int rows_per_warp   = J >= 48 ? 32 : 16;
    constexpr int ntx             = rows_per_warp / tile_C::I;
    constexpr int rows_per_slab   = nwarps * tile_C::I;
    constexpr int groups_per_slab = J / tile_C::J;

    const int ty = threadIdx.y;
    const int ty_ntx_mod = ty % ntx;
    const int ty_ntx_div = ty / ntx;
    const int lane = int(threadIdx.x) & 31;
    const block_nvfp4_w4a8_mmq * __restrict__ y_blocks = y + ty_ntx_mod * tile_C::J;

    if constexpr (J >= 48) {
#pragma unroll
        for (int weight_frag = 0; weight_frag < 4; ++weight_frag) {
#pragma unroll
            for (int slab_row0 = 0; slab_row0 < I; slab_row0 += rows_per_slab) {
                const int i0 = slab_row0 + ty_ntx_div * rows_per_warp;
#pragma unroll
                for (int n = 0; n < ntx; ++n) {
                    tile_A A_packed;
                    uint32_t scale_word;
                    const int row_base = i0 + n * tile_A::I;
                    ggml_cuda_mmq_load_nvfp4_tile_A<fallback>(
                        A_packed, scale_word, x_blocks, stride_row_x, row_base, k00 / 8 + weight_frag, i_max);
                    const uint32_t scale_word_lo = __shfl_sync(0xFFFFFFFFu, scale_word, lane & ~3);
                    const uint32_t scale_word_hi = __shfl_sync(0xFFFFFFFFu, scale_word, (lane & ~3) + 1);

#pragma unroll
                    for (int pair = 0; pair < 2; ++pair) {
                        tile_A A;
                        ggml_cuda_mmq_w4a8_make_scaled_e4m3_tile_A(
                            A, A_packed, scale_word_lo, scale_word_hi, pair);

#pragma unroll
                        for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
                            const block_nvfp4_w4a8_mmq * __restrict__ y_blocks_j = y_blocks + j0;
                            tile_B B;
                            tile_C C = {};
                            ggml_cuda_mmq_load_w4a8_tile_B<J>(B, y_blocks_j, 2 * weight_frag + pair);
                            mma_fp8_fp8(C, A, B);
                            const int sum_j = slab_row0 / rows_per_slab * groups_per_slab + j0 / tile_C::J;
                            float * __restrict__ sum_n = sum + (sum_j + n) * tile_C::ne;
#pragma unroll
                            for (int l = 0; l < tile_C::ne; ++l) {
                                sum_n[l] += C.x[l] * (8.0f * tensor_scale);
                            }
                        }
                    }
                }
            }
        }
    } else {
#pragma unroll
        for (int weight_frag = 0; weight_frag < 4; ++weight_frag) {
#pragma unroll
            for (int j0 = 0; j0 < J; j0 += ntx * tile_C::J) {
                const block_nvfp4_w4a8_mmq * __restrict__ y_blocks_j = y_blocks + j0;
                tile_B B[2];
                ggml_cuda_mmq_load_w4a8_tile_B<J>(B[0], y_blocks_j, 2 * weight_frag + 0);
                ggml_cuda_mmq_load_w4a8_tile_B<J>(B[1], y_blocks_j, 2 * weight_frag + 1);

#pragma unroll
                for (int slab_row0 = 0; slab_row0 < I; slab_row0 += rows_per_slab) {
                    const int i0 = slab_row0 + ty_ntx_div * rows_per_warp;
                    const int sum_j = slab_row0 / rows_per_slab * groups_per_slab + j0 / tile_C::J;

#pragma unroll
                    for (int n = 0; n < ntx; ++n) {
                        tile_A A_packed;
                        uint32_t scale_word;
                        const int row_base = i0 + n * tile_A::I;
                        ggml_cuda_mmq_load_nvfp4_tile_A<fallback>(
                            A_packed, scale_word, x_blocks, stride_row_x, row_base, k00 / 8 + weight_frag, i_max);
                        const uint32_t scale_word_lo = __shfl_sync(0xFFFFFFFFu, scale_word, lane & ~3);
                        const uint32_t scale_word_hi = __shfl_sync(0xFFFFFFFFu, scale_word, (lane & ~3) + 1);
                        float * __restrict__ sum_n = sum + (sum_j + n) * tile_C::ne;

#pragma unroll
                        for (int pair = 0; pair < 2; ++pair) {
                            tile_A A;
                            tile_C C = {};
                            ggml_cuda_mmq_w4a8_make_scaled_e4m3_tile_A(
                                A, A_packed, scale_word_lo, scale_word_hi, pair);
                            mma_fp8_fp8(C, A, B[pair]);
#pragma unroll
                            for (int l = 0; l < tile_C::ne; ++l) {
                                sum_n[l] += C.x[l] * (8.0f * tensor_scale);
                            }
                        }
                    }
                }
            }
        }
    }
}
#endif // defined(BLACKWELL_MMA_AVAILABLE)

