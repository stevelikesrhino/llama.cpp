#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "ggml.h"
#include "ggml-common.h"
#ifndef GGML_HD
#define GGML_HD __host__ __device__
#endif // GGML_HD

struct  __align__(16) block_nvfp4_blackwell_frag {
    uint32_t regs[32][4];
    uint32_t scales_u32[32];
};

struct  __align__(16) block_nvfp4_blackwell {
    block_nvfp4_blackwell_frag tiles[4];
};

struct  __align__(16) block_nvfp4_blackwell_tensor {
    float   weight_scale;
    float   input_scale;
    uint8_t pad[8];
    block_nvfp4_blackwell tiles[];
};

static_assert(sizeof(block_nvfp4_blackwell_frag) == 640, "unexpected nvfp4 blackwell fragment size");
static_assert(sizeof(block_nvfp4_blackwell) == 4 * sizeof(block_nvfp4_blackwell_frag), "unexpected nvfp4 blackwell size");
static_assert(sizeof(block_nvfp4_blackwell_tensor) == 16, "unexpected nvfp4 blackwell tensor header size");
static_assert(alignof(block_nvfp4_blackwell_frag) == 16, "nvfp4 blackwell fragment must be 16B aligned");
static_assert(alignof(block_nvfp4_blackwell) == 16, "nvfp4 blackwell must be 16B aligned");
static_assert(alignof(block_nvfp4_blackwell_tensor) == 16, "nvfp4 blackwell tensor must be 16B aligned");

static inline GGML_HD uint32_t ggml_cuda_nvfp4_tile_q_word(
    const block_nvfp4_blackwell & tile, int row_in_tile, int frag_idx, int pack_idx) {
        const int lane = ((row_in_tile & 7) * 4) + (pack_idx & 3);
        const int reg  = (row_in_tile >> 3) + ((pack_idx >> 2) << 1);
        return tile.tiles[frag_idx].regs[lane][reg];
}

static inline GGML_HD uint32_t ggml_cuda_nvfp4_tile_scale_word(
    const block_nvfp4_blackwell & tile, int row_in_tile, int frag_idx) {
        const int lane = ((row_in_tile & 7) * 4) + (row_in_tile >> 3);
        return tile.tiles[frag_idx].scales_u32[lane];
}

static inline GGML_HD int64_t ggml_cuda_bw_div_up(int64_t n, int64_t d) {
        return (n + d - 1) / d;
}

static inline uint32_t ggml_cuda_bw_pack8(const uint8_t * p, int shift) {
    return
        (((uint32_t)((p[0] >> shift) & 0x0F)) <<  0) |
        (((uint32_t)((p[1] >> shift) & 0x0F)) <<  4) |
        (((uint32_t)((p[2] >> shift) & 0x0F)) <<  8) |
        (((uint32_t)((p[3] >> shift) & 0x0F)) << 12) |
        (((uint32_t)((p[4] >> shift) & 0x0F)) << 16) |
        (((uint32_t)((p[5] >> shift) & 0x0F)) << 20) |
        (((uint32_t)((p[6] >> shift) & 0x0F)) << 24) |
        (((uint32_t)((p[7] >> shift) & 0x0F)) << 28);
}

static inline GGML_HD int64_t ggml_cuda_nvfp4_blocks_per_row(int64_t ne0) {
    return ggml_cuda_bw_div_up(ne0, QK_K);
}

static inline size_t ggml_cuda_nvfp4_plane_size(int64_t ne0, int64_t nrows) {
    return (size_t) ggml_cuda_bw_div_up(nrows, 16) *
           (size_t) ggml_cuda_nvfp4_blocks_per_row(ne0) * sizeof(block_nvfp4_blackwell);
}

static inline size_t ggml_cuda_nvfp4_tensor_size(int64_t ne0, int64_t ne1, int64_t nplanes) {
    return sizeof(block_nvfp4_blackwell_tensor) + (size_t) nplanes * ggml_cuda_nvfp4_plane_size(ne0, ne1);
}

static inline size_t ggml_cuda_nvfp4_tensor_alloc_size(const ggml_tensor * tensor) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    return ggml_cuda_nvfp4_tensor_size(ne0, tensor->ne[1], nplanes);
}

static inline void ggml_cuda_nvfp4_set_tensor_header(const ggml_tensor * tensor, block_nvfp4_blackwell_tensor * dst) {
    float weight_scale = 1.0f;
    float input_scale  = 1.0f;
    const ggml_tensor * weight_scale_t = tensor->src[0];
    const ggml_tensor * input_scale_t  = tensor->src[1];
    if (weight_scale_t != nullptr && ggml_is_scalar(weight_scale_t) && weight_scale_t->type == GGML_TYPE_F32 &&
            weight_scale_t->data != nullptr && (weight_scale_t->buffer == nullptr || ggml_backend_buffer_is_host(weight_scale_t->buffer))) {
        memcpy(&weight_scale, weight_scale_t->data, sizeof(weight_scale));
    } else {
        memcpy(&weight_scale, &tensor->op_params[0], sizeof(weight_scale)); // CUDA-owned NVFP4 cached by llama-model
    }
    if (input_scale_t != nullptr && ggml_is_scalar(input_scale_t) && input_scale_t->type == GGML_TYPE_F32 &&
            input_scale_t->data != nullptr && (input_scale_t->buffer == nullptr || ggml_backend_buffer_is_host(input_scale_t->buffer))) {
        memcpy(&input_scale, input_scale_t->data, sizeof(input_scale));
    } else {
        memcpy(&input_scale, &tensor->op_params[1], sizeof(input_scale));
    }

    dst->weight_scale = weight_scale > 0.0f ? weight_scale : 1.0f;
    dst->input_scale  = input_scale  > 0.0f ? input_scale  : 1.0f;
    memset(dst->pad, 0, sizeof(dst->pad));
}

static inline void ggml_cuda_repack_tiles_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst) {
    GGML_ASSERT(ne0 % QK_NVFP4 == 0);

    const int64_t src_blocks_per_row = ggml_cuda_bw_div_up(ne0, QK_NVFP4);
    const int64_t dst_blocks_per_row = ggml_cuda_nvfp4_blocks_per_row(ne0);
    const int64_t tile_rows = ggml_cuda_bw_div_up(nrows, 16);
    const size_t src_row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);

    const uint8_t * src_bytes = (const uint8_t *) src;
    block_nvfp4_blackwell * dst_blocks = (block_nvfp4_blackwell *) dst;

    for (int64_t tile_row = 0; tile_row < tile_rows; ++tile_row) {
        const int64_t row0 = tile_row * 16;
        const int rows_in_tile = (int) ((row0 + 16 <= nrows) ? 16 : (nrows - row0));

        for (int64_t block_col = 0; block_col < dst_blocks_per_row; ++block_col) {
            const int64_t src_block0 = block_col * 4;
            const int frags_in_block = (int) ((src_block0 + 4 <= src_blocks_per_row) ? 4 : (src_blocks_per_row - src_block0));

            block_nvfp4_blackwell & out = dst_blocks[tile_row * dst_blocks_per_row + block_col];
            if (rows_in_tile != 16 || frags_in_block != 4) {
                memset(&out, 0, sizeof(out));
            }

            for (int row_in_tile = 0; row_in_tile < rows_in_tile; ++row_in_tile) {
                const int64_t row = row0 + row_in_tile;
                const block_nvfp4 * src_row = (const block_nvfp4 *) (src_bytes + row * src_row_size);
                const int lane_base = (row_in_tile & 7) * 4;
                const int row_half = row_in_tile >> 3;
                const int scale_lane = lane_base + row_half;

                for (int frag = 0; frag < frags_in_block; ++frag) {
                    const block_nvfp4 & in = src_row[src_block0 + frag];
                    block_nvfp4_blackwell_frag & tile = out.tiles[frag];

                    const uint8_t * p0 = in.qs +  0;
                    const uint8_t * p1 = in.qs +  8;
                    const uint8_t * p2 = in.qs + 16;
                    const uint8_t * p3 = in.qs + 24;
                    tile.regs[lane_base + 0][row_half + 0] = ggml_cuda_bw_pack8(p0, 0);
                    tile.regs[lane_base + 1][row_half + 0] = ggml_cuda_bw_pack8(p0, 4);
                    tile.regs[lane_base + 2][row_half + 0] = ggml_cuda_bw_pack8(p1, 0);
                    tile.regs[lane_base + 3][row_half + 0] = ggml_cuda_bw_pack8(p1, 4);
                    tile.regs[lane_base + 0][row_half + 2] = ggml_cuda_bw_pack8(p2, 0);
                    tile.regs[lane_base + 1][row_half + 2] = ggml_cuda_bw_pack8(p2, 4);
                    tile.regs[lane_base + 2][row_half + 2] = ggml_cuda_bw_pack8(p3, 0);
                    tile.regs[lane_base + 3][row_half + 2] = ggml_cuda_bw_pack8(p3, 4);

                    uint32_t d = 0;
                    memcpy(&d, in.d, sizeof(d));
                    tile.scales_u32[scale_lane + 0] = d;
                    tile.scales_u32[scale_lane + 2] = d;
                }
            }
        }
    }
}

static inline void ggml_cuda_repack_tensor_nvfp4(const ggml_tensor * tensor, const void * src, void * dst) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    const size_t src_plane_size = ggml_row_size(GGML_TYPE_NVFP4, ne0) * ne1;
    const size_t dst_plane_size = ggml_cuda_nvfp4_plane_size(ne0, ne1);
    block_nvfp4_blackwell_tensor * dst_tensor = (block_nvfp4_blackwell_tensor *) dst;

    ggml_cuda_nvfp4_set_tensor_header(tensor, dst_tensor);
    char * dst_tiles = (char *) dst_tensor->tiles;

    for (int64_t plane = 0; plane < nplanes; ++plane) {
        ggml_cuda_repack_tiles_nvfp4(ne0, ne1,
                (const char *) src + plane * src_plane_size,
                dst_tiles + plane * dst_plane_size);
    }
}

static inline void ggml_cuda_unpack_tiles_nvfp4(int64_t ne0, int64_t nrows, const void * src, void * dst) {
    GGML_ASSERT(ne0 % QK_NVFP4 == 0);

    const int64_t src_blocks_per_row = ggml_cuda_nvfp4_blocks_per_row(ne0);
    const int64_t dst_blocks_per_row = ggml_cuda_bw_div_up(ne0, QK_NVFP4);
    const size_t dst_row_size = ggml_row_size(GGML_TYPE_NVFP4, ne0);

    const block_nvfp4_blackwell * src_blocks = (const block_nvfp4_blackwell *) src;

    for (int64_t row = 0; row < nrows; ++row) {
        block_nvfp4 * dst_row = (block_nvfp4 *) ((uint8_t *) dst + row * dst_row_size);
        const int64_t tile_row = row / 16;
        const int row_in_tile = (int) (row % 16);
        const int lane_base = (row_in_tile & 7) * 4;
        const int row_half = row_in_tile >> 3;
        const int scale_lane = lane_base + row_half;

        for (int64_t block_col = 0; block_col < src_blocks_per_row; ++block_col) {
            const int64_t dst_block0 = block_col * 4;
            const int frags_in_block = (int) ((dst_block0 + 4 <= dst_blocks_per_row) ? 4 : (dst_blocks_per_row - dst_block0));
            const block_nvfp4_blackwell & in = src_blocks[tile_row * src_blocks_per_row + block_col];

            for (int frag = 0; frag < frags_in_block; ++frag) {
                const block_nvfp4_blackwell_frag & tile = in.tiles[frag];
                block_nvfp4 & out = dst_row[dst_block0 + frag];

                for (int g = 0; g < 4; ++g) {
                    const uint32_t lo = tile.regs[lane_base + 2*(g & 1) + 0][row_half + 2*(g >> 1)];
                    const uint32_t hi = tile.regs[lane_base + 2*(g & 1) + 1][row_half + 2*(g >> 1)];
                    uint8_t * p = out.qs + 8*g;
                    for (int i = 0; i < 8; ++i) {
                        p[i] = ((lo >> (4*i)) & 0x0F) | (((hi >> (4*i)) & 0x0F) << 4);
                    }
                }

                const uint32_t d = tile.scales_u32[scale_lane];
                memcpy(out.d, &d, sizeof(d));
            }
        }
    }
}

static inline void ggml_cuda_unpack_tensor_nvfp4(const ggml_tensor * tensor, const void * src, void * dst) {
    const int64_t ne0 = tensor->ne[0];
    const int64_t ne1 = tensor->ne[1];
    const int64_t nplanes = tensor->ne[2] * tensor->ne[3];
    const size_t src_plane_size = ggml_cuda_nvfp4_plane_size(ne0, ne1);
    const size_t dst_plane_size = ggml_row_size(GGML_TYPE_NVFP4, ne0) * ne1;
    const char * src_tiles = (const char *) ((const block_nvfp4_blackwell_tensor *) src)->tiles;

    for (int64_t plane = 0; plane < nplanes; ++plane) {
        ggml_cuda_unpack_tiles_nvfp4(ne0, ne1,
                src_tiles + plane * src_plane_size,
                (char *) dst + plane * dst_plane_size);
    }
}
