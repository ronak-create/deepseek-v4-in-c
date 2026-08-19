/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_matmul.c - matrix-vector products that consume the checkpoint's own
 * packed bytes, without ever widening a tensor.
 *
 * WHY NOT DEQUANTISE FIRST
 *   A Flash routed expert is 13,369,344 bytes packed. Widened to f32 it is
 *   8x that, a token touches six per layer across 43 layers, and none of them
 *   is resident. Dequantising into a scratch buffer would move ~26 GB per token
 *   through memory to save nothing. So the unpacking happens in the inner loop,
 *   where the byte is already in a register.
 *
 * THE REDUCTION ORDER IS PART OF THE CONTRACT
 *   Inherited from kimi-k3-in-c: accumulate in DOUBLE via fma(), in a fixed
 *   16-accumulator tree, so that a scalar build, an OpenMP build and (later) an
 *   AVX2 build produce BITWISE identical output. -ffp-contract=off in the
 *   Makefile stops the compiler fusing anything behind our back. The point is
 *   that a performance change can never quietly become an accuracy change.
 *
 *   No AVX2 path exists yet. When one is added it must reproduce this tree
 *   lane-for-lane and be gated on bit equality against the scalar path, exactly
 *   as K3 does. Adding a faster kernel that is merely *close* would forfeit the
 *   property this whole file is arranged around.
 *
 * WHERE THE SCALES GO
 *   inference/kernel.py fp4_gemm applies the weight scale to the ACCUMULATOR
 *   once per K block, not to each weight:
 *
 *       C_local_accum[i,j] += C_local[i,j] * scale_a[i] * scale_b[j]
 *
 *   with block_K = 32 = the weight group size. This file does the same: the dot
 *   product of one block is formed first, then multiplied by that block's scale,
 *   then added to the row total. Scaling each weight individually would be
 *   algebraically equal and numerically different, and would cost a multiply per
 *   element instead of per block.
 *
 *   Activations arrive already in f32 here, so there is no act scale: this is a
 *   dequantised-weight GEMV, not the FP8xFP4 GEMM the GPU path will want.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

#include "dsv4.h"
#include "dsv4_quant.h"

/* One block's dot product, in the same 16-accumulator tree K3 uses. `n` is the
 * block width, always a multiple of 16 for the real geometries (32 and 128), so
 * the tail loop only runs on the hand-built fixtures. */
#define BLOCK_DOT(LOAD)                                                        \
    do {                                                                       \
        double a[16] = {0};                                                    \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16)                                            \
            for (int l = 0; l < 16; l++)                                       \
                a[l] = fma(LOAD(i + l), (double)xb[i + l], a[l]);              \
        double b0 = (a[0] + a[4]) + (a[8]  + a[12]);                           \
        double b1 = (a[1] + a[5]) + (a[9]  + a[13]);                           \
        double b2 = (a[2] + a[6]) + (a[10] + a[14]);                           \
        double b3 = (a[3] + a[7]) + (a[11] + a[15]);                           \
        acc = (b0 + b1) + (b2 + b3);                                           \
        for (; i < n; i++) acc = fma(LOAD(i), (double)xb[i], acc);             \
    } while (0)

/* ------------------------------------------------------------------ bf16 --- */

static inline double bf16d(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return (double)v.f;
}

void dsv4_matmul_bf16(float *y, const float *x, const uint16_t *W,
                      int in, int out)
{
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const uint16_t *row = W + (size_t)o * in;
        const float *xb = x;
        const int n = in;
        double acc;
#define LOAD_BF16(k) bf16d(row[k])
        BLOCK_DOT(LOAD_BF16);
#undef LOAD_BF16
        y[o] = (float)acc;
    }
}

/* ------------------------------------------------------------------- fp8 ---
 * W is [out, in] E4M3 bytes; S is the E8M0 grid, ceil(out/128) x ceil(in/128).
 * The scale changes every blk_c columns along the row. */
void dsv4_matmul_fp8(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    const int sc = (in + blk_c - 1) / blk_c;   /* scale columns per row */
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        const uint8_t *row = W + (size_t)o * in;
        double total = 0.0;
        for (int c0 = 0; c0 < in; c0 += blk_c) {
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;
            const uint8_t *wb = row + c0;
            const float   *xb = x + c0;
            double acc;
#define LOAD_FP8(k) ((double)dsv4_e4m3_to_f32(wb[k]))
            BLOCK_DOT(LOAD_FP8);
#undef LOAD_FP8
            /* One multiply per block, as kernel.py does -- not per element. */
            total += acc * (double)dsv4_e8m0_to_f32(S[(o / blk_r) * (size_t)sc
                                                      + (c0 / blk_c)]);
        }
        y[o] = (float)total;
    }
}

/* ------------------------------------------------------------------- fp4 ---
 * W is [out, in/2] BYTES holding two E2M1 values each, packed along in (the
 * reduce dimension). S is the E8M0 grid, out x (in/32).
 *
 * NOTE the element index into a packed row is the LOGICAL column, and
 * dsv4_fp4_at does the byte/nibble split. Whether the even element is the low
 * or the high nibble is the one unverified thing in this path -- see
 * DSV4_FP4_LOW_NIBBLE_FIRST in dsv4_quant.h. */
void dsv4_matmul_fp4(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    const int sc = (in + blk_c - 1) / blk_c;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (out > 64)
#endif
    for (int o = 0; o < out; o++) {
        /* Stored row is in/2 bytes wide. Getting this stride wrong reads half a
         * row of the wrong values and still produces finite numbers. */
        const uint8_t *row = W + (size_t)o * ((size_t)in / 2);
        double total = 0.0;
        for (int c0 = 0; c0 < in; c0 += blk_c) {
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;
            const float *xb = x + c0;
            double acc;
#define LOAD_FP4(k) ((double)dsv4_fp4_at(row, (int64_t)c0 + (k)))
            BLOCK_DOT(LOAD_FP4);
#undef LOAD_FP4
            total += acc * (double)dsv4_e8m0_to_f32(S[(o / blk_r) * (size_t)sc
                                                      + (c0 / blk_c)]);
        }
        y[o] = (float)total;
    }
}

/* ---------------------------------------------------------------- dispatch --
 * The one call every weight matrix goes through. The branch is on a per-matrix
 * tag decided at bind time, outside the inner loops, so it costs nothing
 * measurable. A matrix carries its own block geometry precisely so that no
 * caller has to remember whether this one is 128x128 or 1x32. */
void dsv4_mmq(float *y, const float *x, const DSV4QMat *m)
{
    switch (m->wdt) {
    case DSV4_WBF16:
        dsv4_matmul_bf16(y, x, (const uint16_t *)m->w, m->cols, m->rows);
        break;
    case DSV4_WFP8:
        dsv4_matmul_fp8(y, x, (const uint8_t *)m->w, (const uint8_t *)m->s,
                        m->cols, m->rows, m->blk_r, m->blk_c);
        break;
    case DSV4_WFP4:
        dsv4_matmul_fp4(y, x, (const uint8_t *)m->w, (const uint8_t *)m->s,
                        m->cols, m->rows, m->blk_r, m->blk_c);
        break;
    default:
        /* Reaching here means a matrix was bound with a tag nothing implements.
         * Producing zeros would be a running model with a missing projection. */
        fprintf(stderr, "dsv4_mmq: matrix has unimplemented wdt %d "
                        "(%dx%d)\n", m->wdt, m->rows, m->cols);
        abort();
    }
}
