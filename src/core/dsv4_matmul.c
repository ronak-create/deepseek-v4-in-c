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
#include <string.h>

#include "dsv4.h"
#include "dsv4_quant.h"
#include "dsv4_cuda.h"

/* One block's dot product, in the same 16-accumulator tree K3 uses. `n` is the
 * block width, always a multiple of 16 for the real geometries (32 and 128), so
 * the tail loop only runs on the hand-built fixtures. */
#define BLOCK_DOT_S(LOAD)                                                      \
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

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>

/* THE SAME TREE, FOUR LANES AT A TIME -- and it is the same tree, not merely a
 * similar one. Put a[0..3] in v0, a[4..7] in v1, a[8..11] in v2, a[12..15] in
 * v3, and lane j of each vector holds a[j], a[4+j], a[8+j], a[12+j]. Then
 *
 *     b[j] = (a[j] + a[4+j]) + (a[8+j] + a[12+j])
 *
 * is exactly (v0 + v1) + (v2 + v3), elementwise, with the same grouping. The
 * final (b0 + b1) + (b2 + b3) is then done on the four extracted lanes in the
 * scalar order. _mm256_fmadd_pd is the same IEEE fused multiply-add as fma(),
 * and float -> double is exact, so every intermediate is bit-for-bit what the
 * scalar path computes. That is a claim the matmul gate checks at runtime
 * rather than a claim this comment makes.
 *
 * LOAD4(i) must return the four weights at logical columns i..i+3 as doubles. */
#define XB4(i) _mm256_loadu_pd(xdb + (i))
#define BLOCK_DOT_V(LOAD1, LOAD4)                                              \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            v0 = _mm256_fmadd_pd(LOAD4(i),      XB4(i),      v0);              \
            v1 = _mm256_fmadd_pd(LOAD4(i + 4),  XB4(i + 4),  v1);              \
            v2 = _mm256_fmadd_pd(LOAD4(i + 8),  XB4(i + 8),  v2);              \
            v3 = _mm256_fmadd_pd(LOAD4(i + 12), XB4(i + 12), v3);              \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma(LOAD1(i), (double)xb[i], acc);            \
    } while (0)
#endif

/* --------------------------------------------------------------- loaders ---
 * One per storage format, in two widths. The scalar width is the definition;
 * the wider one must agree with it element for element, which the matmul gate
 * checks rather than assumes. */

static inline double bf16d(uint16_t h)
{
    union { uint32_t u; float f; } v;
    v.u = (uint32_t)h << 16;
    return (double)v.f;
}

#define LOAD1_BF16(k) bf16d(row[k])
#define LOAD1_FP8(k)  ((double)dsv4_e4m3_to_f32(wb[k]))
#define LOAD1_FP4(k)  ((double)dsv4_fp4_at(row, (int64_t)c0 + (k)))

#if defined(__AVX2__) && defined(__FMA__)
/* BF16 widens without a table: the stored halfword IS the top half of the f32,
 * so shifting it up 16 bits and reinterpreting is the whole conversion. */
static inline __m256d load4_bf16(const uint16_t *row, int i)
{
    const __m128i h = _mm_loadl_epi64((const __m128i *)(row + i));
    const __m128i w = _mm_slli_epi32(_mm_cvtepu16_epi32(h), 16);
    return _mm256_cvtps_pd(_mm_castsi128_ps(w));
}

/* FP8 keeps its scalar table lookups: 256 possible values, and a gather would
 * cost more than four L1 hits. _mm_set_ps, NOT a stack array reloaded with
 * _mm_loadu_ps -- four scalar stores followed by a 128-bit load is a
 * store-to-load forwarding stall, and it cost more than the vector arithmetic
 * saved. The first version of this file measured the expert matmuls at 41.8 s
 * against the scalar path's 27.7 s for exactly that reason. */
static inline __m256d load4_fp8(const uint8_t *wb, int i)
{
    return _mm256_cvtps_pd(_mm_set_ps(dsv4_e4m3_to_f32(wb[i + 3]),
                                      dsv4_e4m3_to_f32(wb[i + 2]),
                                      dsv4_e4m3_to_f32(wb[i + 1]),
                                      dsv4_e4m3_to_f32(wb[i + 0])));
}

static inline __m256d load4_fp4(const uint8_t *row, int64_t base, int i)
{
    return _mm256_cvtps_pd(_mm_set_ps(dsv4_fp4_at(row, base + i + 3),
                                      dsv4_fp4_at(row, base + i + 2),
                                      dsv4_fp4_at(row, base + i + 1),
                                      dsv4_fp4_at(row, base + i + 0)));
}

/* EIGHT FP4 VALUES AT ONCE, without eight table lookups.
 *
 * e2m1 decomposes exactly: code & 7 selects a magnitude from
 * {0, .5, 1, 1.5, 2, 3, 4, 6} and bit 3 is the sign. So a magnitude is one
 * cross-lane permute over an 8-entry table, and the sign is that bit shifted
 * into the float's sign position and OR-ed in. That reproduces dsv4_e2m1_lut
 * entry for entry, including -0.0 for code 8, which is why it can replace the
 * lookup rather than approximate it.
 *
 * Four packed bytes carry the eight nibbles. Duplicate each byte, widen to
 * eight 32-bit lanes, then shift the odd lanes down by four so lane k holds the
 * nibble of logical element base+k.
 *
 * Measured honestly: this is worth almost nothing on its own (81.5 GF/s against
 * 80.9 for the four-lookup form). FP4 was never dequant-bound. It is kept
 * because it is not slower and it removes the table from the hot loop, but the
 * gain in this file came from elsewhere -- see widen_x. */
static inline __m256 fp4_8(const uint8_t *row, int64_t base)
{
    uint32_t four;
    memcpy(&four, row + (base >> 1), 4);
    __m128i b = _mm_cvtsi32_si128((int)four);
    b = _mm_shuffle_epi8(b, _mm_setr_epi8(0, 0, 1, 1, 2, 2, 3, 3,
                                          -1, -1, -1, -1, -1, -1, -1, -1));
    __m256i idx = _mm256_cvtepu8_epi32(b);
#if DSV4_FP4_LOW_NIBBLE_FIRST
    const __m256i sh = _mm256_setr_epi32(0, 4, 0, 4, 0, 4, 0, 4);
#else
    const __m256i sh = _mm256_setr_epi32(4, 0, 4, 0, 4, 0, 4, 0);
#endif
    idx = _mm256_and_si256(_mm256_srlv_epi32(idx, sh), _mm256_set1_epi32(0x0F));
    const __m256 mag = _mm256_permutevar8x32_ps(
        _mm256_setr_ps(0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f),
        _mm256_and_si256(idx, _mm256_set1_epi32(7)));
    const __m256i sgn = _mm256_slli_epi32(
        _mm256_and_si256(idx, _mm256_set1_epi32(8)), 28);
    return _mm256_castsi256_ps(
        _mm256_or_si256(_mm256_castps_si256(mag), sgn));
}

#define PD_LO(f) _mm256_cvtps_pd(_mm256_castps256_ps128(f))
#define PD_HI(f) _mm256_cvtps_pd(_mm256_extractf128_ps((f), 1))

/* Same tree, same grouping, same order -- only the dequant changes. */
#define BLOCK_DOT_V_FP4(LOAD1, LOAD4)                                          \
    do {                                                                       \
        __m256d v0 = _mm256_setzero_pd(), v1 = _mm256_setzero_pd();            \
        __m256d v2 = _mm256_setzero_pd(), v3 = _mm256_setzero_pd();            \
        int i = 0;                                                             \
        for (; i + 15 < n; i += 16) {                                          \
            const __m256 f0 = fp4_8(row, (int64_t)c0 + i);                     \
            const __m256 f1 = fp4_8(row, (int64_t)c0 + i + 8);                 \
            v0 = _mm256_fmadd_pd(PD_LO(f0), XB4(i),      v0);                  \
            v1 = _mm256_fmadd_pd(PD_HI(f0), XB4(i + 4),  v1);                  \
            v2 = _mm256_fmadd_pd(PD_LO(f1), XB4(i + 8),  v2);                  \
            v3 = _mm256_fmadd_pd(PD_HI(f1), XB4(i + 12), v3);                  \
        }                                                                      \
        double bb[4];                                                          \
        _mm256_storeu_pd(bb, _mm256_add_pd(_mm256_add_pd(v0, v1),              \
                                           _mm256_add_pd(v2, v3)));            \
        acc = (bb[0] + bb[1]) + (bb[2] + bb[3]);                               \
        for (; i < n; i++) acc = fma(LOAD1(i), (double)xb[i], acc);            \
    } while (0)

#define LOAD4_BF16(k) load4_bf16(row, (k))
#define LOAD4_FP8(k)  load4_fp8(wb, (k))
#define LOAD4_FP4(k)  load4_fp4(row, (int64_t)c0, (k))
#else
/* Never expanded without AVX2, but they must still parse. */
#define LOAD4_BF16(k) 0
#define LOAD4_FP8(k)  0
#define LOAD4_FP4(k)  0
#define BLOCK_DOT_V_FP4(LOAD1, LOAD4) BLOCK_DOT_S(LOAD1)
#endif

/* ONE f32 -> f64 PASS OVER THE ACTIVATIONS, reused by every output row.
 *
 * x is the same vector for all `out` rows, so widening it inside the row loop
 * did the work `out` times over: a 2048x4096 expert matrix converted 2.1M
 * values where 1024 suffice. (double)float is exact, so hoisting it changes the
 * count and nothing else -- the gate still reports bit equality.
 *
 * malloc rather than a fixed array: Pro's widest reduce dimension is 16384, and
 * a kernel that is correct only up to some assumed maximum is precisely the
 * kind of thing this engine keeps finding in itself. */
static double *widen_x(const float *x, int in)
{
    double *d = (double *)malloc((size_t)in * sizeof *d);
    if (!d) {
        fprintf(stderr, "dsv4_matmul: could not widen %d activations\n", in);
        return NULL;
    }
    for (int i = 0; i < in; i++) d[i] = (double)x[i];
    return d;
}

/* The scalar tree, called with the vector signature so one body serves both. */
#define BLOCK_DOT_S2(LOAD1, LOAD4) BLOCK_DOT_S(LOAD1)

/* ---------------------------------------------------------------- kernels ---
 * Emitted twice: once always, as `..._scalar`, and once as `..._avx2` where the
 * machine has AVX2+FMA. Two instantiations of ONE body, so they cannot drift
 * apart, and both are callable at runtime so the gate can compare them on real
 * data instead of trusting the argument that they must agree. */
#define GEN_MATMULS(SUF, DOT, DOT4)                                            \
                                                                               \
void dsv4_matmul_bf16##SUF(float *y, const float *x, const uint16_t *W,        \
                           int in, int out)                                    \
{                                                                              \
    double *xd = widen_x(x, in);                                               \
    if (!xd) return;                                                           \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                 \
    for (int o = 0; o < out; o++) {                                            \
        const uint16_t *row = W + (size_t)o * in;                              \
        const float  *xb  = x;                                                 \
        const double *xdb = xd; (void)xdb;                                     \
        const int n = in;                                                      \
        double acc;                                                            \
        DOT(LOAD1_BF16, LOAD4_BF16);                                           \
        y[o] = (float)acc;                                                     \
    }                                                                          \
    free(xd);                                                                  \
}                                                                              \
                                                                               \
void dsv4_matmul_fp8##SUF(float *y, const float *x, const uint8_t *W,          \
                          const uint8_t *S, int in, int out,                   \
                          int blk_r, int blk_c)                                \
{                                                                              \
    const int sc = (in + blk_c - 1) / blk_c;                                   \
    double *xd = widen_x(x, in);                                               \
    if (!xd) return;                                                           \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                 \
    for (int o = 0; o < out; o++) {                                            \
        const uint8_t *rw = W + (size_t)o * in;                                \
        double total = 0.0;                                                    \
        for (int c0 = 0; c0 < in; c0 += blk_c) {                               \
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;               \
            const uint8_t *wb  = rw + c0;                                      \
            const float   *xb  = x + c0;                                       \
            const double  *xdb = xd + c0; (void)xdb;                           \
            double acc;                                                        \
            DOT(LOAD1_FP8, LOAD4_FP8);                                         \
            /* One multiply per block, as kernel.py does -- not per element. */\
            total += acc * (double)dsv4_e8m0_to_f32(                           \
                        S[(o / blk_r) * (size_t)sc + (c0 / blk_c)]);           \
        }                                                                      \
        y[o] = (float)total;                                                   \
    }                                                                          \
    free(xd);                                                                  \
}                                                                              \
                                                                               \
void dsv4_matmul_fp4##SUF(float *y, const float *x, const uint8_t *W,          \
                          const uint8_t *S, int in, int out,                   \
                          int blk_r, int blk_c)                                \
{                                                                              \
    const int sc = (in + blk_c - 1) / blk_c;                                   \
    double *xd = widen_x(x, in);                                               \
    if (!xd) return;                                                           \
    _Pragma("omp parallel for schedule(static) if (out > 64)")                 \
    for (int o = 0; o < out; o++) {                                            \
        /* Stored row is in/2 bytes wide. Getting this stride wrong reads half  \
         * a row of the wrong values and still produces finite numbers. */     \
        const uint8_t *row = W + (size_t)o * ((size_t)in / 2);                 \
        double total = 0.0;                                                    \
        for (int c0 = 0; c0 < in; c0 += blk_c) {                               \
            const int n = (in - c0 < blk_c) ? (in - c0) : blk_c;               \
            const float  *xb  = x + c0;                                        \
            const double *xdb = xd + c0; (void)xdb;                            \
            double acc;                                                        \
            DOT4(LOAD1_FP4, LOAD4_FP4);                                        \
            total += acc * (double)dsv4_e8m0_to_f32(                           \
                        S[(o / blk_r) * (size_t)sc + (c0 / blk_c)]);           \
        }                                                                      \
        y[o] = (float)total;                                                   \
    }                                                                          \
    free(xd);                                                                  \
}

GEN_MATMULS(_scalar, BLOCK_DOT_S2, BLOCK_DOT_S2)

#if defined(__AVX2__) && defined(__FMA__)
GEN_MATMULS(_avx2, BLOCK_DOT_V, BLOCK_DOT_V_FP4)
#define PICK(name) name##_avx2
int dsv4_matmul_has_avx2(void) { return 1; }
#else
#define PICK(name) name##_scalar
int dsv4_matmul_has_avx2(void) { return 0; }
#endif

/* The names the rest of the engine calls. */
void dsv4_matmul_bf16(float *y, const float *x, const uint16_t *W,
                      int in, int out)
{
    PICK(dsv4_matmul_bf16)(y, x, W, in, out);
}

void dsv4_matmul_fp8(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    PICK(dsv4_matmul_fp8)(y, x, W, S, in, out, blk_r, blk_c);
}

void dsv4_matmul_fp4(float *y, const float *x, const uint8_t *W,
                     const uint8_t *S, int in, int out, int blk_r, int blk_c)
{
    PICK(dsv4_matmul_fp4)(y, x, W, S, in, out, blk_r, blk_c);
}

/* ---------------------------------------------------------------- dispatch --
 * The one call every weight matrix goes through. The branch is on a per-matrix
 * tag decided at bind time, outside the inner loops, so it costs nothing
 * measurable. A matrix carries its own block geometry precisely so that no
 * caller has to remember whether this one is 128x128 or 1x32. */
void dsv4_mmq(float *y, const float *x, const DSV4QMat *m)
{
    /* If this matrix is resident on the device, it runs there. The check is a
     * pointer lookup over a few hundred entries, against a matmul that moves
     * tens of MB -- and in a build without CUDA it is a call that returns 0.
     *
     * Note what is NOT here: any attempt to keep the GPU bit-identical. The
     * device is opt-in and gated separately, on relative error and argmax
     * identity; see dsv4_cuda.h. */
    if (dsv4_cuda_has(m)) { dsv4_cuda_mmq(y, x, m); return; }

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
