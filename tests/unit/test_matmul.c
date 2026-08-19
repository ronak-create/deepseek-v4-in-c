/* SPDX-License-Identifier: Apache-2.0 */
/* test_matmul.c - the packed matmuls, against exactly-representable references.
 *
 * HOW THE EXPECTATIONS ARE BUILT
 *   Every fixture here is chosen so the true dot product is EXACTLY
 *   representable in f32: the weights come from the e2m1/e4m3 tables, the scales
 *   are powers of two, and the activations are small integers or halves. That
 *   lets the check be bit equality rather than a tolerance.
 *
 *   A tolerance would be the wrong instrument. The failures this file is looking
 *   for -- a wrong block stride, a scale applied per element instead of per
 *   block, a packed row read at full width -- all produce answers that are
 *   plausibly close. Only exactness separates them.
 *
 * WHAT IS DELIBERATELY NOT ASSERTED
 *   Agreement with PyTorch. That needs torch and the real checkpoint, and it is
 *   the Phase 3 oracle's job. These gates establish internal correctness and the
 *   invariants the reduction order is supposed to have.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#ifdef _OPENMP
#include <omp.h>
#endif
#include "dsv4.h"
#include "dsv4_quant.h"

void dsv4_matmul_bf16(float *, const float *, const uint16_t *, int, int);
void dsv4_matmul_fp8(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_matmul_fp4(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_mmq(float *, const float *, const DSV4QMat *);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

static void eqf(const char *what, float got, float want)
{
    CHECK(memcmp(&got, &want, sizeof got) == 0,
          "%s = %.9g, expected %.9g", what, (double)got, (double)want);
}

int main(void)
{
    printf("DeepSeek-V4 packed matmul gate\n");

    /* ---------------------------------------------------------------- fp8 -- */
    printf("\n-- GATE 1  FP8, one 128-wide block, scale applied once --\n");
    {
        enum { IN = 128, OUT = 2 };
        static uint8_t W[OUT * IN];
        static uint8_t S[OUT * 1];
        static float x[IN], y[OUT];

        /* row 0: all 1.0 (0x38).  row 1: all 2.0 (0x40). */
        memset(W,          0x38, IN);
        memset(W + IN,     0x40, IN);
        /* THE GRID IS ceil(OUT/blk_r) x ceil(IN/blk_c) = 1x1 HERE. With
         * blk_r = 128 and only two rows, BOTH rows share S[0]. An earlier draft
         * of this gate set S[0] and S[1] and expected them to apply per row,
         * which is not what a 128-row block means. The row dimension is
         * exercised properly below, with enough rows to have two row-blocks. */
        S[0] = 128;          /* 2^1 = 2, shared by both rows */
        S[1] = 0;            /* must be IGNORED */
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp8(y, x, W, S, IN, OUT, 128, 128);
        /* row 0: 128 * 1.0 * 2 = 256 ; row 1: 128 * 2.0 * 2 = 512 */
        eqf("fp8 row0", y[0], 256.0f);
        eqf("fp8 row1", y[1], 512.0f);
        printf("  ok    256 and 512, both rows sharing one 128-row block scale\n");
    }

    printf("\n-- GATE 1b  the ROW dimension of the scale grid --\n");
    {
        /* 256 rows = two row-blocks at blk_r 128, so S has two entries and the
         * second half of the output must pick up the second one. */
        enum { IN = 128, OUT = 256 };
        static uint8_t W[OUT * IN];
        static uint8_t S[2];
        static float x[IN], y[OUT];
        memset(W, 0x38, sizeof W);      /* every weight 1.0 */
        S[0] = 127;                     /* rows   0..127 -> scale 1 */
        S[1] = 129;                     /* rows 128..255 -> scale 4 */
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp8(y, x, W, S, IN, OUT, 128, 128);
        eqf("fp8 row 0",   y[0],   128.0f);
        eqf("fp8 row 127", y[127], 128.0f);
        eqf("fp8 row 128", y[128], 512.0f);
        eqf("fp8 row 255", y[255], 512.0f);
        CHECK(y[127] != y[128], "the row-block boundary had no effect");
        printf("  ok    128 below the boundary, 512 above it\n");
    }

    printf("\n-- GATE 2  FP8, the scale must change ACROSS blocks --\n");
    {
        /* Two blocks in one row with different scales. A kernel that reads only
         * the first block's scale, or indexes the grid by element rather than by
         * block, gets 256 here instead of 384 -- finite, and wrong. */
        enum { IN = 256, OUT = 1 };
        static uint8_t W[IN];
        static uint8_t S[2];
        static float x[IN], y[OUT];
        memset(W, 0x38, IN);            /* all 1.0 */
        S[0] = 127;                     /* block 0 scale 1 */
        S[1] = 128;                     /* block 1 scale 2 */
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp8(y, x, W, S, IN, OUT, 128, 128);
        /* 128*1*1 + 128*1*2 = 384 */
        eqf("fp8 two blocks", y[0], 384.0f);
        CHECK(y[0] != 256.0f, "y equals the single-scale answer; scales not per block");
        printf("  ok    384 (not 256), so scales advance per block\n");
    }

    /* ---------------------------------------------------------------- fp4 -- */
    printf("\n-- GATE 3  FP4, packed two per byte, 1x32 blocks --\n");
    {
        enum { IN = 64, OUT = 2 };           /* two 32-wide blocks per row */
        static uint8_t W[OUT * (IN / 2)];
        static uint8_t S[OUT * 2];
        static float x[IN], y[OUT];

        /* Every nibble = code 2 = 1.0, so both nibble orders agree here and this
         * gate is independent of DSV4_FP4_LOW_NIBBLE_FIRST. */
        memset(W, 0x22, sizeof W);
        S[0] = 127; S[1] = 128;              /* row 0: scales 1, 2 */
        S[2] = 127; S[3] = 127;              /* row 1: scales 1, 1 */
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp4(y, x, W, S, IN, OUT, 1, 32);
        /* row 0: 32*1*1 + 32*1*2 = 96 ; row 1: 32 + 32 = 64 */
        eqf("fp4 row0", y[0], 96.0f);
        eqf("fp4 row1", y[1], 64.0f);
        printf("  ok    96 and 64, with per-row and per-block scales\n");
    }

    printf("\n-- GATE 4  FP4 row stride is in/2 BYTES, not in --\n");
    {
        /* Row 1 is deliberately different from row 0. A kernel that strides by
         * `in` rather than `in/2` reads past row 0 into row 1's territory and
         * still returns a finite number. */
        enum { IN = 32, OUT = 2 };
        static uint8_t W[OUT * (IN / 2)];
        static uint8_t S[OUT];
        static float x[IN], y[OUT];
        memset(W,              0x22, IN / 2);   /* row 0: all 1.0 */
        memset(W + (IN / 2),   0x44, IN / 2);   /* row 1: all 2.0 (code 4) */
        S[0] = 127; S[1] = 127;
        for (int i = 0; i < IN; i++) x[i] = 1.0f;

        dsv4_matmul_fp4(y, x, W, S, IN, OUT, 1, 32);
        eqf("fp4 stride row0", y[0], 32.0f);    /* 32 * 1.0 */
        eqf("fp4 stride row1", y[1], 64.0f);    /* 32 * 2.0 */
        CHECK(y[0] != y[1], "both rows read the same bytes; stride is wrong");
        printf("  ok    rows are distinct: 32 and 64\n");
    }

    printf("\n-- GATE 5  signed values and a non-trivial activation --\n");
    {
        enum { IN = 32, OUT = 1 };
        static uint8_t W[IN / 2];
        static uint8_t S[1];
        static float x[IN], y[OUT];
        /* alternate code 2 (+1.0) and code 10 (-1.0) */
        memset(W, 0xA2, sizeof W);
        S[0] = 128;                              /* scale 2 */
        for (int i = 0; i < IN; i++) x[i] = (float)(i + 1);

        dsv4_matmul_fp4(y, x, W, S, IN, OUT, 1, 32);
        /* Whichever nibble is first, the row is 16 of (+1) and 16 of (-1)
         * against x = 1..32. If +1 lands on the odd positions the sum is +16;
         * if on the even ones it is -16. Both are valid until the order is
         * verified, so assert the magnitude, which is order-independent. */
        CHECK(fabsf(y[0]) == 32.0f, "|y| = %.9g, expected 32 (16 pairs x 1 x scale 2)",
              (double)fabsf(y[0]));
        printf("  ok    |y| = %.0f  (sign depends on the unverified nibble order)\n",
               (double)fabsf(y[0]));
    }

    printf("\n-- GATE 6  bf16, and dispatch through dsv4_mmq --\n");
    {
        enum { IN = 32, OUT = 2 };
        static uint16_t W[OUT * IN];
        static float x[IN], y[OUT], y2[OUT];
        for (int o = 0; o < OUT; o++)
            for (int i = 0; i < IN; i++)
                W[o * IN + i] = (o == 0) ? 0x3F80 : 0x4000;   /* 1.0 : 2.0 */
        for (int i = 0; i < IN; i++) x[i] = 0.5f;

        dsv4_matmul_bf16(y, x, W, IN, OUT);
        eqf("bf16 row0", y[0], 16.0f);
        eqf("bf16 row1", y[1], 32.0f);

        DSV4QMat m; memset(&m, 0, sizeof m);
        m.wdt = DSV4_WBF16; m.w = W; m.rows = OUT; m.cols = IN;
        dsv4_mmq(y2, x, &m);
        CHECK(memcmp(y, y2, sizeof y) == 0, "dsv4_mmq disagrees with the direct call");
        printf("  ok    16 and 32, dispatch matches the direct call\n");
    }

    printf("\n-- GATE 7  the result must not depend on thread count --\n");
    {
        /* The reduction order is fixed per output row and rows are independent,
         * so OpenMP scheduling cannot change a single bit. If this ever fails,
         * a kernel has started accumulating ACROSS rows. */
        enum { IN = 256, OUT = 96 };
        static uint8_t W[OUT * IN];
        static uint8_t S[OUT * 2];
        static float x[IN], a[OUT], b[OUT];
        unsigned st = 12345u;
        for (int i = 0; i < OUT * IN; i++) { st = st * 1103515245u + 12345u;
                                             W[i] = (uint8_t)((st >> 16) & 0x7E); }
        for (int i = 0; i < OUT * 2; i++)  { S[i] = (uint8_t)(120 + (i % 9)); }
        for (int i = 0; i < IN; i++)       { st = st * 1103515245u + 12345u;
                                             x[i] = (float)((int)((st >> 20) & 0xFF) - 128) * 0.01f; }
#ifdef _OPENMP
        omp_set_num_threads(1);
#endif
        dsv4_matmul_fp8(a, x, W, S, IN, OUT, 128, 128);
#ifdef _OPENMP
        omp_set_num_threads(8);
#endif
        dsv4_matmul_fp8(b, x, W, S, IN, OUT, 128, 128);
        CHECK(memcmp(a, b, sizeof a) == 0,
              "1-thread and 8-thread results differ; the reduction is not fixed");
        int finite = 0;
        for (int i = 0; i < OUT; i++) if (isfinite(a[i])) finite++;
        CHECK(finite == OUT, "%d of %d outputs are finite", finite, OUT);
        printf("  ok    bit-identical across 1 and 8 threads, all finite\n");
    }

    printf("\n");
    if (fails) { printf("MATMUL GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("MATMUL GATE PASSED\n");
    return 0;
}
