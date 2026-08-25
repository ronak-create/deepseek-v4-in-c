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

/* Both instantiations, so the gate can run them against each other. */
void dsv4_matmul_bf16_scalar(float *, const float *, const uint16_t *, int, int);
void dsv4_matmul_fp8_scalar(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);
void dsv4_mmq_n(float *, const float *, const DSV4QMat *, int);
void dsv4_matmul_fp4_scalar(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);
int  dsv4_matmul_has_avx2(void);

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
    printf("\n-- GATE  the AVX2 path is BITWISE the scalar path --\n");
    {
        /* The whole file is arranged around one property: a faster kernel must
         * not be a different kernel. So this compares exact bit patterns, not
         * a tolerance. A tolerance here would defeat the purpose -- "close" is
         * precisely the failure mode -ffp-contract=off exists to prevent.
         *
         * Sizes are deliberately not multiples of the vector width in the
         * output dimension, and the FP8 block width 128 and FP4 block width 32
         * both exercise the 16-wide main loop plus the scalar tail. */
        enum { IN = 256, OUT = 37 };
        static float x[IN], ys[OUT], yv[OUT];
        static uint8_t W8[(size_t)OUT * IN], W4[(size_t)OUT * IN / 2];
        static uint16_t Wb[(size_t)OUT * IN];
        static uint8_t S8[OUT * 2], S4[(size_t)OUT * (IN / 32)];

        unsigned r = 12345u;
        for (int i = 0; i < IN; i++) {
            r = r * 1103515245u + 12345u;
            x[i] = (float)((int)((r >> 16) & 0xffff) - 32768) * 1e-3f;
        }
        /* e4m3fn: 0x7F and 0xFF are the only NaNs. Excluded on purpose --
         * a NaN makes both paths produce a NaN whose payload need not have the
         * same bits, so the comparison would fail on the data rather than on
         * the arithmetic. That is exactly what the first run of this gate did,
         * and fp4 passed alongside it because all 16 e2m1 codes are finite. */
        for (size_t i = 0; i < sizeof W8; i++) {
            r = r * 1103515245u + 12345u;
            uint8_t b = (uint8_t)(r >> 16);
            if ((b & 0x7Fu) == 0x7Fu) b &= 0xFEu;
            W8[i] = b;
        }
        for (size_t i = 0; i < sizeof W4; i++) { r = r * 1103515245u + 12345u;
                                                 W4[i] = (uint8_t)(r >> 16); }
        /* bf16 is the top half of an f32, so build it from a finite float
         * rather than from random bits, which would land on NaN and inf. */
        for (size_t i = 0; i < (size_t)OUT * IN; i++) {
            r = r * 1103515245u + 12345u;
            union { uint32_t u; float f; } v;
            v.f = (float)((int)((r >> 16) & 0xffff) - 32768) * 1e-3f;
            Wb[i] = (uint16_t)(v.u >> 16);
        }
        /* Keep scales in a sane exponent range so nothing overflows to inf --
         * an inf == inf comparison would pass while hiding a real difference. */
        for (size_t i = 0; i < sizeof S8; i++) S8[i] = 127;
        for (size_t i = 0; i < sizeof S4; i++) S4[i] = 127;

        printf("  build has AVX2+FMA: %s\n",
               dsv4_matmul_has_avx2() ? "yes" : "no (paths are the same code)");

        dsv4_matmul_bf16_scalar(ys, x, Wb, IN, OUT);
        dsv4_matmul_bf16(yv, x, Wb, IN, OUT);
        CHECK(memcmp(ys, yv, sizeof ys) == 0, "bf16 differs between paths");

        dsv4_matmul_fp8_scalar(ys, x, W8, S8, IN, OUT, 128, 128);
        dsv4_matmul_fp8(yv, x, W8, S8, IN, OUT, 128, 128);
        CHECK(memcmp(ys, yv, sizeof ys) == 0, "fp8 differs between paths");

        dsv4_matmul_fp4_scalar(ys, x, W4, S4, IN, OUT, 1, 32);
        dsv4_matmul_fp4(yv, x, W4, S4, IN, OUT, 1, 32);
        CHECK(memcmp(ys, yv, sizeof ys) == 0, "fp4 differs between paths");

        if (!fails)
            printf("  ok    bf16, fp8 and fp4 identical to the last bit over "
                   "%d x %d\n", OUT, IN);
    }


    printf("\n-- GATE  the batched GEMM equals nt separate GEMVs, bit for bit --\n");
    {
        /* This is the property the whole prefill change rests on. dsv4_mmq_n
         * reorders WHEN each dot product is issued so that one loaded weight
         * block serves every token in the batch -- and reordering independent
         * work is only safe if the work really is independent. If it is not,
         * the failure mode is a prompt that produces slightly different logits
         * and therefore, eventually, a different token: fluent, plausible, and
         * wrong. So this compares bit patterns.
         *
         * Every weight format, because they fold the per-block scale
         * differently and the batched kernels hoist that fold out of the token
         * loop. A batch size that is not a multiple of anything, so no tail is
         * skipped. Activations that differ per token, so a kernel that
         * broadcast token 0 across the batch could not pass. */
        enum { IN = 256, OUT = 37, NT = 5 };
        static float xb[NT * IN], ybatch[NT * OUT], yone[OUT];
        static uint8_t W8[(size_t)OUT * IN], W4[(size_t)OUT * IN / 2];
        static uint16_t Wb[(size_t)OUT * IN];
        static uint8_t S8[OUT * 2], S4[(size_t)OUT * (IN / 32)];

        unsigned r = 777u;
        for (int i = 0; i < NT * IN; i++) {
            r = r * 1103515245u + 12345u;
            xb[i] = (float)((int)((r >> 16) & 0xffff) - 32768) * 1e-3f;
        }
        for (size_t i = 0; i < sizeof W8; i++) {
            r = r * 1103515245u + 12345u;
            uint8_t b = (uint8_t)(r >> 16);
            if ((b & 0x7Fu) == 0x7Fu) b &= 0xFEu;      /* never a NaN weight */
            W8[i] = b;
        }
        for (size_t i = 0; i < sizeof W4; i++) {
            r = r * 1103515245u + 12345u;
            W4[i] = (uint8_t)(r >> 16);
        }
        for (size_t i = 0; i < (size_t)OUT * IN; i++) {
            r = r * 1103515245u + 12345u;
            union { uint32_t u; float f; } v;
            v.f = (float)((int)((r >> 16) & 0xffff) - 32768) * 1e-3f;
            Wb[i] = (uint16_t)(v.u >> 16);
        }
        /* Scales VARY here, unlike the gate above. A constant 127 is exactly
         * 1.0, so the whole per-block scale multiply cancels -- and the fold is
         * precisely what the batched kernels rearranged. */
        for (size_t i = 0; i < sizeof S8; i++) {
            r = r * 1103515245u + 12345u;
            S8[i] = (uint8_t)(120u + ((r >> 16) % 15u));
        }
        for (size_t i = 0; i < sizeof S4; i++) {
            r = r * 1103515245u + 12345u;
            S4[i] = (uint8_t)(120u + ((r >> 16) % 15u));
        }

        const struct { const char *name; DSV4QMat m; } cases[3] = {
            { "bf16", { Wb, NULL, DSV4_WBF16, OUT, IN, 0,   0   } },
            { "fp8",  { W8, S8,   DSV4_WFP8,  OUT, IN, 128, 128 } },
            { "fp4",  { W4, S4,   DSV4_WFP4,  OUT, IN, 1,   32  } },
        };

        for (int ci = 0; ci < 3; ci++) {
            const DSV4QMat *m = &cases[ci].m;
            memset(ybatch, 0, sizeof ybatch);
            dsv4_mmq_n(ybatch, xb, m, NT);

            int bad = 0;
            for (int t = 0; t < NT && !bad; t++) {
                dsv4_mmq(yone, xb + (size_t)t * IN, m);
                if (memcmp(yone, ybatch + (size_t)t * OUT,
                           (size_t)OUT * sizeof(float)) != 0) {
                    /* Name the first differing row: "they differ" is not
                     * actionable, "row 12 of token 3" is. */
                    int at = 0;
                    for (int o = 0; o < OUT; o++)
                        if (yone[o] != ybatch[(size_t)t * OUT + o]) { at = o; break; }
                    CHECK(0, "%s batch token %d row %d: gemv %.17g, gemm %.17g",
                          cases[ci].name, t, at, (double)yone[at],
                          (double)ybatch[(size_t)t * OUT + at]);
                    bad = 1;
                }
            }
            if (!bad)
                printf("  ok    %-4s %d tokens x %dx%d identical to %d GEMVs\n",
                       cases[ci].name, NT, OUT, IN, NT);
        }

        /* nt == 1 must land on the GEMV itself, not on a batched kernel that
         * happens to agree. The decode path is every token after the prompt and
         * it is the path the whole-model oracle gates. */
        {
            const DSV4QMat *m = &cases[1].m;
            dsv4_mmq(yone, xb, m);
            memset(ybatch, 0, sizeof ybatch);
            dsv4_mmq_n(ybatch, xb, m, 1);
            CHECK(memcmp(yone, ybatch, (size_t)OUT * sizeof(float)) == 0,
                  "nt == 1 does not match the GEMV");
        }
    }

    if (fails) { printf("MATMUL GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("MATMUL GATE PASSED\n");
    return 0;
}
