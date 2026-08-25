/* SPDX-License-Identifier: Apache-2.0 */
/* test_cuda.c - the GPU path, gated on the only terms a GPU can meet.
 *
 * WHY THIS GATE LOOKS DIFFERENT FROM EVERY OTHER ONE
 *   Every other kernel in this engine is checked for BIT equality, because the
 *   16-accumulator tree and -ffp-contract=off make scalar, OpenMP and AVX2
 *   produce identical bytes. A GPU cannot join that club: its reduction order
 *   falls out of block and warp geometry, and pretending otherwise would
 *   quietly weaken the guarantee the rest of the codebase rests on.
 *
 *   So the terms change, and they are stated rather than assumed:
 *
 *     GATE 1  the DEVICE's E4M3 decoder must be EXACT against the host's, on
 *             all 256 codes. This one IS bit equality, and it must be: the
 *             decoder is a table, not a reduction, and there is no numerical
 *             excuse for it to differ. It is also the one piece of duplicated
 *             logic in the GPU path, so it is the one most likely to drift.
 *
 *     GATE 2  the matmul must agree to a bounded RELATIVE error, measured
 *             against the vector's own scale rather than element-wise -- a
 *             cancelled element thousands of times below the vector RMS says
 *             nothing about whether the kernel is right.
 *
 *     GATE 3  argmax identity, which is what actually decides a token.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4.h"
#include "dsv4_quant.h"
#include "dsv4_cuda.h"

void dsv4_matmul_fp8(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
int  dsv4_cuda_dump_e4m3(float *host256);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

int main(void)
{
    printf("DeepSeek-V4 CUDA co-processor gate\n");

    if (!dsv4_cuda_available()) {
        printf("  SKIP  built without CUDA, or no device present.\n");
        printf("        The CPU path is the reference and the default, so this\n"
               "        is a complete build, not a degraded one.\n");
        printf("\nCUDA GATE SKIPPED\n");
        return 0;
    }
    if (dsv4_cuda_init() != 0) { printf("\nCUDA GATE FAILED: init\n"); return 1; }

    printf("\n-- GATE 1  the device E4M3 decoder is EXACT against the host --\n");
    {
        float dev[256];
        if (dsv4_cuda_dump_e4m3(dev) != 0) {
            CHECK(0, "could not read the device decoder");
        } else {
            int bad = 0, nan_ok = 0;
            for (int i = 0; i < 256; i++) {
                const float h = dsv4_e4m3_to_f32((uint8_t)i);
                if (isnan(h)) {
                    /* The two NaN codes: require NaN, not a payload match. */
                    if (!isnan(dev[i])) { bad++;
                        CHECK(0, "code 0x%02X: host NaN, device %g", i, dev[i]); }
                    else nan_ok++;
                    continue;
                }
                if (memcmp(&h, &dev[i], sizeof h) != 0) {
                    bad++;
                    if (bad <= 4)
                        CHECK(0, "code 0x%02X: host %.9g, device %.9g",
                              i, (double)h, (double)dev[i]);
                }
            }
            if (!bad)
                printf("  ok    all 256 codes identical (%d NaN codes checked "
                       "as NaN)\n", nan_ok);
        }
    }

    printf("\n-- GATE 2/3  FP8 matmul against the CPU reference --\n");
    {
        /* A real trunk shape. wo_b is 4096x8192 and is one of the matrices the
         * profile says dominates: 33.5 MB re-read from RAM every layer. */
        const int in = 8192, out = 4096;
        const int blk_r = 128, blk_c = 128;
        const int sc = (in + blk_c - 1) / blk_c;

        uint8_t *W = (uint8_t *)malloc((size_t)out * in);
        uint8_t *S = (uint8_t *)malloc((size_t)((out + blk_r - 1) / blk_r) * sc);
        float *x  = (float *)malloc((size_t)in * sizeof *x);
        float *yc = (float *)malloc((size_t)out * sizeof *yc);
        float *yg = (float *)malloc((size_t)out * sizeof *yg);
        if (!W || !S || !x || !yc || !yg) { printf("  FAIL  oom\n"); return 1; }

        unsigned r = 20260820u;
        for (int i = 0; i < in; i++) {
            r = r * 1103515245u + 12345u;
            x[i] = (float)((int)((r >> 16) & 0xffff) - 32768) * 1e-3f;
        }
        for (size_t i = 0; i < (size_t)out * in; i++) {
            r = r * 1103515245u + 12345u;
            uint8_t b = (uint8_t)(r >> 16);
            /* No NaN weights: a NaN would make every comparison below
             * meaningless while looking like agreement. */
            if ((b & 0x7Fu) == 0x7Fu) b &= 0xFEu;
            W[i] = b;
        }
        /* Scales must VARY. A constant 127 is exactly 1.0, so the whole
         * per-block scale multiply cancels and the gate cannot see a
         * kernel that gets the scale path wrong -- or one whose scale
         * arithmetic is too narrow. Spread them over 2^-7..2^7. */
        for (size_t i = 0; i < (size_t)((out + blk_r - 1) / blk_r) * sc; i++) {
            r = r * 1103515245u + 12345u;
            S[i] = (uint8_t)(120u + ((r >> 16) % 15u));
        }

        DSV4QMat m;
        memset(&m, 0, sizeof m);
        m.w = W; m.s = S; m.wdt = DSV4_WFP8;
        m.rows = out; m.cols = in; m.blk_r = blk_r; m.blk_c = blk_c;

        if (dsv4_cuda_upload(&m) != 0) {
            CHECK(0, "upload of a %.1f MB matrix failed (free VRAM %.2f GB)",
                  (double)((size_t)out * in) / 1048576.0,
                  (double)dsv4_cuda_free_vram() / 1073741824.0);
        } else {
            double t0 = now();
            dsv4_matmul_fp8(yc, x, W, S, in, out, blk_r, blk_c);
            const double tc = now() - t0;

            dsv4_cuda_mmq(yg, x, &m);            /* warm up, then time */
            t0 = now();
            dsv4_cuda_mmq(yg, x, &m);
            const double tg = now() - t0;

            double sq = 0.0;
            for (int i = 0; i < out; i++) sq += (double)yc[i] * yc[i];
            const double rms = sqrt(sq / out);

            double worst = 0.0; int at = 0;
            for (int i = 0; i < out; i++) {
                const double d = fabs((double)yg[i] - (double)yc[i]);
                const double s = fmax(rms, fmax(fabs((double)yg[i]),
                                                fabs((double)yc[i])));
                const double rel = s > 0.0 ? d / s : 0.0;
                if (rel > worst) { worst = rel; at = i; }
            }

            int ac = 0, ag = 0;
            for (int i = 1; i < out; i++) {
                if (yc[i] > yc[ac]) ac = i;
                if (yg[i] > yg[ag]) ag = i;
            }

            const double flop = 2.0 * (double)in * (double)out;
            printf("  cpu %.1f GF/s   gpu %.1f GF/s   %.1fx\n",
                   flop / tc * 1e-9, flop / tg * 1e-9, tc / tg);

            /* 1e-5 of the vector's own scale. Loose enough to allow a different
             * reduction order, tight enough that a wrong kernel cannot pass:
             * a swapped nibble or a misindexed scale moves this by orders of
             * magnitude, not by a few ulp. */
            CHECK(worst < 1e-5, "worst relative error %.3g at [%d] "
                  "(cpu %.6g vs gpu %.6g)", worst, at,
                  (double)yc[at], (double)yg[at]);
            CHECK(ac == ag, "argmax differs: cpu %d, gpu %d", ac, ag);
            if (worst < 1e-5 && ac == ag)
                printf("  ok    %dx%d agrees to %.2g relative, argmax %d both\n",
                       out, in, worst, ac);
        }
        free(W); free(S); free(x); free(yc); free(yg);
    }

    dsv4_cuda_shutdown();
    printf("\n");
    if (fails) { printf("CUDA GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("CUDA GATE PASSED\n");
    return 0;
}
