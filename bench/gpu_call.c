/* SPDX-License-Identifier: Apache-2.0 */
/* bench/gpu_call.c - what does one dsv4_cuda_mmq call cost, and what does that
 * cost actually depend on?
 *
 * THE CLAIM THIS EXISTS TO KILL
 *   The README used to say there is "a ~1.3 ms floor per call that barely
 *   varies with size". That was inferred from exactly two matrices --
 *   wo_b 4096x8192 and wq_b 32768x1024 -- which are BOTH 33.5 MB. Two points at
 *   the same size cannot separate a fixed overhead from a cost linear in bytes.
 *   They agreed because they were the same measurement twice.
 *
 * WHAT THIS MEASURES
 *   The full host-visible call -- H2D activation copy, kernel, D2H result copy
 *   and the implicit sync -- over weight matrices spanning four orders of
 *   magnitude, with the weights already resident in VRAM. That is how the engine
 *   uses the device: upload once per run, call once per layer per token.
 *
 *   A least-squares fit of time against weight bytes then separates the two
 *   terms the old claim conflated: a fixed per-call cost, and a slope that is
 *   the achieved effective bandwidth. The INTERCEPT is the floor, not the
 *   smallest measured time.
 *
 * READ THE CPU COLUMN WITH CARE BELOW ~1 MB
 *   At 512x512 the CPU column reports ~5.3 ms with the default 20 OpenMP
 *   threads and ~0.41 ms with OMP_NUM_THREADS=1 -- 13x SLOWER for having more
 *   threads, repeatably, and after a warm-up call. At that size the column is
 *   measuring OpenMP fork-join inside a process that holds a live CUDA context
 *   and pinned host memory, not a matmul. Whether the CUDA context is the
 *   cause is NOT measured here; bench/gpu_contention.c documents a related
 *   effect and does not explain this one either. The fit below uses only the
 *   GPU column, so the conclusion does not depend on it.
 *
 * WHY IT MATTERS
 *   The floor decides which matrices are worth sending to the device at all. A
 *   1.3 ms floor says "only the 33.5 MB ones". A much smaller floor says the
 *   small matrices deserve a second look.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4.h"
#include "dsv4_cuda.h"

void dsv4_matmul_fp8(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* Real Flash/Pro trunk shapes plus deliberately small ones, so the fit has
 * leverage at both ends instead of two points sitting on top of each other. */
static const struct { int rows, cols; const char *what; } SHAPES[] = {
    {   512,   512, "tiny probe"           },
    {  1024,  1024, "small probe"          },
    {  2048,  2048, "medium probe"         },
    {  4096,  1024, "wkv-ish"              },
    {  2048,  4096, "expert w1 shape"      },
    {  4096,  4096, "square 16.8 MB"       },
    {  4096,  8192, "flash wo_b  33.5 MB"  },
    { 32768,  1024, "flash wq_b  33.5 MB"  },
    {  8192,  8192, "square 67.1 MB"       },
    {  4096, 16384, "pro-width   67.1 MB"  },
};
#define NSHAPE ((int)(sizeof SHAPES / sizeof SHAPES[0]))

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 200;

    if (!dsv4_cuda_available()) {
        printf("no CUDA device; nothing to measure\n");
        return 0;
    }
    if (dsv4_cuda_init() != 0) { printf("cuda init failed\n"); return 1; }

    printf("per-call cost of dsv4_cuda_mmq, weights already resident\n");
    printf("(%d reps each, best-of: the minimum is the one run with nothing\n"
           " unrelated scheduled on top of it)\n\n", reps);
    printf("%-22s %8s %10s %10s %10s\n",
           "shape", "MB", "gpu ms", "cpu ms", "GB/s eff");
    printf("(below ~1 MB the cpu column is OpenMP fork-join, not matmul --"
           " see the header)\n\n");

    /* EVERY buffer stays alive to the end of the run, deliberately.
     * dsv4_cuda_upload keys residency on the HOST weight pointer, which is
     * sound in the engine because the trunk pins each layer for the whole
     * run. A bench that frees W between shapes breaks that contract the
     * moment malloc hands the same address back: the next upload reports a
     * hit, mmq then runs with the PREVIOUS matrix's dimensions, and copies
     * 32768 floats into a 4096-float result. It corrupts the heap, and it
     * is the bench that is wrong, not the engine. */
    uint8_t *keepW[NSHAPE], *keepS[NSHAPE];
    float   *keepx[NSHAPE], *keepy[NSHAPE];
    for (int i = 0; i < NSHAPE; i++) {
        keepW[i] = NULL; keepS[i] = NULL; keepx[i] = NULL; keepy[i] = NULL;
    }

    /* Accumulators for a least-squares fit of ms against MB. */
    double sx = 0.0, sy = 0.0, sxx = 0.0, sxy = 0.0;
    int    nfit = 0;

    for (int s = 0; s < NSHAPE; s++) {
        const int out = SHAPES[s].rows, in = SHAPES[s].cols;
        const int blk_r = 128, blk_c = 128;
        const int sc = (in + blk_c - 1) / blk_c;
        const size_t wb = (size_t)out * in;
        const size_t sb = (size_t)((out + blk_r - 1) / blk_r) * sc;

        uint8_t *W = (uint8_t *)malloc(wb);
        uint8_t *S = (uint8_t *)malloc(sb);
        float *x = (float *)malloc((size_t)in * sizeof *x);
        float *y = (float *)malloc((size_t)out * sizeof *y);
        if (!W || !S || !x || !y) { printf("  oom at %dx%d\n", out, in); break; }
        keepW[s] = W; keepS[s] = S; keepx[s] = x; keepy[s] = y;

        unsigned r = 20260825u;
        for (int i = 0; i < in; i++) {
            r = r * 1103515245u + 12345u;
            x[i] = (float)((int)((r >> 16) & 0xffff) - 32768) * 1e-3f;
        }
        for (size_t i = 0; i < wb; i++) {
            r = r * 1103515245u + 12345u;
            uint8_t b = (uint8_t)(r >> 16);
            if ((b & 0x7Fu) == 0x7Fu) b &= 0xFEu;   /* never a NaN weight */
            W[i] = b;
        }
        for (size_t i = 0; i < sb; i++) {
            r = r * 1103515245u + 12345u;
            S[i] = (uint8_t)(120u + ((r >> 16) % 15u));
        }

        DSV4QMat m;
        memset(&m, 0, sizeof m);
        m.w = W; m.s = S; m.wdt = DSV4_WFP8;
        m.rows = out; m.cols = in; m.blk_r = blk_r; m.blk_c = blk_c;

        if (dsv4_cuda_upload(&m) != 0) {
            printf("%-22s %8.1f   upload failed (%.2f GB VRAM free)\n",
                   SHAPES[s].what, (double)wb / 1048576.0,
                   (double)dsv4_cuda_free_vram() / 1073741824.0);
            continue;
        }

        dsv4_cuda_mmq(y, x, &m);                    /* warm the path */
        double g = 1e30;
        for (int i = 0; i < reps; i++) {
            const double t0 = now();
            dsv4_cuda_mmq(y, x, &m);
            const double dt = now() - t0;
            if (dt < g) g = dt;
        }

        /* The CPU column is what the device is being compared against. Fewer
         * reps: it is 3-40x slower and far less variable.
         *
         * The warm-up call is not optional. Without it the FIRST shape
         * absorbs the whole OpenMP team spin-up and reports something like
         * 6 ms for a 0.25 MB matrix -- a number wrong by two orders of
         * magnitude, sitting in a table next to correct ones. */
        dsv4_matmul_fp8(y, x, W, S, in, out, blk_r, blk_c);
        double c = 1e30;
        const int creps = reps / 20 > 3 ? reps / 20 : 3;
        for (int i = 0; i < creps; i++) {
            const double t0 = now();
            dsv4_matmul_fp8(y, x, W, S, in, out, blk_r, blk_c);
            const double dt = now() - t0;
            if (dt < c) c = dt;
        }

        const double mb = (double)wb / 1048576.0;
        printf("%-22s %8.1f %10.3f %10.3f %10.1f\n",
               SHAPES[s].what, mb, g * 1e3, c * 1e3,
               (double)(wb + sb) / g / 1e9);

        sx += mb; sy += g * 1e3; sxx += mb * mb; sxy += mb * g * 1e3; nfit++;
    }

    if (nfit >= 2) {
        const double den   = (double)nfit * sxx - sx * sx;
        const double slope = ((double)nfit * sxy - sx * sy) / den;  /* ms per MB */
        const double icept = (sy - slope * sx) / (double)nfit;      /* ms */
        printf("\nleast-squares fit over %d shapes:\n", nfit);
        printf("  ms = %.4f + %.5f * MB\n", icept, slope);
        printf("  fixed per-call cost : %.0f us\n", icept * 1e3);
        if (slope > 0.0)
            printf("  marginal bandwidth  : %.1f GB/s\n",
                   1.048576 / slope);
        printf("\nThe intercept is the floor, not the smallest measured time.\n"
               "If it sits far below the smallest matrix's total cost, that\n"
               "matrix is bandwidth-bound like the big ones, and the old\n"
               "\"fixed ~1.3 ms floor\" reading was an artifact of having\n"
               "measured only one size.\n");
    }

    dsv4_cuda_shutdown();
    for (int i = 0; i < NSHAPE; i++) {
        free(keepW[i]); free(keepS[i]); free(keepx[i]); free(keepy[i]);
    }
    return 0;
}
