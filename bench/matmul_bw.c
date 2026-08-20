/* SPDX-License-Identifier: Apache-2.0 */
/* bench/matmul_bw.c - scalar vs AVX2, on the shapes the model actually uses.
 *
 * A full generation run takes ~100 s and its wall clock moves with disk state,
 * so it is a poor instrument for a kernel change. This times the kernels
 * directly, at Flash's real geometries, and reports GFLOP/s for both paths
 * plus whether they agree to the bit -- because a faster kernel that is not the
 * same kernel is not an improvement, it is a bug.
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
void dsv4_matmul_fp4(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
void dsv4_matmul_fp8_scalar(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);
void dsv4_matmul_fp4_scalar(float *, const float *, const uint8_t *,
                            const uint8_t *, int, int, int, int);
int  dsv4_matmul_has_avx2(void);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static unsigned rng = 987654321u;
static unsigned nextr(void) { rng = rng * 1103515245u + 12345u; return rng; }

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 20;
    const int gpu = dsv4_cuda_available() && dsv4_cuda_init() == 0;

    printf("matmul throughput, Flash geometries, AVX2 %s\n",
           dsv4_matmul_has_avx2() ? "COMPILED IN" : "not available");
    printf("%-22s %10s %10s %8s  %s\n",
           "kernel", "scalar", "avx2", "speedup", "bitwise equal");

    struct { const char *name; int in, out, blk_r, blk_c, fp4; } job[] = {
        { "fp4 expert w1 2048x4096", 4096, 2048, 1, 32,  1 },
        { "fp4 expert w2 4096x2048", 2048, 4096, 1, 32,  1 },
        { "fp8 attn wo_b 4096x8192", 8192, 4096, 128, 128, 0 },
        { "fp8 attn wq_b 32768x1024", 1024, 32768, 128, 128, 0 },
    };

    for (unsigned j = 0; j < sizeof job / sizeof job[0]; j++) {
        const int in = job[j].in, out = job[j].out;
        const size_t wbytes = job[j].fp4 ? (size_t)out * in / 2
                                         : (size_t)out * in;
        const int sc = (in + job[j].blk_c - 1) / job[j].blk_c;
        const size_t sbytes = (size_t)((out + job[j].blk_r - 1) / job[j].blk_r)
                            * (size_t)sc;

        float *x = malloc((size_t)in * sizeof *x);
        float *ya = malloc((size_t)out * sizeof *ya);
        float *yb = malloc((size_t)out * sizeof *yb);
        float *yg = malloc((size_t)out * sizeof *yg);
        uint8_t *W = malloc(wbytes), *S = malloc(sbytes);
        if (!x || !ya || !yb || !yg || !W || !S) { printf("  out of memory\n"); return 1; }

        for (int i = 0; i < in; i++)
            x[i] = (float)((int)(nextr() >> 16 & 0xffff) - 32768) * 1e-3f;
        for (size_t i = 0; i < wbytes; i++) {
            uint8_t b = (uint8_t)(nextr() >> 16);
            /* e4m3fn NaNs excluded: a NaN result compares by payload, not by
             * arithmetic, and would make this say "equal" for the wrong reason
             * or "unequal" for no reason. */
            if (!job[j].fp4 && (b & 0x7Fu) == 0x7Fu) b &= 0xFEu;
            W[i] = b;
        }
        memset(S, 127, sbytes);          /* scale 2^0 */

        double ts = 1e30, tv = 1e30;
        for (int r = 0; r < reps; r++) {
            double t0 = now();
            if (job[j].fp4) dsv4_matmul_fp4_scalar(ya, x, W, S, in, out,
                                                   job[j].blk_r, job[j].blk_c);
            else            dsv4_matmul_fp8_scalar(ya, x, W, S, in, out,
                                                   job[j].blk_r, job[j].blk_c);
            double t1 = now();
            if (job[j].fp4) dsv4_matmul_fp4(yb, x, W, S, in, out,
                                            job[j].blk_r, job[j].blk_c);
            else            dsv4_matmul_fp8(yb, x, W, S, in, out,
                                            job[j].blk_r, job[j].blk_c);
            double t2 = now();
            /* Best of N, not mean: the minimum is the one measurement not
             * contaminated by whatever else the machine was doing. */
            if (t1 - t0 < ts) ts = t1 - t0;
            if (t2 - t1 < tv) tv = t2 - t1;
        }

        /* PER-CALL LATENCY IS THE QUESTION FOR A GEMV.
         *
         * A matrix-VECTOR product moves a lot of weight and very little
         * activation, so if each offloaded call costs a fixed round trip, the
         * GPU can lose on small matrices however fast its arithmetic is. This
         * times a full dsv4_cuda_mmq -- H2D copy, kernel, D2H copy -- which is
         * what the engine actually pays per call, not a kernel in isolation. */
        double tgpu = 0.0;
        DSV4QMat mm;
        memset(&mm, 0, sizeof mm);
        mm.w = W; mm.s = S; mm.wdt = job[j].fp4 ? DSV4_WFP4 : DSV4_WFP8;
        mm.rows = out; mm.cols = in;
        mm.blk_r = job[j].blk_r; mm.blk_c = job[j].blk_c;
        if (gpu && dsv4_cuda_upload(&mm) == 0) {
            dsv4_cuda_mmq(yg, x, &mm);                 /* warm */
            tgpu = 1e30;
            for (int r = 0; r < reps; r++) {
                const double a0 = now();
                dsv4_cuda_mmq(yg, x, &mm);
                const double a1 = now() - a0;
                if (a1 < tgpu) tgpu = a1;
            }
        }

        const int same = memcmp(ya, yb, (size_t)out * sizeof *ya) == 0;
        const double flop = 2.0 * (double)in * (double)out;
        printf("%-22s %7.1f GF %7.1f GF %7.2fx  %s\n",
               job[j].name, flop / ts * 1e-9, flop / tv * 1e-9, ts / tv,
               same ? "yes" : "NO -- THE PATHS DISAGREE");
        if (tgpu > 0.0 && tgpu < 1e29)
            printf("%-22s %7.1f GF  %7.3f ms/call   cpu(avx2) %7.3f ms/call\n",
                   "  ^ gpu round trip", flop / tgpu * 1e-9,
                   tgpu * 1e3, tv * 1e3);

        free(x); free(ya); free(yb); free(yg); free(W); free(S);
    }
    return 0;
}
