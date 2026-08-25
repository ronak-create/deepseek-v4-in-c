/* SPDX-License-Identifier: Apache-2.0 */
/* bench/gemm_bw.c - what does batching a matmul over nt tokens actually buy?
 *
 * THE CLAIM UNDER TEST
 *   A GEMV reads a weight matrix once and does one multiply-add per weight:
 *   arithmetic intensity ~1 flop per byte, so it is bound by how fast the
 *   weights arrive from DRAM. Feed nt tokens through the same loaded weights
 *   and the intensity becomes ~nt. Where that stops helping is the number that
 *   caps DSV4_MAX_BATCH and decides what batched prefill can be worth.
 *
 * THE TRAP THIS BENCH HAD TO BE REWRITTEN TO AVOID
 *   Time a GEMV in a tight loop over ONE weight matrix and the weights stay in
 *   L3 from one call to the next. The kernel then measures as compute-bound,
 *   the GEMV baseline looks far too good, and batching appears to buy nothing.
 *
 *   That is not the engine's situation. Between two uses of the same weight
 *   matrix, a decoder layer touches ~165 MB of other weights, so every matrix
 *   is COLD in cache every time it is used. A bench that keeps them warm is
 *   measuring a program nobody runs.
 *
 *   So this allocates many distinct copies of each matrix, totalling far more
 *   than L3, and cycles through them. Every call reads cold weights, batched
 *   and unbatched alike, which is the only way the ratio means anything.
 *
 *   The first version did not do this. It reported a 53x speedup on a 4 MB FP4
 *   expert and 1.00x on another 4 MB FP4 expert, which is not a result, it is
 *   two different cache states.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4.h"

void dsv4_mmq(float *, const float *, const DSV4QMat *);
void dsv4_mmq_n(float *, const float *, const DSV4QMat *, int);
int  dsv4_matmul_has_avx2(void);

/* Total weight bytes to spread the copies over. Comfortably past any L3 on a
 * machine this engine would run on, so the last copy has evicted the first. */
#define SPREAD_BYTES (512u * 1024u * 1024u)

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

static unsigned rng = 20260825u;
static unsigned nextr(void) { rng = rng * 1103515245u + 12345u; return rng; }

static const struct { int rows, cols, fp4; const char *what; } SHAPES[] = {
    { 2048, 4096, 1, "fp4 expert w1  2048x4096" },
    { 4096, 2048, 1, "fp4 expert w2  4096x2048" },
    { 4096, 8192, 0, "fp8 attn wo_b  4096x8192" },
    { 2048, 4096, 0, "fp8 shared w1  2048x4096" },
};
#define NSHAPE ((int)(sizeof SHAPES / sizeof SHAPES[0]))

static const int NT[] = { 1, 2, 4, 8, 16, 32, 64 };
#define NNT ((int)(sizeof NT / sizeof NT[0]))

int main(int argc, char **argv)
{
    const int reps = argc > 1 ? atoi(argv[1]) : 16;

    printf("batched matmul throughput, Flash geometries, AVX2 %s\n",
           dsv4_matmul_has_avx2() ? "COMPILED IN" : "not available");
    printf("Weights are COLD on every call: each rep uses a different copy,\n"
           "%u MB of them, so nothing survives in cache between calls.\n",
           SPREAD_BYTES / (1024u * 1024u));
    printf("Speedup is PER TOKEN. 1.00x means batching bought nothing.\n\n");

    for (int s = 0; s < NSHAPE; s++) {
        const int out = SHAPES[s].rows, in = SHAPES[s].cols;
        const int blk_r = SHAPES[s].fp4 ? 1 : 128;
        const int blk_c = SHAPES[s].fp4 ? 32 : 128;
        const int sc = (in + blk_c - 1) / blk_c;

        const size_t wb = SHAPES[s].fp4 ? (size_t)out * in / 2
                                        : (size_t)out * in;
        const size_t sb = (size_t)((out + blk_r - 1) / blk_r) * sc;

        int ncopy = (int)(SPREAD_BYTES / wb);
        if (ncopy < 2) ncopy = 2;

        uint8_t **W = (uint8_t **)calloc((size_t)ncopy, sizeof *W);
        uint8_t **S = (uint8_t **)calloc((size_t)ncopy, sizeof *S);
        float *x = (float *)malloc((size_t)in * DSV4_MAX_BATCH * sizeof *x);
        float *y = (float *)malloc((size_t)out * DSV4_MAX_BATCH * sizeof *y);
        if (!W || !S || !x || !y) { printf("  out of memory\n"); return 1; }

        for (int i = 0; i < in * DSV4_MAX_BATCH; i++)
            x[i] = (float)((int)((nextr() >> 16) & 0xffff) - 32768) * 1e-3f;

        for (int cpy = 0; cpy < ncopy; cpy++) {
            W[cpy] = (uint8_t *)malloc(wb);
            S[cpy] = (uint8_t *)malloc(sb);
            if (!W[cpy] || !S[cpy]) { printf("  out of memory\n"); return 1; }
            for (size_t i = 0; i < wb; i++) {
                uint8_t b = (uint8_t)(nextr() >> 16);
                if (!SHAPES[s].fp4 && (b & 0x7Fu) == 0x7Fu) b &= 0xFEu;
                W[cpy][i] = b;
            }
            for (size_t i = 0; i < sb; i++)
                S[cpy][i] = (uint8_t)(120u + ((nextr() >> 16) % 15u));
        }

        DSV4QMat m;
        memset(&m, 0, sizeof m);
        m.wdt = SHAPES[s].fp4 ? DSV4_WFP4 : DSV4_WFP8;
        m.rows = out; m.cols = in; m.blk_r = blk_r; m.blk_c = blk_c;

        /* Spin up the OpenMP team and ramp clocks before anything is timed.
         * Without this the FIRST point measured absorbs all of it -- the
         * earlier version reported 9-10 ms for a matrix that settles at 0.15,
         * and best-of-5 did not remove it because the ramp is longer than five
         * calls. */
        m.w = W[0]; m.s = S[0];
        for (int wcall = 0; wcall < 60; wcall++) dsv4_mmq_n(y, x, &m, 8);

        printf("%s   (%.1f MB each, %d copies)\n",
               SHAPES[s].what, (double)wb / 1048576.0, ncopy);
        printf("  %4s %12s %12s %10s %10s\n",
               "nt", "ms/batch", "ms/token", "GF/s", "speedup");

        double base = 0.0;
        for (int q = 0; q < NNT; q++) {
            const int nt = NT[q];
            /* MEAN, not best-of. Best-of would quietly select the reps that
             * happened to hit a warm copy, which is the very effect this bench
             * is built to exclude. */
            /* Reps scaled so every point gets a comparable amount of wall
             * clock. A fixed count times a 0.15 ms call is 2.4 ms in total, and
             * then one scheduler hiccup IS the measurement. Three runs of the
             * identical binary put this shape's nt=1 point at 0.159, 0.617 and
             * 0.985 ms before this was fixed -- a 6x spread on the very point
             * every speedup below is divided by. The big shapes were stable to
             * 10% throughout, which is how the cause was narrowed to duration.
             */
            int n_rep = reps;
            while ((double)n_rep * nt < 512.0) n_rep *= 2;

            const double t0 = now();
            for (int r = 0; r < n_rep; r++) {
                m.w = W[r % ncopy];
                m.s = S[r % ncopy];
                dsv4_mmq_n(y, x, &m, nt);
            }
            const double dt = (now() - t0) / n_rep;

            const double per = dt / nt;
            if (q == 0) base = per;
            /* GF/s, not GB/s. The weight bytes are the same at every nt, so
             * a bandwidth column would fall by construction and say nothing.
             * Throughput against the kernel's known GEMV peak is the check
             * that matters: a shape already at peak cannot gain from batching,
             * and one far below it should. */
            const double flop = 2.0 * (double)in * (double)out * nt;
            printf("  %4d %12.3f %12.4f %10.1f %9.2fx\n",
                   nt, dt * 1e3, per * 1e3, flop / dt * 1e-9, base / per);
        }
        printf("\n");

        for (int cpy = 0; cpy < ncopy; cpy++) { free(W[cpy]); free(S[cpy]); }
        free(W); free(S); free(x); free(y);
    }

    printf("Read where the per-token column stops falling. Past that point a\n"
           "longer prefill chunk costs stack and activation memory and buys\n"
           "nothing, and that is what caps DSV4_MAX_BATCH.\n");
    return 0;
}
