/* SPDX-License-Identifier: Apache-2.0 */
/* bench/gpu_contention.c - does GPU activity slow down CPU work, and why?
 *
 * THE OBSERVATION THIS EXISTS TO EXPLAIN
 *   Running the model with --gpu made the routed-expert matmuls -- which never
 *   touch the device -- go from 16.9 s to 59.7 s. PCIe latency cannot explain
 *   that: those kernels do not cross the bus. The first guess was that the CPU
 *   and dGPU share a laptop power envelope, but the machine was on AC in
 *   performance mode throughout, so that explanation is weak.
 *
 *   The remaining suspect is CUDA's default synchronisation, which SPINS: the
 *   calling thread busy-waits for the device instead of sleeping. With OpenMP
 *   already running one thread per core, a spinning thread does not merely add
 *   load, it competes with the very threads it is waiting behind.
 *
 * WHAT THIS MEASURES
 *   The same CPU-only FP4 matmul, timed twice: once with the device idle, once
 *   with a background thread issuing GPU work continuously. The ratio is the
 *   contention. Run it as `gpu_contention spin` and `gpu_contention block` --
 *   the sync mode has to be chosen before the CUDA context exists, so the two
 *   cannot be compared inside one process.
 */
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "dsv4.h"
#include "dsv4_cuda.h"

void dsv4_matmul_fp4(float *, const float *, const uint8_t *, const uint8_t *,
                     int, int, int, int);
int  dsv4_cuda_set_blocking_sync(void);

static double now(void)
{
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + 1e-9 * (double)t.tv_nsec;
}

/* the GPU load generator */
static volatile int g_stop;
static DSV4QMat     g_m;
static float       *g_gx, *g_gy;
static long         g_calls;

/* Calls per second to aim for, 0 = as fast as possible. The rate is the whole
 * point: if contention scales with how OFTEN we cross into the device rather
 * than with how much work we send, then the fix is fewer, bigger calls -- which
 * is exactly what running a whole attention block on-device would do. */
static double g_period;

static void *hammer(void *arg)
{
    (void)arg;
    while (!g_stop) {
        const double t = now();
        dsv4_cuda_mmq(g_gy, g_gx, &g_m);
        g_calls++;
        if (g_period > 0.0) {
            /* SLEEP, do not spin. The first version of this throttle busy-
             * waited, which meant every "rate" still pinned a core solid -- so
             * contention came out flat at ~23x for 1200, 300, 90 and 20 calls
             * per second, and what was actually being measured was one spinning
             * thread oversubscribing a 20-thread OpenMP pool, not the device. */
            const double left = g_period - (now() - t);
            if (left > 0.0) {
                struct timespec ts;
                ts.tv_sec  = (time_t)left;
                ts.tv_nsec = (long)((left - (double)ts.tv_sec) * 1e9);
                nanosleep(&ts, NULL);
            }
        }
    }
    return NULL;
}

int main(int argc, char **argv)
{
    const int blocking = (argc > 1 && strcmp(argv[1], "block") == 0);
    const double rate = (argc > 2) ? atof(argv[2]) : 0.0;   /* calls/sec */
    g_period = (rate > 0.0) ? 1.0 / rate : 0.0;

    if (blocking && dsv4_cuda_set_blocking_sync() != 0) {
        printf("could not select blocking sync\n");
        return 1;
    }
    if (!dsv4_cuda_available() || dsv4_cuda_init() != 0) {
        printf("no CUDA device; nothing to measure\n");
        return 0;
    }
    printf("sync mode: %s\n", blocking ? "BLOCKING (thread sleeps)"
                                       : "SPIN (default -- thread busy-waits)");

    /* CPU victim: one routed expert's w1, the shape the model spends 25% of
     * its time in and which never goes near the device. */
    const int cin = 4096, cout = 2048;
    uint8_t *W4 = malloc((size_t)cout * cin / 2);
    uint8_t *S4 = malloc((size_t)cout * (cin / 32));
    float *cx = malloc((size_t)cin * sizeof *cx);
    float *cy = malloc((size_t)cout * sizeof *cy);

    /* GPU load: a real trunk matrix. */
    const int gin = 8192, gout = 4096;
    uint8_t *W8 = malloc((size_t)gout * gin);
    uint8_t *S8 = malloc((size_t)(gout / 128) * (gin / 128));
    g_gx = malloc((size_t)gin * sizeof *g_gx);
    g_gy = malloc((size_t)gout * sizeof *g_gy);
    if (!W4 || !S4 || !cx || !cy || !W8 || !S8 || !g_gx || !g_gy) return 1;

    unsigned r = 7u;
    for (int i = 0; i < cin; i++) { r = r*1103515245u+12345u;
        cx[i] = (float)((int)((r>>16)&0xffff)-32768)*1e-3f; }
    for (int i = 0; i < gin; i++) { r = r*1103515245u+12345u;
        g_gx[i] = (float)((int)((r>>16)&0xffff)-32768)*1e-3f; }
    for (size_t i = 0; i < (size_t)cout*cin/2; i++) { r = r*1103515245u+12345u;
        W4[i] = (uint8_t)(r>>16); }
    for (size_t i = 0; i < (size_t)gout*gin; i++) { r = r*1103515245u+12345u;
        uint8_t b=(uint8_t)(r>>16); if ((b&0x7Fu)==0x7Fu) b&=0xFEu; W8[i]=b; }
    memset(S4, 127, (size_t)cout*(cin/32));
    memset(S8, 127, (size_t)(gout/128)*(gin/128));

    memset(&g_m, 0, sizeof g_m);
    g_m.w = W8; g_m.s = S8; g_m.wdt = DSV4_WFP8;
    g_m.rows = gout; g_m.cols = gin; g_m.blk_r = 128; g_m.blk_c = 128;
    if (dsv4_cuda_upload(&g_m) != 0) { printf("upload failed\n"); return 1; }

    const double flop = 2.0 * (double)cin * (double)cout;
    const int reps = 60;

    double quiet = 1e30;
    for (int i = 0; i < reps; i++) {
        const double t0 = now();
        dsv4_matmul_fp4(cy, cx, W4, S4, cin, cout, 1, 32);
        const double dt = now() - t0;
        if (dt < quiet) quiet = dt;
    }
    printf("  cpu fp4 matmul, device IDLE : %6.1f GF/s  (%.3f ms)\n",
           flop / quiet * 1e-9, quiet * 1e3);

    pthread_t th;
    g_stop = 0; g_calls = 0;
    pthread_create(&th, NULL, hammer, NULL);
    /* let the GPU get going before timing anything -- sleeping, for the same
     * reason the throttle sleeps */
    { struct timespec ts = { 0, 300000000L }; nanosleep(&ts, NULL); }

    double busy = 1e30;
    const double t_start = now();
    for (int i = 0; i < reps; i++) {
        const double t0 = now();
        dsv4_matmul_fp4(cy, cx, W4, S4, cin, cout, 1, 32);
        const double dt = now() - t0;
        if (dt < busy) busy = dt;
    }
    const double span = now() - t_start;
    g_stop = 1;
    pthread_join(th, NULL);

    printf("  cpu fp4 matmul, device BUSY : %6.1f GF/s  (%.3f ms)\n",
           flop / busy * 1e-9, busy * 1e3);
    printf("  contention: CPU work is %.2fx slower while the GPU is active\n",
           busy / quiet);
    printf("  (%ld gpu calls issued during a %.1f s window)\n", g_calls, span);

    dsv4_cuda_shutdown();
    return 0;
}
