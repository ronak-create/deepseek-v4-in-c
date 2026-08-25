/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cuda.cu - FP8 matrix-vector products with the weights resident in VRAM.
 *
 * See include/dsv4/dsv4_cuda.h for what runs here and what deliberately does
 * not. In one line: the dense trunk's FP8 matmuls, because they are bandwidth
 * bound against resident weights; not the routed FP4 experts, because those are
 * streamed off NVMe and a PCIe hop would only add to a disk read.
 */
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_cuda.h"

#define CUDA_TRY(call, what)                                                   \
    do {                                                                       \
        const cudaError_t e_ = (call);                                         \
        if (e_ != cudaSuccess) {                                               \
            fprintf(stderr, "dsv4_cuda: %s: %s\n", (what),                     \
                    cudaGetErrorString(e_));                                   \
            return -1;                                                         \
        }                                                                      \
    } while (0)

/* ---------------------------------------------------------------- decode ---
 * E4M3 and E8M0 on the device, digit for digit what dsv4_quant.h does on the
 * host. Written out rather than shared because a __device__ function cannot
 * include the host header's unions without pulling in the rest of it, and a
 * SECOND implementation of a decoder is exactly the kind of thing that drifts.
 * The gate therefore compares device output against the host decoder on every
 * one of the 256 E4M3 codes before it compares any matrix. */
__device__ __forceinline__ float d_e4m3(unsigned char b)
{
    const unsigned int sign = (unsigned int)(b & 0x80u) << 24;
    const unsigned int exp  = (b >> 3) & 0x0Fu;
    const unsigned int man  = b & 0x07u;
    unsigned int u;

    if (exp == 0u) {                       /* subnormal, or a signed zero */
        if (man == 0u) { u = sign; }
        else {
            unsigned int e = 120u;         /* 127 - 7 + 0, then normalise up */
            unsigned int m = man;
            while (!(m & 0x04u)) { m <<= 1; e--; }
            m &= 0x03u;
            u = sign | (e << 23) | (m << 21);
        }
    } else if (exp == 0x0Fu && man == 0x07u) {
        u = sign | 0x7FC00000u;            /* the only NaN encodings */
    } else {
        u = sign | ((exp + 120u) << 23) | (man << 20);
    }
    return __int_as_float((int)u);
}

__device__ __forceinline__ float d_e8m0(unsigned char b)
{
    if (b == 0xFFu) return __int_as_float(0x7FC00000);      /* NaN */
    if (b == 0x00u) return ldexpf(1.0f, -127);              /* 2^-127 */
    return ldexpf(1.0f, (int)b - 127);
}

/* ---------------------------------------------------------------- kernel ---
 * One CUDA block per output row; one warp per 128-column scale block.
 *
 * The scale is applied ONCE PER BLOCK to the block's dot product, not per
 * element -- matching inference/kernel.py's fp4_gemm and the CPU path. Scaling
 * each weight would be algebraically equal, numerically different, and cost a
 * multiply per element instead of per block.
 *
 * ACCUMULATION IS SPLIT, AND THE SPLIT IS MEASURED, NOT ASSUMED.
 * The inner product over one 128-column scale block runs in float; the scale
 * multiply, the fold across blocks and the final cross-warp sum run in double.
 *
 * This kernel used to be double throughout, justified by "bandwidth-bound, so
 * precision is close to free". That was never measured and it is false --
 * consumer Blackwell runs FP64 at a small fraction of FP32, and the shipped
 * kernel was FP64-throughput-bound, not DRAM-bound. On identical memory
 * traffic (RTX 5060 Laptop, 50 iterations, warm):
 *
 *     shape          double      float       hybrid    worst rel vs double
 *     4096x8192      1.271 ms    0.373 ms    0.382 ms  float 4.3e-7  hyb 2.8e-7
 *     4096x16384     2.538 ms    1.015 ms    1.040 ms  float 6.4e-7  hyb 2.8e-7
 *     129280x4096   19.99  ms    7.365 ms    7.523 ms  float 8.2e-7  hyb 4.3e-7
 *
 * Hybrid costs 2-2.5% over pure float and buys back the property that matters:
 * float's error grows with the reduce dimension because the cross-block sum
 * gets longer, while the hybrid's does not -- only the fixed 128-element inner
 * sum is narrow. Pro's reduce dimension is 16384, so that is not academic.
 * Argmax was identical to the double kernel in every row of that table.
 */
__global__ void k_mmq_fp8(float *__restrict__ y,
                          const float *__restrict__ x,
                          const unsigned char *__restrict__ W,
                          const unsigned char *__restrict__ S,
                          int in, int blk_r, int blk_c, int sc)
{
    const int o = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;

    const unsigned char *row = W + (size_t)o * in;

    double mine = 0.0;
    for (int c0 = warp * blk_c; c0 < in; c0 += nwarp * blk_c) {
        const int n = min(blk_c, in - c0);
        float acc = 0.0f;
        for (int k = lane; k < n; k += 32)
            acc = fmaf(d_e4m3(row[c0 + k]), x[c0 + k], acc);
        /* Reduce the warp's 32 partials. */
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0)
            mine += (double)acc * (double)d_e8m0(S[(size_t)(o / blk_r) * sc
                                           + (c0 / blk_c)]);
    }

    /* One partial per warp leader; fold them in shared memory. */
    __shared__ double part[32];
    if (lane == 0) part[warp] = mine;
    __syncthreads();
    if (threadIdx.x == 0) {
        double total = 0.0;
        for (int w = 0; w < nwarp; w++) total += part[w];
        y[o] = (float)total;
    }
}

/* Elements of one scale block a single lane owns, and so the size of the
 * decoded-weight register cache in k_mmq_fp8_n. blk_c is 128 on both released
 * checkpoints, which puts the real value at 4; 8 leaves headroom without
 * spending registers that matter. A wider block than 32*this falls back to
 * decoding inline, so the kernel stays correct either way. */
#define DSV4_CU_MAXSLICE 8

/* Batched FP8: nt activation vectors against ONE resident weight matrix.
 *
 * WHY IT EXISTS. Prefill runs nt prompt tokens through the same weights, and
 * until now dsv4_mmq_n had no device branch at all -- so turning on --batch
 * silently moved every FP8 matmul from the GPU back to the CPU. That was
 * documented as deliberate ("a batched device path is a separate piece of
 * work"), and this is that work.
 *
 * THE LOOP ORDER IS THE POINT, exactly as on the CPU. One block owns one output
 * row, walks that row's weights once, and serves every token from the bytes it
 * has in hand. The single-token kernel re-reads the whole 33.5 MB matrix per
 * token; this reads it once per batch. The token loop sits INSIDE the scale
 * block so the 128 weight bytes stay in L1 across all nt of them.
 *
 * PER (ROW, TOKEN) THE REDUCTION IS IDENTICAL TO k_mmq_fp8 -- same warp
 * striding, same 32-lane shuffle fold, same float-inside-block/double-above
 * split. So nt == 1 through here is bit-identical to the single-token kernel,
 * and the gate in tests/unit/test_cuda.c checks precisely that rather than
 * trusting the claim. What batching changes is which dot products are in
 * flight, not how any one of them is summed.
 *
 * y is [nt][rows], matching the CPU batched kernels.
 */
__global__ void k_mmq_fp8_n(float *__restrict__ y,
                            const float *__restrict__ x,
                            const unsigned char *__restrict__ W,
                            const unsigned char *__restrict__ S,
                            int in, int blk_r, int blk_c, int sc, int nt)
{
    const int o = blockIdx.x;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int nwarp = blockDim.x >> 5;

    const unsigned char *row = W + (size_t)o * in;

    /* [nwarp][nt] partials. Only lane 0 of each warp ever touches its slice, so
     * the init below needs no barrier against the accumulation that follows. */
    extern __shared__ double part_n[];
    double *mine = part_n + (size_t)warp * nt;
    if (lane == 0)
        for (int t = 0; t < nt; t++) mine[t] = 0.0;

    for (int c0 = warp * blk_c; c0 < in; c0 += nwarp * blk_c) {
        const int n = min(blk_c, in - c0);
        /* Hoisted out of the token loop: one conversion instead of nt of them,
         * of the same value. Not a reassociation. */
        const double s = (double)d_e8m0(S[(size_t)(o / blk_r) * sc
                                        + (c0 / blk_c)]);

        /* DECODE THIS LANE'S SLICE ONCE, then let every token spend it.
         *
         * The first version of this kernel called d_e4m3 inside the token loop
         * and was compute-bound because of it: 8 tokens took 2.92 ms against
         * 4.00 ms for 8 separate single-vector calls, only 1.37x, when the
         * weight traffic had already fallen 8x. The decode, not the DRAM read,
         * was the cost. A lane owns at most blk_c/32 elements of a scale block
         * -- 4 at the shipped blk_c of 128 -- so they sit in registers.
         *
         * Bit-exactness is untouched: the same value is computed once instead
         * of nt times and the fmaf order per (row, token) does not move. */
        float wv[DSV4_CU_MAXSLICE];
        int nv = 0;
        const int slice = (n - lane + 31) / 32;
        if (slice <= DSV4_CU_MAXSLICE)
            for (int k = lane; k < n; k += 32) wv[nv++] = d_e4m3(row[c0 + k]);

        for (int t = 0; t < nt; t++) {
            const float *xt = x + (size_t)t * in;
            float acc = 0.0f;
            if (slice <= DSV4_CU_MAXSLICE) {
                int j = 0;
                for (int k = lane; k < n; k += 32)
                    acc = fmaf(wv[j++], xt[c0 + k], acc);
            } else {
                /* Only reachable if a future checkpoint ships a scale block
                 * wider than 32*DSV4_CU_MAXSLICE. Same arithmetic, no cache. */
                for (int k = lane; k < n; k += 32)
                    acc = fmaf(d_e4m3(row[c0 + k]), xt[c0 + k], acc);
            }
            for (int off = 16; off > 0; off >>= 1)
                acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
            if (lane == 0) mine[t] += (double)acc * s;
        }
    }

    __syncthreads();
    for (int t = threadIdx.x; t < nt; t += blockDim.x) {
        double total = 0.0;
        for (int w = 0; w < nwarp; w++) total += part_n[(size_t)w * nt + t];
        y[(size_t)t * gridDim.x + o] = (float)total;
    }
}

/* ------------------------------------------------------------- residency ---
 * Matrices are keyed by their host weight pointer. That is sound here because
 * the trunk pins each layer once for the life of the run, so the pointer is
 * stable; it would NOT be sound for the routed experts, whose slots are reused
 * by the LRU -- which is another reason they stay on the CPU. */
typedef struct {
    const void    *key;          /* host m->w */
    unsigned char *dw, *ds;
    int rows, cols, blk_r, blk_c, sc;
} Res;

static Res  *g_res;
static int   g_nres, g_cap;
static float *g_x, *g_y;         /* device activation staging */
static float *g_hx, *g_hy;       /* PINNED host staging -- see dsv4_cuda_mmq */
static int    g_hxcap, g_hycap;
static int    g_xcap, g_ycap;
static int    g_ready;

extern "C" int dsv4_cuda_available(void)
{
    int n = 0;
    return (cudaGetDeviceCount(&n) == cudaSuccess && n > 0) ? 1 : 0;
}

/* Choose how the calling thread waits for the device. MUST be called before any
 * context exists, i.e. before dsv4_cuda_init.
 *
 * The default is cudaDeviceScheduleSpin: the thread busy-waits. That is the
 * right default for a process whose CPU is otherwise idle, and the wrong one
 * here -- every core is already running an OpenMP worker on the routed experts,
 * so a spinning thread competes with the work it is waiting behind. */
extern "C" int dsv4_cuda_set_blocking_sync(void)
{
    CUDA_TRY(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync),
             "cudaSetDeviceFlags");
    return 0;
}

extern "C" int dsv4_cuda_init(void)
{
    int n = 0;
    CUDA_TRY(cudaGetDeviceCount(&n), "cudaGetDeviceCount");
    if (n < 1) { fprintf(stderr, "dsv4_cuda: no device\n"); return -1; }
    CUDA_TRY(cudaSetDevice(0), "cudaSetDevice");

    cudaDeviceProp p;
    CUDA_TRY(cudaGetDeviceProperties(&p, 0), "cudaGetDeviceProperties");
    size_t freeb = 0, totb = 0;
    CUDA_TRY(cudaMemGetInfo(&freeb, &totb), "cudaMemGetInfo");
    printf("cuda: %s, sm_%d%d, %.2f GB free of %.2f GB\n",
           p.name, p.major, p.minor,
           (double)freeb / 1073741824.0, (double)totb / 1073741824.0);
    g_ready = 1;
    return 0;
}

extern "C" size_t dsv4_cuda_free_vram(void)
{
    size_t freeb = 0, totb = 0;
    if (!g_ready || cudaMemGetInfo(&freeb, &totb) != cudaSuccess) return 0;
    return freeb;
}

static Res *find(const void *key)
{
    for (int i = 0; i < g_nres; i++) if (g_res[i].key == key) return &g_res[i];
    return NULL;
}

extern "C" int dsv4_cuda_has(const DSV4QMat *m)
{
    return (g_ready && m && m->w && find(m->w)) ? 1 : 0;
}

extern "C" int dsv4_cuda_upload(const DSV4QMat *m)
{
    if (!g_ready || !m || !m->w || !m->s) return -1;
    if (m->wdt != DSV4_WFP8) return -1;          /* FP4 stays on the CPU */
    if (find(m->w)) return 0;

    const int sc = (m->cols + m->blk_c - 1) / m->blk_c;
    const size_t wb = (size_t)m->rows * m->cols;
    const size_t sb = (size_t)((m->rows + m->blk_r - 1) / m->blk_r) * sc;

    /* Leave headroom: running the device to the last byte makes the next
     * allocation fail somewhere less convenient. */
    size_t freeb = dsv4_cuda_free_vram();
    if (wb + sb + (256u << 20) > freeb) return -1;

    unsigned char *dw = NULL, *ds = NULL;
    if (cudaMalloc((void **)&dw, wb) != cudaSuccess) return -1;
    if (cudaMalloc((void **)&ds, sb) != cudaSuccess) { cudaFree(dw); return -1; }
    if (cudaMemcpy(dw, m->w, wb, cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(ds, m->s, sb, cudaMemcpyHostToDevice) != cudaSuccess) {
        cudaFree(dw); cudaFree(ds); return -1;
    }

    if (g_nres == g_cap) {
        const int cap = g_cap ? g_cap * 2 : 256;
        Res *r = (Res *)realloc(g_res, (size_t)cap * sizeof *r);
        if (!r) { cudaFree(dw); cudaFree(ds); return -1; }
        g_res = r; g_cap = cap;
    }
    Res *r = &g_res[g_nres++];
    r->key = m->w; r->dw = dw; r->ds = ds;
    r->rows = m->rows; r->cols = m->cols;
    r->blk_r = m->blk_r; r->blk_c = m->blk_c; r->sc = sc;
    return 0;
}

static int stage(int in, int out)
{
    if (in > g_xcap) {
        if (g_x) cudaFree(g_x);
        if (cudaMalloc((void **)&g_x, (size_t)in * sizeof(float)) != cudaSuccess)
            { g_x = NULL; g_xcap = 0; return -1; }
        g_xcap = in;
    }
    if (out > g_ycap) {
        if (g_y) cudaFree(g_y);
        if (cudaMalloc((void **)&g_y, (size_t)out * sizeof(float)) != cudaSuccess)
            { g_y = NULL; g_ycap = 0; return -1; }
        g_ycap = out;
    }
    /* PINNED HOST STAGING, allocated once and reused.
     *
     * Copying straight from the caller's pageable buffer looks simpler and is
     * catastrophic here: the driver has to page-lock those pages for the DMA
     * and release them afterwards, on every single call. That work is not local
     * to the calling thread -- changing the process page tables forces TLB
     * shootdown IPIs across every core, and this engine has an OpenMP worker
     * pinned to all of them crunching routed experts.
     *
     * Measured with bench/gpu_contention.c: with pageable copies, CPU-side FP4
     * matmul fell from 114 GF/s to 3.6 GF/s while the device was busy -- 32x,
     * and only 23x with blocking sync, which is how we knew spinning was not
     * the real cause. */
    if (in > g_hxcap) {
        if (g_hx) cudaFreeHost(g_hx);
        if (cudaHostAlloc((void **)&g_hx, (size_t)in * sizeof(float),
                          cudaHostAllocDefault) != cudaSuccess)
            { g_hx = NULL; g_hxcap = 0; return -1; }
        g_hxcap = in;
    }
    if (out > g_hycap) {
        if (g_hy) cudaFreeHost(g_hy);
        if (cudaHostAlloc((void **)&g_hy, (size_t)out * sizeof(float),
                          cudaHostAllocDefault) != cudaSuccess)
            { g_hy = NULL; g_hycap = 0; return -1; }
        g_hycap = out;
    }
    return 0;
}

extern "C" void dsv4_cuda_mmq(float *y, const float *x, const DSV4QMat *m)
{
    Res *r = find(m->w);
    if (!r || stage(r->cols, r->rows) != 0) {
        fprintf(stderr, "dsv4_cuda: mmq called for a matrix that is not "
                        "resident\n");
        abort();
    }
    memcpy(g_hx, x, (size_t)r->cols * sizeof(float));
    cudaMemcpy(g_x, g_hx, (size_t)r->cols * sizeof(float),
               cudaMemcpyHostToDevice);
    k_mmq_fp8<<<r->rows, 128>>>(g_y, g_x, r->dw, r->ds,
                                r->cols, r->blk_r, r->blk_c, r->sc);
    cudaMemcpy(g_hy, g_y, (size_t)r->rows * sizeof(float),
               cudaMemcpyDeviceToHost);
    memcpy(y, g_hy, (size_t)r->rows * sizeof(float));
}

/* nt activations in, nt results out, one weight pass on the device.
 *
 * Staging grows to nt*cols in and nt*rows out; both still cross PCIe as pinned
 * copies for the reason in stage(), and both are tiny next to the weight matrix
 * that no longer crosses at all -- 4096 floats per token against 33.5 MB. */
extern "C" void dsv4_cuda_mmq_n(float *y, const float *x, const DSV4QMat *m,
                                int nt)
{
    if (nt <= 1) { dsv4_cuda_mmq(y, x, m); return; }

    Res *r = find(m->w);
    if (!r || nt > DSV4_MAX_BATCH ||
        stage(nt * r->cols, nt * r->rows) != 0) {
        fprintf(stderr, "dsv4_cuda: mmq_n called for a matrix that is not "
                        "resident, or a batch that will not stage\n");
        abort();
    }

    const size_t xn = (size_t)nt * r->cols, yn = (size_t)nt * r->rows;
    const int threads = 128;
    const size_t shmem = (size_t)(threads / 32) * (size_t)nt * sizeof(double);

    memcpy(g_hx, x, xn * sizeof(float));
    cudaMemcpy(g_x, g_hx, xn * sizeof(float), cudaMemcpyHostToDevice);
    k_mmq_fp8_n<<<r->rows, threads, shmem>>>(g_y, g_x, r->dw, r->ds,
                                             r->cols, r->blk_r, r->blk_c,
                                             r->sc, nt);
    cudaMemcpy(g_hy, g_y, yn * sizeof(float), cudaMemcpyDeviceToHost);
    memcpy(y, g_hy, yn * sizeof(float));
}

extern "C" void dsv4_cuda_shutdown(void)
{
    for (int i = 0; i < g_nres; i++) {
        cudaFree(g_res[i].dw);
        cudaFree(g_res[i].ds);
    }
    free(g_res); g_res = NULL; g_nres = g_cap = 0;
    if (g_x) cudaFree(g_x);
    if (g_y) cudaFree(g_y);
    if (g_hx) cudaFreeHost(g_hx);
    if (g_hy) cudaFreeHost(g_hy);
    g_x = g_y = g_hx = g_hy = NULL;
    g_xcap = g_ycap = g_hxcap = g_hycap = 0;
    g_ready = 0;
}

/* A device-side dump of the E4M3 decoder, so the gate can check all 256 codes
 * against the host's table instead of assuming two implementations agree. */
__global__ void k_dump_e4m3(float *out)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < 256) out[i] = d_e4m3((unsigned char)i);
}

extern "C" int dsv4_cuda_dump_e4m3(float *host256)
{
    float *d = NULL;
    CUDA_TRY(cudaMalloc((void **)&d, 256 * sizeof(float)), "cudaMalloc");
    k_dump_e4m3<<<1, 256>>>(d);
    CUDA_TRY(cudaMemcpy(host256, d, 256 * sizeof(float),
                        cudaMemcpyDeviceToHost), "cudaMemcpy");
    cudaFree(d);
    return 0;
}
