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
        double acc = 0.0;
        for (int k = lane; k < n; k += 32)
            acc = fma((double)d_e4m3(row[c0 + k]), (double)x[c0 + k], acc);
        /* Reduce the warp's 32 partials. */
        for (int off = 16; off > 0; off >>= 1)
            acc += __shfl_down_sync(0xFFFFFFFFu, acc, off);
        if (lane == 0)
            mine += acc * (double)d_e8m0(S[(size_t)(o / blk_r) * sc
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
