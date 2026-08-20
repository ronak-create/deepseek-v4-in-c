/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cuda.h - optional GPU co-processor for the DENSE TRUNK only.
 *
 * WHAT GOES TO THE GPU, AND WHY THOSE THINGS
 *   Measured on the real Flash checkpoint at 2.65 s/token:
 *
 *     attention (CSA/HCA/RoPE)   32.6%   FP8 matmuls against resident weights
 *     routed experts, matmul     24.5%   FP4, streamed from NVMe
 *     routed experts, disk       27.3%   NVMe
 *     shared expert               6.5%   FP8, resident
 *
 *   The FP8 kernels are DRAM-bandwidth-bound on the CPU, not compute-bound:
 *   bench/matmul_bw.c has them at ~18 GF/s and, unlike FP4, they did not move at
 *   all when the AVX2 work took FP4 to 2.8x. A 33.5 MB weight matrix re-read
 *   from RAM every layer is the whole cost. That is precisely the workload that
 *   improves by sitting in 448 GB/s VRAM instead of ~8 GB/s of achieved system
 *   memory bandwidth.
 *
 *   The routed experts are the opposite case and deliberately STAY on the CPU.
 *   They are 137 GB, none of it resident, streamed from disk per token. Sending
 *   them to the GPU would add a PCIe hop to a disk read and improve nothing.
 *
 * WHAT DOES NOT FIT
 *   RTX 5060 Laptop usable VRAM is 8151 MiB = 7.96 GB. Trunk 6.27 GB fits with
 *   1.69 GB to spare. Trunk + embed + lm_head is 8.34 GB and does NOT -- over by
 *   0.38 GB. So embed stays on the CPU (it is a row gather, no arithmetic) and
 *   lm_head stays too (one matmul at the very end of a token). Neither belongs
 *   on the critical path anyway.
 *
 * THE CORRECTNESS CONTRACT IS DIFFERENT HERE, AND THAT IS THE POINT
 *   Everything else in this engine is bit-identical across scalar, OpenMP and
 *   AVX2, enforced by a fixed 16-accumulator tree and -ffp-contract=off. A GPU
 *   cannot honour that: its reduction order is a function of block and warp
 *   geometry. Pretending otherwise would silently weaken the guarantee the whole
 *   codebase is arranged around. So instead:
 *
 *     - The CPU path stays the reference and the default.
 *     - --gpu is opt-in and gated SEPARATELY, on top-1 token identity over a
 *       long run plus a bounded logit delta -- never on bit equality.
 *     - No GPU result may ever regenerate a fixture.
 *
 *   Accumulation is in double on the device for the same reason the CPU does it:
 *   these are bandwidth-bound kernels, so the arithmetic precision is close to
 *   free, and it keeps the delta against the reference small enough that a real
 *   divergence stands out from rounding.
 */
#ifndef DSV4_CUDA_H
#define DSV4_CUDA_H

#include <stddef.h>

#include "dsv4.h"

#ifdef __cplusplus
extern "C" {
#endif

/* 1 when this binary was built with CUDA and a usable device is present.
 * Safe to call in a CPU-only build, where it is compiled to return 0. */
int dsv4_cuda_available(void);

/* Bring up the device. Returns 0 on success. Prints what it found, including
 * free VRAM, because whether the trunk fits is the first thing that decides
 * whether --gpu is worth using at all. */
int dsv4_cuda_init(void);

/* Copy one quantised matrix into VRAM and remember it, keyed by its host weight
 * pointer. Returns 0 on success, -1 if it would not fit or the copy failed --
 * in which case that matrix simply keeps running on the CPU, which is why the
 * failure is recoverable rather than fatal.
 *
 * Only FP8 matrices are accepted today. Trunk layer pointers are stable for the
 * life of the run because pinned layers are allocated once, which is what makes
 * a pointer-keyed table sound here. */
int dsv4_cuda_upload(const DSV4QMat *m);

/* Is this matrix resident on the device? */
int dsv4_cuda_has(const DSV4QMat *m);

/* y = m * x, on the device. Only valid when dsv4_cuda_has(m). Activations move
 * over PCIe each call -- 16 KB in, 16 KB out for a 4096-wide layer, against a
 * 33.5 MB weight read that no longer crosses the bus at all. */
void dsv4_cuda_mmq(float *y, const float *x, const DSV4QMat *m);

/* Bytes of VRAM still free, for the budget planner. */
size_t dsv4_cuda_free_vram(void);

void dsv4_cuda_shutdown(void);

#ifdef __cplusplus
}
#endif

#endif /* DSV4_CUDA_H */
