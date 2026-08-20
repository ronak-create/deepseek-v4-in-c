/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cuda_stub.c - what dsv4_cuda.h means in a build without CUDA.
 *
 * The CPU path is the default and the reference, so a CPU-only build is not a
 * degraded build and nothing here is an error path. dsv4_cuda_available()
 * returning 0 is the whole interface: every caller already has to handle a
 * machine with no device, so it also handles a binary with no CUDA. */
#include <stddef.h>

#include "dsv4_cuda.h"

int    dsv4_cuda_available(void)               { return 0; }
int    dsv4_cuda_init(void)                    { return -1; }
int    dsv4_cuda_upload(const DSV4QMat *m)     { (void)m; return -1; }
int    dsv4_cuda_has(const DSV4QMat *m)        { (void)m; return 0; }
size_t dsv4_cuda_free_vram(void)               { return 0; }
void   dsv4_cuda_shutdown(void)                { }

/* The gate declares this so it can compare the device decoder against the
 * host's. Without a device there is nothing to dump, and the gate skips long
 * before it would call this. */
int dsv4_cuda_dump_e4m3(float *host256) { (void)host256; return -1; }
int dsv4_cuda_set_blocking_sync(void) { return -1; }

void dsv4_cuda_mmq(float *y, const float *x, const DSV4QMat *m)
{
    /* Unreachable: dsv4_cuda_has() is always 0 here, and callers must check it
     * before dispatching. Left as a no-op rather than an abort so that a build
     * without CUDA cannot be crashed by a caller that forgets. */
    (void)y; (void)x; (void)m;
}
