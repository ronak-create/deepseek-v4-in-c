/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_trunk.h - stream the dense trunk, so RAM becomes a dial instead of a floor.
 *
 * Ported in design from kimi-k3-in-c/src/io/k3_trunk.c (Apache-2.0).
 *
 * WHY STREAM IT
 *   Flash's trunk is 6.37 GB (measured: 21 indexed layers at 165.47 MB, 20
 *   compressed at 139.28, 2 dense at 131.03) and Pro's is ~50 GB. Holding it
 *   resident is the floor that decides which machine can run the model at all.
 *   Streaming turns that floor into a dial: pin as many layers as the budget
 *   allows, read the rest.
 *
 * WHY LRU WOULD BE THE WORST POSSIBLE POLICY HERE
 *   The trunk is walked 0, 1, ... N-1 on EVERY token. That cyclic scan is the
 *   classic LRU pathology: with fewer slots than layers, the next layer needed
 *   is always the one just evicted, and the hit rate is zero however much
 *   memory is added. So this PINS a prefix and streams the rest through a small
 *   ring. Pin K layers and the hit rate is exactly K/N, deterministically, and
 *   every extra gigabyte buys its fair share.
 *
 *   The routed-expert cache is the opposite case and does use LRU: which six of
 *   256 experts fire is data dependent, so recency genuinely predicts reuse.
 *
 * WHY THE READ IS AFFORDABLE
 *   Unlike expert routing, the trunk's access order is FIXED, so the next read
 *   is always known before it is needed. Measured on this machine: one pread
 *   per layer from a packed trunk runs at 4.80 GB/s with O_DIRECT.
 *
 * LAYOUT
 *   tools/pack_trunk.py writes trunk.bin, where each layer is ONE contiguous
 *   run at a 4096-aligned offset, and trunk.json, which records every tensor's
 *   offset WITHIN its run. The offsets are recorded rather than derived because
 *   the source checkpoint splits a layer's trunk across two runs with the
 *   routed experts in between -- see the packer for the measurement.
 */
#ifndef DSV4_TRUNK_H
#define DSV4_TRUNK_H

#include "dsv4.h"
#include "dsv4_bind.h"

#define DSV4_TRUNK_ALIGN 4096   /* pack_trunk.py pads runs so O_DIRECT works */

typedef struct {
    char    *name;
    int64_t  off;          /* byte offset WITHIN the layer run */
    int64_t  nbytes;
    int      dtype;        /* DSV4Dtype */
} DSV4TrunkTensor;

typedef struct {
    int64_t          file_off;   /* offset in trunk.bin */
    int64_t          nbytes;
    DSV4TrunkTensor *t;
    int              nt;
} DSV4TrunkLayer;

typedef struct {
    int              fd;
    int              direct;     /* 1 when opened O_DIRECT */
    int              n_layers;
    DSV4TrunkLayer  *lay;
    char            *json_arena; /* backs every tensor name; freed on close */
    char            *json_text;

    unsigned char  **pin;        /* [npin] one exact allocation per pinned layer */
    unsigned char   *ring;       /* [nslot] uniform slots                        */
    int64_t          slot_bytes;
    int              nslot, npin;
    int             *layer_of;   /* [nslot] which layer occupies each slot       */
    int32_t         *slot_of;    /* [n_layers], -1 when not resident             */
    int              next_slot;

    unsigned char   *widen;      /* one widen area, reused per bind              */
    size_t           widen_cap;

    uint64_t         hits, misses, bytes_read;
    double           load_seconds;
} DSV4Trunk;

/* budget_bytes sizes the pinned prefix and the ring. Returns 0 on success. */
int  dsv4_trunk_open(DSV4Trunk *tr, const char *dir, const DSV4Cfg *c,
                     int64_t budget_bytes);
void dsv4_trunk_close(DSV4Trunk *tr);

/* Make layer L resident and point b's weight pointers at it. */
int  dsv4_trunk_bind(DSV4Trunk *tr, const DSV4Cfg *c, int L, DSV4LayerBind *b);

void dsv4_trunk_report(const DSV4Trunk *tr, const char *label);

#endif /* DSV4_TRUNK_H */
