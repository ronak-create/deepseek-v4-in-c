/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache.h - LRU cache for the streamed routed experts.
 *
 * Ported in shape from kimi-k3-in-c/src/cache/k3_cache.c (Apache-2.0).
 *
 * WHY LRU HERE AND NOT FOR THE TRUNK
 *   The trunk is walked in a fixed cyclic order every token, which is the
 *   classic LRU pathology: with fewer slots than layers the next thing needed is
 *   always the thing just evicted, and the hit rate is zero however much memory
 *   is added. So the trunk pins a prefix instead.
 *
 *   Routed experts are the opposite case. Which six of 256 fire is data
 *   dependent and skewed, so recency genuinely predicts reuse and every extra
 *   gigabyte buys hit rate.
 *
 * MEASURED SIZES (DeepSeek-V4-Flash, from the real checkpoint)
 *   one expert          13,369,344 B  (w1/w2/w3 weight + scale, six tensors)
 *   per layer           256 experts = 3.19 GB
 *   whole model         43 layers    = 137 GB, none of it resident
 *   per token           6 x 43 x 13,369,344 = 3.21 GB read
 *
 * READ SIZE MATTERS, BUT LESS THAN FIRST CLAIMED. Measured on the target NVMe
 * while the disk was also being written:
 *      4.46 MB (one matrix)      4.0 GB/s
 *     13.37 MB (one expert)      4.5 GB/s
 *     26.74 MB (two experts)     4.7 GB/s
 *     80.22 MB (a layer's six)   4.9 GB/s
 *   So coalescing an expert's six tensors into one read is worth ~12%, and
 *   batching a whole layer's six experts ~22%. Worth doing, not decisive.
 */
#ifndef DSV4_CACHE_H
#define DSV4_CACHE_H

#include <stdio.h>

#include "dsv4.h"
#include "dsv4_st.h"
#include "dsv4_bind.h"

typedef struct {
    int      layer, expert;    /* key; layer < 0 means the slot is empty */
    uint64_t stamp;            /* for LRU */
    DSV4ExpertW w;             /* points into `bytes` */
    unsigned char *bytes;
} DSV4Slot;

typedef struct {
    const DSV4St  *st;
    const DSV4Cfg *cfg;
    DSV4Slot      *slot;
    int            nslot;
    int64_t        expert_bytes;
    uint64_t       clock;

    /* Offsets of the six tensors inside a slot, computed once. */
    int64_t  off_w[3], off_s[3];
    int64_t  len_w[3], len_s[3];

    /* One 4096-aligned staging buffer for O_DIRECT reads.
     *
     * O_DIRECT needs the offset, the length AND the buffer all aligned, so a
     * read is widened outward to the enclosing window and the payload starts
     * somewhere inside it. The slot cannot receive that directly -- its run
     * offsets are not aligned -- so the window lands here and the payload is
     * copied across. One 12 MB memcpy at ~10 GB/s costs ~1 ms against the ~10 ms
     * the unbuffered read saves.
     *
     * NOT thread-safe: one buffer per cache. If expert loads are ever issued
     * concurrently this must become per-thread. */
    unsigned char *bounce;
    int64_t        bounce_cap;
    int            direct;      /* 1 when the shards opened O_DIRECT */

    /* Optional routing log: every (layer, expert) request in issue order, one
     * per line. Purely diagnostic and off unless a path is given. Written for
     * one reason -- the cache policy should be chosen from a measured access
     * trace, not from an argument about what routing probably looks like. */
    FILE *route_log;

    /* stats */
    uint64_t hits, misses, evictions;
    uint64_t bytes_read;
} DSV4Cache;

/* budget_bytes decides the slot count. Returns 0 on success.
 * A budget below one expert is an error: a cache that cannot hold a single
 * entry would miss on every access and read the same bytes six times a layer. */
int  dsv4_cache_init(DSV4Cache *c, const DSV4St *st, const DSV4Cfg *cfg,
                     int64_t budget_bytes);
void dsv4_cache_free(DSV4Cache *c);

/* Fetch one expert, loading it if absent. Returns NULL only on a read failure,
 * which callers must treat as fatal: routing a token through fewer experts than
 * the model specifies is a silently different model. */
const DSV4ExpertW *dsv4_cache_get(DSV4Cache *c, int layer, int expert);

void dsv4_cache_reset_stats(DSV4Cache *c);
void dsv4_cache_report(const DSV4Cache *c, const char *label);

#endif /* DSV4_CACHE_H */
