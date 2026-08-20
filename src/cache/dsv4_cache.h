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

/* Slack so each coalesced run can be positioned at the 4096 residue its FILE
 * offset demands, which is what lets O_DIRECT land straight in the slot. At
 * most two runs in practice (scales, then weights); four runs' worth is bought
 * so the placement never has to think about it. */
#define DSV4_SLOT_SLACK (6 * 4096)

typedef struct {
    int      layer, expert;    /* key; layer < 0 means the slot is empty */
    uint64_t stamp;            /* for LRU */
    DSV4ExpertW w;             /* points into `bytes` */
    unsigned char *bytes;      /* 4096-aligned; O_DIRECT reads land here */
    /* Where each of the six tensors ended up in `bytes`, in the order
     * (w1.s, w2.s, w3.s, w1.w, w2.w, w3.w). Not a constant layout any more:
     * a run is placed wherever its file offset's alignment residue requires,
     * so this is recomputed on every load and bind_slot reads it. */
    int64_t  toff[6];
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
    /* Bytes actually allocated per slot: the payload plus DSV4_SLOT_SLACK.
     * expert_bytes stays the true payload size so the "GB read from disk"
     * accounting keeps meaning what it says. */
    int64_t  slot_bytes;

    /* One 4096-aligned staging buffer per thread, now only a FALLBACK.
     *
     * O_DIRECT needs the offset, the length AND the buffer all aligned, so a
     * read is widened outward to the enclosing window and the payload starts
     * somewhere inside it. This used to mean every run landed here and was then
     * memcpy'd into the slot -- and that copy was NOT the rounding error the
     * old comment here claimed. Measured 2026-08-20: the drive does a 12.75 MB
     * O_DIRECT read in 2.81 ms, while the engine was taking 3.60 ms per expert.
     * The ~0.8 ms difference was the memcpy, ~4 s of a 45 s run.
     *
     * So the slot is now allocated 4096-aligned with DSV4_SLOT_SLACK to spare,
     * and each run is placed at the offset whose residue matches its file
     * offset -- then the widened window is read STRAIGHT INTO THE SLOT and the
     * payload is already where it belongs. The staging buffer survives only for
     * the case where a placement would not fit, which the slack is sized to
     * prevent.
     *
     * ONE PER THREAD. Expert loads within a layer ARE issued concurrently --
     * see dsv4_cache_get_many -- so a single shared staging buffer would have
     * every reader scribbling over the others. nbounce is fixed at cache
     * construction from the OpenMP thread count. */
    unsigned char **bounce;
    int             nbounce;
    int64_t         bounce_cap;
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
/* Bytes one routed expert occupies, from config alone. Exposed because the
 * budget planner has to reserve room for topk of them BEFORE handing the trunk
 * its share -- otherwise a model whose trunk is larger than RAM (Pro's is ~50
 * GB) lets the trunk swallow the whole budget and the cache is left below the
 * minimum it needs to serve a single layer. */
int64_t dsv4_cache_expert_bytes(const DSV4Cfg *cfg);

/* What one slot actually COSTS: the payload plus the alignment slack that lets
 * an O_DIRECT read land in it without a copy. This, not expert_bytes, is the
 * figure a budget has to be divided by -- the slack is real resident memory. */
int64_t dsv4_cache_slot_bytes(const DSV4Cfg *cfg);

int  dsv4_cache_init(DSV4Cache *c, const DSV4St *st, const DSV4Cfg *cfg,
                     int64_t budget_bytes);
void dsv4_cache_free(DSV4Cache *c);

/* Fetch one expert, loading it if absent. Returns NULL only on a read failure,
 * which callers must treat as fatal: routing a token through fewer experts than
 * the model specifies is a silently different model. */
const DSV4ExpertW *dsv4_cache_get(DSV4Cache *c, int layer, int expert);

/* Fetch n experts of one layer AT ONCE, reading the misses concurrently.
 *
 * WHY THIS EXISTS, AND WHY IT IS PER LAYER
 *   Serialised one at a time, expert reads run at queue depth one: measured in
 *   situ at 2.0 GB/s against 4.4 GB/s for the same drive under a benchmark, and
 *   6.2 ms per 12.75 MB miss where an isolated read takes 2.8 ms. The fix is
 *   more requests in flight, and the only place to find them is within a layer.
 *
 *   Cross-LAYER prefetch is not merely unimplemented, it is impossible: layer
 *   L+1's router consumes layer L's output, so L+1's expert ids do not exist
 *   until L has finished. (Layers below num_hash_layers are the exception --
 *   tid2eid makes their experts a pure function of the token id.) Within one
 *   layer, though, the top-k experts are chosen together and are mutually
 *   independent, which is exactly the parallelism this takes.
 *
 * DETERMINISM. Slot selection, LRU stamps and eviction all happen in a serial
 * pass in request order, before any I/O starts; only the reads themselves run
 * in parallel. So the cache's state after a call does not depend on which read
 * finished first, and the caller still accumulates in k order. Output stays
 * bit-identical.
 *
 * out[k] is NULL for any expert that failed to load; callers must treat that as
 * fatal for the same reason dsv4_cache_get's NULL is. Returns 0 if every
 * expert loaded. */
int dsv4_cache_get_many(DSV4Cache *c, int layer, const int *experts, int n,
                        const DSV4ExpertW **out);

void dsv4_cache_reset_stats(DSV4Cache *c);
void dsv4_cache_report(const DSV4Cache *c, const char *label);

#endif /* DSV4_CACHE_H */
