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

#include <pthread.h>
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

    /* READER POOL -- see dsv4_cache_get_many_begin.
     *
     * A small set of threads that exist only to sit inside pread(). They are
     * pthreads and not an OpenMP team on purpose: the caller is expected to be
     * running its own full-width OpenMP matmul at the same time, and an inner
     * parallel region opened from inside that would either collapse to one
     * thread or oversubscribe every core. These threads are almost always
     * blocked on the device, so they cost the matmul close to nothing.
     *
     * Created once at cache init, not per layer: a token touches 43 layers, so
     * per-batch thread creation would be ~6,700 clone() calls in a 26-token
     * run for no reason. */
    pthread_t      *rth;
    int             nrth;
    pthread_mutex_t rmx;
    pthread_cond_t  rwork, rdone;
    struct DSV4CacheBatch *rbatch;   /* the batch in flight, NULL when idle */
    int             rnext, rdone_n, rshutdown;

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
 *   Serialised one at a time, expert reads run at queue depth one, and one
 *   request in flight does not saturate the drive. `cache_bw <model> qd` sweeps
 *   depth 1..32 with expert-sized O_DIRECT reads over distinct regions of the
 *   real checkpoint. Three consecutive runs agree to within 2%:
 *
 *     QD    1     2     3     4     6     8    12    16    24    32
 *   GB/s  3.95  4.98  4.56  4.66  4.97  4.81  4.94  4.70  4.61  4.77
 *
 *   Concurrency is worth about 25%, ALL of it arrives by QD2, and the curve is
 *   flat from there to 32. This function is still the right shape -- 25% of the
 *   largest single component in the profile is worth having, and a layer's
 *   top-k is the only place independent reads exist -- but read the result
 *   correctly:
 *
 *     - An io_uring submission ring buys NOTHING here. Six OpenMP preads
 *       already sit past the knee. Worth measuring before building it.
 *     - More DRIVES is the only remaining lever on read bandwidth, and that is
 *       a hardware answer, not a software one.
 *
 *   An older version of this comment gave the in-situ rate as 2.0 GB/s against
 *   4.4 benchmarked, and implied depth would close the gap. Both halves are now
 *   obsolete: with the cache reading straight into its slot, a live run moves
 *   32.16 GB of experts in 7.8-10.5 s = 3.1-4.1 GB/s, which is 75-95% of what
 *   the sweep says this drive does at ANY depth. There is no large in-situ gap
 *   left to explain.
 *
 *   Within one layer the top-k experts are chosen together and are mutually
 *   independent, which is exactly the parallelism this takes. Cross-LAYER is
 *   three separate questions, and an earlier version of this comment collapsed
 *   them into one wrong answer ("impossible"):
 *
 *     - EXACT, on a scored layer: genuinely impossible. Layer L+1's router
 *       consumes layer L's output, so L+1's expert ids do not exist until L
 *       has finished. No amount of engineering gets around a data dependency.
 *
 *     - EXACT, on a hash layer (layer < num_hash_layers -- 3 of 43 on both
 *       released models): possible and unimplemented. tid2eid makes those
 *       experts a pure function of the token id, so the moment a token is
 *       sampled, the experts for the NEXT token's layers 0..2 are known
 *       exactly. ~7% of expert traffic; small, but free of any guess.
 *
 *     - SPECULATIVE, on any layer: possible and unimplemented. A predictor
 *       that guesses L+1 from L costs correctness nothing -- a wrong guess is
 *       a wasted read, not a wrong token -- provided it prefetches into slots
 *       that cannot evict a live entry. colibri (JustVugg/colibri) reports
 *       71.6% one-layer-ahead predictability on its models; whether that holds
 *       for DeepSeek-V4's routing is unmeasured here, and cheap to settle by
 *       replaying a --route-log through tools/sim_cache.py before writing any
 *       of it.
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

/* ---------------------------------------------------- overlapped fetching ---
 * dsv4_cache_get_many with the wait pulled apart, so a caller can compute
 * against the experts it already has while the ones it does not are read.
 *
 * WHY THIS IS WORTH THE EXTRA API
 *   Before this existed, a layer's MoE was a hard barrier: read all six
 *   experts, THEN run all six matmuls. Disk and expert matmul are the two
 *   largest movable components in the profile and they were strictly
 *   serialised, while the hit rate at a realistic budget is around 50% -- so
 *   roughly half the matmuls could have run during the reads and did not.
 *
 *   begin() does the entire serial decide pass exactly as get_many does, fills
 *   out[k] for every HIT immediately, leaves out[k] NULL for every miss, and
 *   hands the misses to the reader pool. end() waits for the pool and fills in
 *   the rest. Between the two calls the caller owns the CPU.
 *
 * WHAT IS AND IS NOT PROMISED
 *   Every decision that touches cache state -- hit or miss, which slot is
 *   evicted, what stamp each entry gets -- is still made inside begin(), in
 *   request order, before a byte is read. So cache state after end() does not
 *   depend on read completion order, exactly as before.
 *
 *   The caller must NOT accumulate expert outputs in completion order. Compute
 *   each k into its OWN accumulator and sum them in k order afterwards, or the
 *   result stops being bit-identical for a reason that has nothing to do with
 *   this cache.
 *
 *   One batch per cache at a time. begin() must be followed by end() before the
 *   next begin(); nothing else may touch the cache in between.
 *
 * A batch is an opaque handle the caller allocates (usually on the stack). */
typedef struct DSV4CacheBatch DSV4CacheBatch;
struct DSV4CacheBatch {
    int layer, n, ntodo;
    int slot_of[DSV4_MAX_TOPK];         /* slot each pending read fills   */
    int k_of[DSV4_MAX_TOPK];            /* which request that read serves */
    const DSV4Tensor *t_of[DSV4_MAX_TOPK][6];
    int expert_of[DSV4_MAX_TOPK];
    int failed[DSV4_MAX_TOPK];
    uint64_t got[DSV4_MAX_TOPK];
    /* Requests that asked for an expert another request in the same batch is
     * already reading. They cannot be answered until that read lands. */
    int ndup, dup_k[DSV4_MAX_TOPK], dup_j[DSV4_MAX_TOPK];
    int rc;
};

int dsv4_cache_get_many_begin(DSV4Cache *c, int layer, const int *experts,
                              int n, const DSV4ExpertW **out,
                              DSV4CacheBatch *b);
int dsv4_cache_get_many_end(DSV4Cache *c, const DSV4ExpertW **out,
                            DSV4CacheBatch *b);

void dsv4_cache_reset_stats(DSV4Cache *c);
void dsv4_cache_report(const DSV4Cache *c, const char *label);

#endif /* DSV4_CACHE_H */
