/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache.c - see dsv4_cache.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

#include <unistd.h>

#include "dsv4_cache.h"

/* Defined with the rest of the reader pool, below load_expert, which they call.
 * Declared here because init and free are above them. */
static void reader_pool_start(DSV4Cache *c);
static void reader_pool_stop(DSV4Cache *c);

/* An expert's six tensors, in the order they are laid out inside a slot.
 * w2 is the down projection and has the transposed shape, which is why its
 * lengths are computed separately rather than assumed equal to w1's. */
static int expert_geometry(DSV4Cache *c, const DSV4Cfg *cfg)
{
    const int64_t inter = cfg->moe_inter, hid = cfg->hidden;
    /* FP4: stored bytes are half the logical columns. Scales are 1x32 on the
     * reduce dimension. Both were verified against the released checkpoint:
     * w1.weight I8 [2048,2048] with w1.scale F8_E8M0 [2048,128]. */
    const int64_t wb[3] = { inter * (hid / 2), hid * (inter / 2), inter * (hid / 2) };
    const int64_t sb[3] = { inter * (hid / 32), hid * (inter / 32), inter * (hid / 32) };

    /* SLOT LAYOUT MIRRORS DISK LAYOUT: all three scales, then all three
     * weights. Measured on the released checkpoint, an expert's six tensors are
     * not six scattered blocks but TWO contiguous runs -- the checkpoint groups
     * by dtype, exactly as it does for the trunk:
     *
     *   layer 2, expert 0
     *     33,255,000 .. 33,779,288    0.75 MB  w1.scale w2.scale w3.scale
     *     [ ~325 MB of other tensors ]
     *    374,830,168 .. 387,413,080   12.00 MB w1.weight w2.weight w3.weight
     *
     * Interleaving them here as (w1.w, w1.s, w2.w, w2.s, ...) would force six
     * reads and a scatter. Matching disk order lets each run land in one pread
     * with no copy at all. */
    int64_t off = 0;
    for (int i = 0; i < 3; i++) { c->off_s[i] = off; c->len_s[i] = sb[i]; off += sb[i]; }
    for (int i = 0; i < 3; i++) { c->off_w[i] = off; c->len_w[i] = wb[i]; off += wb[i]; }
    c->expert_bytes = off;
    c->slot_bytes   = off + DSV4_SLOT_SLACK;
    return 0;
}

/* Point a slot's DSV4ExpertW at its own bytes. Done once per load, not per use. */
static void bind_slot(DSV4Cache *c, DSV4Slot *s)
{
    const DSV4Cfg *cfg = c->cfg;
    DSV4QMat *m[3] = { &s->w.w1, &s->w.w2, &s->w.w3 };
    const int64_t rows[3] = { cfg->moe_inter, cfg->hidden,    cfg->moe_inter };
    const int64_t cols[3] = { cfg->hidden,    cfg->moe_inter, cfg->hidden    };

    for (int i = 0; i < 3; i++) {
        memset(m[i], 0, sizeof *m[i]);
        m[i]->wdt   = DSV4_WFP4;
        m[i]->rows  = (int)rows[i];
        m[i]->cols  = (int)cols[i];
        m[i]->blk_r = 1;
        m[i]->blk_c = 32;
        m[i]->w = s->bytes + s->toff[3 + i];
        m[i]->s = s->bytes + s->toff[i];
    }
}

int64_t dsv4_cache_expert_bytes(const DSV4Cfg *cfg)
{
    DSV4Cache tmp;
    memset(&tmp, 0, sizeof tmp);
    expert_geometry(&tmp, cfg);
    return tmp.expert_bytes;
}

int64_t dsv4_cache_slot_bytes(const DSV4Cfg *cfg)
{
    DSV4Cache tmp;
    memset(&tmp, 0, sizeof tmp);
    expert_geometry(&tmp, cfg);
    return tmp.slot_bytes;
}

int dsv4_cache_init(DSV4Cache *c, const DSV4St *st, const DSV4Cfg *cfg,
                    int64_t budget_bytes)
{
    memset(c, 0, sizeof *c);
    c->st = st; c->cfg = cfg;
    expert_geometry(c, cfg);

    /* A cache smaller than topk cannot serve a whole layer, and it is worth
     * saying so -- but it is a WARNING here, not an error.
     *
     * The constraint belongs to dsv4_cache_get_many, which fetches a layer's
     * top-k together and will not evict a slot it has already claimed for the
     * same call. A caller using dsv4_cache_get one expert at a time is fine
     * with two slots, and the cache gate depends on exactly that to test
     * eviction order. Enforcing it here broke that gate and told me the check
     * was in the wrong place.
     *
     * Prevention lives in the budget planner (dsv4_run.c reserves topk experts
     * before the trunk takes its share) and the last line of defence is in
     * moe(), which aborts rather than emit a token routed through fewer experts
     * than the model specifies. That failure used to be silent: at --budget 3
     * the cache got 5 slots against a top-k of 6 and the fourth generated token
     * quietly changed. */
    if ((int64_t)cfg->topk * c->expert_bytes > budget_bytes)
        fprintf(stderr,
                "dsv4_cache: %lld slots cannot hold one layer's %d experts at "
                "once;\n"
                "  dsv4_cache_get_many will fail. Single-expert use is fine.\n",
                (long long)(budget_bytes / c->expert_bytes), cfg->topk);

    if (budget_bytes < c->expert_bytes) {
        fprintf(stderr, "dsv4_cache: budget %lld bytes cannot hold one expert "
                        "(%lld bytes). Every access would miss and the same\n"
                        "  bytes would be read six times per layer.\n",
                (long long)budget_bytes, (long long)c->expert_bytes);
        return -1;
    }

    /* The largest coalesced run is the three weights; the widened window can
     * overhang by up to one alignment unit at each end. */
    int64_t widest_run = 0;
    for (int i = 0; i < 3; i++) widest_run += c->len_w[i];
    c->bounce_cap = widest_run + 2 * 4096;
    c->nbounce = omp_get_max_threads();
    if (c->nbounce < 1) c->nbounce = 1;
    c->bounce = (unsigned char **)calloc((size_t)c->nbounce, sizeof *c->bounce);
    if (!c->bounce) return -1;
    for (int b = 0; b < c->nbounce; b++)
    if (posix_memalign((void **)&c->bounce[b], 4096, (size_t)c->bounce_cap) != 0) {
        fprintf(stderr, "dsv4_cache: could not allocate a %lld byte staging "
                        "buffer\n", (long long)c->bounce_cap);
        return -1;
    }
    c->direct = (st->dfd && st->nshard > 0 && st->dfd[0] >= 0);
    if (!c->direct)
        fprintf(stderr, "dsv4_cache: O_DIRECT unavailable; expert reads go "
                        "through the page cache, which at a 148 GB working set "
                        "evicts more than it saves\n");

    /* Charged against slot_bytes, not expert_bytes: the alignment slack is
     * real memory and the budget is a promise about peak RSS. */
    c->nslot = (int)(budget_bytes / c->slot_bytes);

    /* SAY SO WHEN THE CACHE CANNOT POSSIBLY WORK.
     *
     * One forward pass touches n_layers * topk experts, in a fixed cyclic order.
     * Under LRU, a hit needs the entry to survive from one visit to a layer
     * until the next -- and between those visits the engine loads every OTHER
     * layer's experts. So if the cache holds fewer than a full pass's working
     * set, each entry is guaranteed to be evicted before it is asked for again
     * and the hit rate is exactly zero, no matter how skewed the routing is.
     *
     * This is not hypothetical. At the old fixed 1/4 budget split, Flash ran
     * 160 slots against a 258-expert working set and returned 0 hits on 2,580
     * requests while reading 32 GB. Silently useless memory is worse than no
     * memory, so the engine now names the threshold it is under. */
    const int64_t ws = (int64_t)cfg->n_layers * cfg->topk;
    if ((int64_t)c->nslot < ws)
        fprintf(stderr,
                "dsv4_cache: %d slots (%.2f GB) is below one forward pass's "
                "working set of\n"
                "  %lld experts (%.2f GB). LRU cannot hit at this size -- every "
                "entry is evicted\n"
                "  before its layer comes round again. --budget must cover "
                "the trunk AND this,\n  so raise it by at least %.1f GB.\n",
                c->nslot, (double)budget_bytes / 1073741824.0,
                (long long)ws, (double)(ws * c->expert_bytes) / 1073741824.0,
                (double)(ws * c->expert_bytes) / 1073741824.0);

    c->slot = (DSV4Slot *)calloc((size_t)c->nslot, sizeof(DSV4Slot));
    if (!c->slot) return -1;

    for (int i = 0; i < c->nslot; i++) {
        c->slot[i].layer = -1;
        /* 4096-aligned so a widened O_DIRECT window can be read straight in. */
        if (posix_memalign((void **)&c->slot[i].bytes, 4096,
                           (size_t)c->slot_bytes) != 0) {
            c->slot[i].bytes = NULL;
            dsv4_cache_free(c); return -1;
        }
        /* Until the first load, the compact layout is the honest description
         * of an empty slot -- bind_slot needs SOMETHING coherent to point at. */
        for (int k = 0; k < 3; k++) {
            c->slot[i].toff[k]     = c->off_s[k];
            c->slot[i].toff[3 + k] = c->off_w[k];
        }
        bind_slot(c, &c->slot[i]);
    }

    reader_pool_start(c);
    return 0;
}

void dsv4_cache_free(DSV4Cache *c)
{
    if (!c || !c->slot) return;
    /* Readers first, and before anything they point at is freed. */
    reader_pool_stop(c);
    for (int i = 0; i < c->nslot; i++) free(c->slot[i].bytes);
    free(c->slot);
    if (c->bounce) {
        for (int b = 0; b < c->nbounce; b++) free(c->bounce[b]);
        free(c->bounce);
    }
    memset(c, 0, sizeof *c);
}

/* Read one expert's six tensors into a slot.
 *
 * They are read individually here rather than as one contiguous range, because
 * the checkpoint does NOT store them adjacently: measured on the real shards,
 * each layer's tensors are grouped by dtype, so an expert's weights and its
 * scales sit far apart. Coalescing is the packer's job, not the reader's -- see
 * the note in dsv4_cache.h about what coalescing is actually worth. */
/* RESOLVE BEFORE EVICTING.
 *
 * Every tensor is located and size-checked before any slot is chosen, so a
 * request that cannot be satisfied costs nothing. The first version of this
 * file read straight into the victim slot and, on failure, marked it empty --
 * which threw away a perfectly good cached expert because a DIFFERENT one was
 * missing. The gate caught it. Six hash lookups are trivial beside a 13 MB
 * read, so there is no reason to be optimistic here. */
static int resolve_expert(DSV4Cache *c, int layer, int expert,
                          const DSV4Tensor *out[6])
{
    static const char *wn[3] = { "w1", "w2", "w3" };
    char name[224];

    for (int i = 0; i < 3; i++) {
        snprintf(name, sizeof name, "layers.%d.ffn.experts.%d.%s.weight",
                 layer, expert, wn[i]);
        const DSV4Tensor *t = dsv4_st_find(c->st, name);
        if (!t || t->nbytes != c->len_w[i]) {
            fprintf(stderr, "dsv4_cache: %s %s (%lld bytes, expected %lld)\n",
                    name, t ? "has the wrong size" : "is missing",
                    t ? (long long)t->nbytes : 0LL, (long long)c->len_w[i]);
            return -1;
        }
        out[i * 2] = t;

        snprintf(name, sizeof name, "layers.%d.ffn.experts.%d.%s.scale",
                 layer, expert, wn[i]);
        const DSV4Tensor *sc = dsv4_st_find(c->st, name);
        if (!sc || sc->nbytes != c->len_s[i]) {
            fprintf(stderr, "dsv4_cache: %s %s (%lld bytes, expected %lld)\n",
                    name, sc ? "has the wrong size" : "is missing",
                    sc ? (long long)sc->nbytes : 0LL, (long long)c->len_s[i]);
            return -1;
        }
        out[i * 2 + 1] = sc;
    }
    return 0;
}

/* Read six already-resolved tensors into a slot, COALESCING adjacent ones.
 *
 * The six arrive as two contiguous runs on disk (see expert_geometry), and the
 * slot is laid out to match, so each run is one pread straight into place. Six
 * separate reads measured 1.41 GB/s against 4.5 GB/s for a single 13 MB read --
 * the cost is four extra syscalls and four seeks per expert, paid 258 times a
 * token.
 *
 * The coalescing is DISCOVERED, not assumed: entries are sorted by file offset
 * and merged only where one ends exactly where the next begins. If a future
 * checkpoint scatters them, this quietly degrades to six reads and stays
 * correct.
 *
 * A short read here is a real I/O failure, not a lookup miss, and still leaves
 * the slot unusable -- so the caller must clear the key. */
static int load_expert(DSV4Cache *c, DSV4Slot *s, const DSV4Tensor *t[6],
                       unsigned char *bounce, uint64_t *bytes_read)
{
    /* (file offset, length, shard, which tensor), in slot order: s,s,s,w,w,w.
     * `idx` survives the sort because the slot offset is no longer a constant
     * derived from position -- it is assigned during placement below. */
    struct { int64_t foff, len; int shard, idx; } r[6];
    for (int i = 0; i < 3; i++) {
        r[i].foff  = t[i * 2 + 1]->off;
        r[i].len   = t[i * 2 + 1]->nbytes; r[i].shard = t[i * 2 + 1]->shard;
        r[i].idx   = i;
        r[3 + i].foff  = t[i * 2]->off;
        r[3 + i].len   = t[i * 2]->nbytes; r[3 + i].shard = t[i * 2]->shard;
        r[3 + i].idx   = 3 + i;
    }
    /* six elements: insertion sort is the right tool */
    for (int i = 1; i < 6; i++)
        for (int j = i; j > 0 && r[j].foff < r[j - 1].foff; j--) {
            const typeof(r[0]) tmp = r[j]; r[j] = r[j - 1]; r[j - 1] = tmp;
        }

    int64_t cursor = 0;
    int i = 0;
    while (i < 6) {
        int j = i;
        /* Extend while the FILE stays contiguous. The slot no longer has to
         * agree: the run is placed as a unit, so its members stay adjacent by
         * construction. */
        while (j + 1 < 6
               && r[j + 1].shard == r[i].shard
               && r[j + 1].foff  == r[j].foff + r[j].len)
            j++;

        int64_t len = 0;
        for (int k = i; k <= j; k++) len += r[k].len;

        /* PLACE THE RUN SO O_DIRECT CAN LAND IT WITHOUT A COPY.
         *
         * dsv4_st_read_aligned widens to the enclosing 4096 window and reports
         * the payload as sitting `pad` bytes into whatever buffer it was given.
         * That buffer must itself be 4096-aligned. So choose the run's slot
         * offset `soff` to have the SAME residue as its file offset: then
         * (bytes + soff - pad) is 4096-aligned, the window is read straight
         * into the slot, and the payload is already in place.
         *
         * Every expert tensor in the released checkpoint is unaligned -- the
         * data section starts at a fixed odd offset -- so this is the normal
         * path, not a special case. */
        const int64_t pad  = r[i].foff & (int64_t)4095;
        /* The window starts `pad` bytes BEFORE the payload, so the payload must
         * sit at least `pad` past the previous run -- otherwise this read reaches
         * back and clobbers the tail of the run already loaded. Reserving only
         * soff >= cursor is not enough, and the damage is silent: the bytes are
         * wrong, the logits drift, routing collapses onto a handful of experts
         * and the cache hit rate goes UP. It cost a 95% hit rate that looked
         * like a win. */
        const int64_t base = cursor + pad;
        const int64_t soff = base + ((pad - base) & (int64_t)4095);
        const int64_t wlen = (pad + len + 4095) & ~(int64_t)4095;
        const int use_slot = (soff - pad + wlen) <= c->slot_bytes;
        int64_t place;

        /* O_DIRECT through the staging buffer. The page cache is not merely
         * unhelpful for these reads, it is harmful: a 148 GB expert working set
         * cannot be cached in 23 GB, so every buffered read evicts pages that
         * WOULD have been reused to hold 12 MB that will not be. Measured cold:
         * 1.00 GB/s buffered against 4.4 GB/s raw O_DIRECT on this disk. */
        if (c->direct && use_slot) {
            /* The fast path: no copy at all. */
            int64_t poff = 0;
            const int64_t n = dsv4_st_read_aligned(
                c->st, r[i].shard, r[i].foff, len,
                s->bytes + soff - pad, c->slot_bytes - (soff - pad), &poff);
            if (n != len || poff != pad) return -1;
            place = soff;
        } else if (c->direct && len <= c->bounce_cap - 2 * 4096) {
            /* fall back to the compact position; no alignment needed here */
            /* Fallback, kept so a checkpoint that scattered these tensors into
             * more runs than the slack allows still loads correctly. */
            int64_t poff = 0;
            const int64_t n = dsv4_st_read_aligned(c->st, r[i].shard, r[i].foff,
                                                   len, bounce, c->bounce_cap,
                                                   &poff);
            if (n != len) return -1;
            if (cursor + len > c->slot_bytes) return -1;
            memcpy(s->bytes + cursor, bounce + poff, (size_t)len);
            place = cursor;
        } else {
            if (cursor + len > c->slot_bytes) return -1;
            place = cursor;
            int64_t got = 0;
            while (got < len) {
                const ssize_t n = pread(c->st->fd[r[i].shard],
                                        s->bytes + place + got,
                                        (size_t)(len - got),
                                        (off_t)(r[i].foff + got));
                if (n <= 0) return -1;
                got += n;
            }
        }

        /* Record where each tensor of this run actually landed. */
        int64_t at = place;
        for (int k = i; k <= j; k++) { s->toff[r[k].idx] = at; at += r[k].len; }
        cursor = place + len;

        *bytes_read += (uint64_t)len;
        i = j + 1;
    }

    /* The layout is per-load now, so the matrix pointers are rebound here
     * rather than once at construction. Touches only this slot, which is what
     * makes it safe inside dsv4_cache_get_many's parallel read pass. */
    bind_slot(c, s);
    return 0;
}

const DSV4ExpertW *dsv4_cache_get(DSV4Cache *c, int layer, int expert)
{
    c->clock++;
    if (c->route_log) fprintf(c->route_log, "%d %d\n", layer, expert);

    for (int i = 0; i < c->nslot; i++) {
        if (c->slot[i].layer == layer && c->slot[i].expert == expert) {
            c->slot[i].stamp = c->clock;
            c->hits++;
            return &c->slot[i].w;
        }
    }

    /* Resolve FIRST, so a request that cannot be satisfied never disturbs the
     * cache. Only once every tensor is known good is a victim chosen. */
    const DSV4Tensor *t[6];
    if (resolve_expert(c, layer, expert, t) != 0) return NULL;

    /* Take an empty slot if there is one, else the least recently used.
     * Scanning is fine at these slot counts: a 24 GB budget is ~1,800 slots and
     * the scan is trivial beside a 13 MB read. */
    int victim = -1;
    uint64_t oldest = UINT64_MAX;
    for (int i = 0; i < c->nslot; i++) {
        if (c->slot[i].layer < 0) { victim = i; break; }
        if (c->slot[i].stamp < oldest) { oldest = c->slot[i].stamp; victim = i; }
    }
    if (c->slot[victim].layer >= 0) c->evictions++;

    if (load_expert(c, &c->slot[victim], t, c->bounce[0], &c->bytes_read) != 0) {
        c->slot[victim].layer = -1;      /* do not leave a half-read slot live */
        return NULL;
    }
    c->slot[victim].layer  = layer;
    c->slot[victim].expert = expert;
    c->slot[victim].stamp  = c->clock;
    c->misses++;
    return &c->slot[victim].w;
}

/* ------------------------------------------------------------ reader pool ---
 * Threads whose entire job is to sit inside pread().
 *
 * pthreads and not an OpenMP team, deliberately. The caller of
 * dsv4_cache_get_many_begin is expected to be running a full-width OpenMP
 * matmul between begin and end, and an inner parallel region opened from
 * inside that would either collapse to a single thread -- taking the reads
 * back to queue depth one, which bench/cache_bw.c measures as a 25% loss --
 * or oversubscribe every core against the very matmul it is meant to hide
 * behind.
 *
 * A pool and not a thread per batch: 43 layers times 26 tokens is ~1,100
 * batches in a short run, and there is no reason to spend a clone() on each.
 */
static void *reader_main(void *arg)
{
    DSV4Cache *c = (DSV4Cache *)arg;

    pthread_mutex_lock(&c->rmx);
    for (;;) {
        while (!c->rshutdown && (!c->rbatch || c->rnext >= c->rbatch->ntodo))
            pthread_cond_wait(&c->rwork, &c->rmx);
        if (c->rshutdown) break;

        DSV4CacheBatch *b = c->rbatch;
        const int j = c->rnext++;
        /* Bounce index is the JOB, not the thread. Jobs in one batch are
         * distinct and never outnumber the buffers, so no two readers can
         * land in the same staging buffer. */
        const int bi = j < c->nbounce ? j : 0;
        pthread_mutex_unlock(&c->rmx);

        uint64_t got = 0;
        const int bad = load_expert(c, &c->slot[b->slot_of[j]], b->t_of[j],
                                    c->bounce[bi], &got) != 0;

        pthread_mutex_lock(&c->rmx);
        b->failed[j] = bad;
        b->got[j]    = got;
        if (++c->rdone_n >= b->ntodo) pthread_cond_broadcast(&c->rdone);
    }
    pthread_mutex_unlock(&c->rmx);
    return NULL;
}

/* Start the pool. Failure here is NOT fatal: nrth stays 0 and every batch falls
 * back to reading on the calling thread, which is exactly the behaviour this
 * cache had before the pool existed. An engine that runs slower beats one that
 * refuses to run. */
static void reader_pool_start(DSV4Cache *c)
{
    /* HOW WIDE SHOULD THE POOL BE? topk. THE PLAUSIBLE ARGUMENT FOR LESS IS
     * WRONG, AND IT WAS MEASURED RATHER THAN ARGUED.
     *
     * The argument: bench/cache_bw.c's queue-depth sweep says this drive
     * plateaus by QD2-4 and is flat to 32, so readers past the plateau return
     * no bandwidth -- while the A/B that landed the overlap measured the expert
     * matmul 34% SLOWER during streaming, DMA contending with a memory-bound
     * FP4 kernel. So a narrower pool should buy back matmul time for free.
     *
     * It does not. Five interleaved rounds on the real checkpoint,
     * --gen 15 --budget 16 --gpu, varying DSV4_READERS:
     *
     *   readers            1      2      3      4      6
     *   exposed I/O ms   667.9  488.5  418.3  375.4  351.3
     *   s/token           1.72   1.54   1.40   1.39   1.39
     *
     * Widest wins, and nothing is bought back below it.
     *
     * The sweep is not wrong; it answers a different question. It measures
     * THROUGHPUT over 48 reads. What end() waits for is the MAKESPAN of one
     * batch of at most six, against a fixed amount of hit-matmul to hide it
     * behind. Halve the readers and a six-miss batch takes three serial rounds
     * instead of one; the drive is no busier, but the caller waits longer. A
     * queue-depth curve cannot see that, because it never has a deadline.
     *
     * DSV4_READERS stays as the knob that produced the table, and for drives
     * whose curve is not this one. */
    int want = DSV4_MAX_TOPK;
    const char *env = getenv("DSV4_READERS");
    if (env) {
        const int n = atoi(env);
        if (n > 0) want = n;
    }
    if (want > DSV4_MAX_TOPK) want = DSV4_MAX_TOPK;
    if (want > c->nbounce) want = c->nbounce;
    if (want < 1) return;

    if (pthread_mutex_init(&c->rmx, NULL) != 0) return;
    if (pthread_cond_init(&c->rwork, NULL) != 0) {
        pthread_mutex_destroy(&c->rmx);
        return;
    }
    if (pthread_cond_init(&c->rdone, NULL) != 0) {
        pthread_cond_destroy(&c->rwork);
        pthread_mutex_destroy(&c->rmx);
        return;
    }

    c->rth = (pthread_t *)calloc((size_t)want, sizeof *c->rth);
    if (!c->rth) return;
    for (int i = 0; i < want; i++) {
        if (pthread_create(&c->rth[i], NULL, reader_main, c) != 0) break;
        c->nrth++;
    }
    if (c->nrth == 0) { free(c->rth); c->rth = NULL; }
}

static void reader_pool_stop(DSV4Cache *c)
{
    if (!c->rth) return;
    pthread_mutex_lock(&c->rmx);
    c->rshutdown = 1;
    pthread_cond_broadcast(&c->rwork);
    pthread_mutex_unlock(&c->rmx);
    for (int i = 0; i < c->nrth; i++) pthread_join(c->rth[i], NULL);
    free(c->rth);
    c->rth  = NULL;
    c->nrth = 0;
    pthread_cond_destroy(&c->rwork);
    pthread_cond_destroy(&c->rdone);
    pthread_mutex_destroy(&c->rmx);
}

/* ---------------------------------------------------------------------------
 * The serial decide pass, unchanged in substance from what dsv4_cache_get_many
 * used to do inline: hits resolved, victims chosen, stamps written, all in
 * request order and all before a byte is read. The misses now go into a batch
 * instead of a local array, and the reads are handed to the pool rather than
 * waited on.
 */
int dsv4_cache_get_many_begin(DSV4Cache *c, int layer, const int *experts,
                              int n, const DSV4ExpertW **out,
                              DSV4CacheBatch *b)
{
    int claimed[DSV4_MAX_TOPK], nclaim = 0;

    b->layer = layer;
    b->n     = n;
    b->ntodo = 0;
    b->ndup  = 0;
    b->rc    = 0;

    if (n > DSV4_MAX_TOPK) { b->rc = -1; return -1; }

    for (int k = 0; k < n; k++) {
        const int e = experts[k];
        out[k] = NULL;
        c->clock++;
        if (c->route_log) fprintf(c->route_log, "%d %d\n", layer, e);

        int found = -1;
        for (int i = 0; i < c->nslot; i++)
            if (c->slot[i].layer == layer && c->slot[i].expert == e) {
                found = i; break;
            }
        if (found >= 0) {
            c->slot[found].stamp = c->clock;
            c->hits++;
            out[k] = &c->slot[found].w;      /* resident: safe to use NOW */
            claimed[nclaim++] = found;
            continue;
        }

        /* A duplicate request inside one call must not take a second slot --
         * top-k yields distinct experts today, but nothing here should depend
         * on that. It also must not publish out[k] yet: the slot it points at
         * is IN FLIGHT, and a caller computing between begin and end would
         * read a half-loaded expert. Recorded, and filled in by end(). */
        int dup = -1;
        for (int j = 0; j < b->ntodo; j++)
            if (b->expert_of[j] == e) { dup = j; break; }
        if (dup >= 0) {
            c->slot[b->slot_of[dup]].stamp = c->clock;
            c->hits++;
            b->dup_k[b->ndup] = k;
            b->dup_j[b->ndup] = dup;
            b->ndup++;
            continue;
        }

        const DSV4Tensor *t[6];
        if (resolve_expert(c, layer, e, t) != 0) { b->rc = -1; continue; }

        int victim = -1;
        uint64_t oldest = UINT64_MAX;
        for (int i = 0; i < c->nslot; i++) {
            int taken = 0;
            for (int j = 0; j < nclaim; j++) if (claimed[j] == i) { taken = 1; break; }
            if (taken) continue;             /* never evict this call's own work */
            if (c->slot[i].layer < 0) { victim = i; break; }
            if (c->slot[i].stamp < oldest) { oldest = c->slot[i].stamp; victim = i; }
        }
        if (victim < 0) { b->rc = -1; continue; }   /* fewer slots than top-k */
        if (c->slot[victim].layer >= 0) c->evictions++;

        c->slot[victim].layer = -1;          /* in flight: not a valid entry yet */
        c->slot[victim].stamp = c->clock;
        claimed[nclaim++] = victim;
        b->slot_of[b->ntodo]   = victim;
        b->k_of[b->ntodo]      = k;
        b->expert_of[b->ntodo] = e;
        memcpy(b->t_of[b->ntodo], t, sizeof t);
        b->failed[b->ntodo] = 0;
        b->got[b->ntodo]    = 0;
        b->ntodo++;
    }

    if (b->ntodo > 0 && c->nrth > 0) {
        pthread_mutex_lock(&c->rmx);
        c->rbatch  = b;
        c->rnext   = 0;
        c->rdone_n = 0;
        pthread_cond_broadcast(&c->rwork);
        pthread_mutex_unlock(&c->rmx);
    }
    return b->rc;
}

int dsv4_cache_get_many_end(DSV4Cache *c, const DSV4ExpertW **out,
                            DSV4CacheBatch *b)
{
    if (b->ntodo > 0) {
        if (c->nrth > 0) {
            pthread_mutex_lock(&c->rmx);
            while (c->rdone_n < b->ntodo) pthread_cond_wait(&c->rdone, &c->rmx);
            c->rbatch = NULL;
            pthread_mutex_unlock(&c->rmx);
        } else {
            /* No pool. Still concurrent, just not overlapped with the caller --
             * exactly what this function did before the pool existed. */
#pragma omp parallel for schedule(dynamic, 1) if (b->ntodo > 1)
            for (int j = 0; j < b->ntodo; j++) {
                int bi = omp_get_thread_num();
                if (bi >= c->nbounce) bi = 0;
                b->failed[j] = load_expert(c, &c->slot[b->slot_of[j]],
                                           b->t_of[j], c->bounce[bi],
                                           &b->got[j]) != 0;
            }
        }
    }

    /* ---- serial pass: publish -------------------------------------- */
    for (int j = 0; j < b->ntodo; j++) {
        DSV4Slot *sl = &c->slot[b->slot_of[j]];
        c->bytes_read += b->got[j];
        if (b->failed[j]) {
            sl->layer = -1;                  /* no half-read slot goes live */
            b->rc = -1;
            continue;
        }
        sl->layer  = b->layer;
        sl->expert = b->expert_of[j];
        c->misses++;
        out[b->k_of[j]] = &sl->w;
    }
    for (int d = 0; d < b->ndup; d++) {
        const int j = b->dup_j[d];
        if (!b->failed[j]) out[b->dup_k[d]] = &c->slot[b->slot_of[j]].w;
    }
    return b->rc;
}

int dsv4_cache_get_many(DSV4Cache *c, int layer, const int *experts, int n,
                        const DSV4ExpertW **out)
{
    DSV4CacheBatch b;
    const int rc  = dsv4_cache_get_many_begin(c, layer, experts, n, out, &b);
    const int rc2 = dsv4_cache_get_many_end(c, out, &b);
    return rc != 0 ? rc : rc2;
}

void dsv4_cache_reset_stats(DSV4Cache *c)
{
    c->hits = c->misses = c->evictions = c->bytes_read = 0;
}

void dsv4_cache_report(const DSV4Cache *c, const char *label)
{
    const uint64_t n = c->hits + c->misses;
    printf("expert cache %s: %d slots x %.2f MB = %.2f GB\n", label, c->nslot,
           (double)c->expert_bytes / 1048576.0,
           (double)c->nslot * (double)c->expert_bytes / 1073741824.0);
    printf("  %llu requests, %llu hits (%.1f%%), %llu misses, %llu evictions\n",
           (unsigned long long)n, (unsigned long long)c->hits,
           n ? 100.0 * (double)c->hits / (double)n : 0.0,
           (unsigned long long)c->misses, (unsigned long long)c->evictions);
    printf("  %.2f GB read from disk\n",
           (double)c->bytes_read / 1073741824.0);
}
