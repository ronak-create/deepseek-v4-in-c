/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cache.c - see dsv4_cache.h. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <unistd.h>

#include "dsv4_cache.h"

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
        m[i]->w = s->bytes + c->off_w[i];
        m[i]->s = s->bytes + c->off_s[i];
    }
}

int dsv4_cache_init(DSV4Cache *c, const DSV4St *st, const DSV4Cfg *cfg,
                    int64_t budget_bytes)
{
    memset(c, 0, sizeof *c);
    c->st = st; c->cfg = cfg;
    expert_geometry(c, cfg);

    if (budget_bytes < c->expert_bytes) {
        fprintf(stderr, "dsv4_cache: budget %lld bytes cannot hold one expert "
                        "(%lld bytes). Every access would miss and the same\n"
                        "  bytes would be read six times per layer.\n",
                (long long)budget_bytes, (long long)c->expert_bytes);
        return -1;
    }

    c->nslot = (int)(budget_bytes / c->expert_bytes);
    c->slot = (DSV4Slot *)calloc((size_t)c->nslot, sizeof(DSV4Slot));
    if (!c->slot) return -1;

    for (int i = 0; i < c->nslot; i++) {
        c->slot[i].layer = -1;
        c->slot[i].bytes = (unsigned char *)malloc((size_t)c->expert_bytes);
        if (!c->slot[i].bytes) { dsv4_cache_free(c); return -1; }
        bind_slot(c, &c->slot[i]);
    }
    return 0;
}

void dsv4_cache_free(DSV4Cache *c)
{
    if (!c || !c->slot) return;
    for (int i = 0; i < c->nslot; i++) free(c->slot[i].bytes);
    free(c->slot);
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
static int load_expert(DSV4Cache *c, DSV4Slot *s, const DSV4Tensor *t[6])
{
    /* (file offset, slot offset, length, shard), in slot order: s,s,s,w,w,w */
    struct { int64_t foff, soff, len; int shard; } r[6];
    for (int i = 0; i < 3; i++) {
        r[i].foff  = t[i * 2 + 1]->off;  r[i].soff = c->off_s[i];
        r[i].len   = t[i * 2 + 1]->nbytes; r[i].shard = t[i * 2 + 1]->shard;
        r[3 + i].foff  = t[i * 2]->off;   r[3 + i].soff = c->off_w[i];
        r[3 + i].len   = t[i * 2]->nbytes; r[3 + i].shard = t[i * 2]->shard;
    }
    /* six elements: insertion sort is the right tool */
    for (int i = 1; i < 6; i++)
        for (int j = i; j > 0 && r[j].foff < r[j - 1].foff; j--) {
            const typeof(r[0]) tmp = r[j]; r[j] = r[j - 1]; r[j - 1] = tmp;
        }

    int i = 0;
    while (i < 6) {
        int j = i;
        /* extend while both the FILE and the SLOT stay contiguous */
        while (j + 1 < 6
               && r[j + 1].shard == r[i].shard
               && r[j + 1].foff  == r[j].foff + r[j].len
               && r[j + 1].soff  == r[j].soff + r[j].len)
            j++;

        int64_t len = 0;
        for (int k = i; k <= j; k++) len += r[k].len;

        int64_t got = 0;
        while (got < len) {
            const ssize_t n = pread(c->st->fd[r[i].shard], s->bytes + r[i].soff + got,
                                    (size_t)(len - got), (off_t)(r[i].foff + got));
            if (n <= 0) return -1;
            got += n;
        }
        c->bytes_read += (uint64_t)len;
        i = j + 1;
    }
    return 0;
}

const DSV4ExpertW *dsv4_cache_get(DSV4Cache *c, int layer, int expert)
{
    c->clock++;

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

    if (load_expert(c, &c->slot[victim], t) != 0) {
        c->slot[victim].layer = -1;      /* do not leave a half-read slot live */
        return NULL;
    }
    c->slot[victim].layer  = layer;
    c->slot[victim].expert = expert;
    c->slot[victim].stamp  = c->clock;
    c->misses++;
    return &c->slot[victim].w;
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
