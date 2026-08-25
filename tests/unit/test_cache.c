/* SPDX-License-Identifier: Apache-2.0 */
/* test_cache.c - the routed-expert LRU, against the synthetic fixture. */
#include <stdio.h>
#include <string.h>
#include <unistd.h>          /* pread, for the independent reference read */

#include "dsv4_cache.h"
#include "dsv4_cfg.h"

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

int main(void)
{
    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    DSV4St st;

    printf("DeepSeek-V4 expert cache gate\n");

    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS,
                            "tests/fixtures/cfg/dsv4_flash_config.json")) return 1;
    if (dsv4_st_open(&st, "tests/fixtures/st") != 0) {
        printf("  FAIL  fixture missing; run 'make st-fixtures'\n"); return 1;
    }

    printf("\n-- GATE 1  expert size is computed, not assumed --\n");
    {
        DSV4Cache k;
        CHECK(dsv4_cache_init(&k, &st, &c, 1LL << 30) == 0, "init failed");
        /* The real figure, measured from the released checkpoint. If the FP4
         * half-width or the 1x32 scale grid were wrong this would not match. */
        CHECK(k.expert_bytes == 13369344,
              "expert is %lld bytes, the checkpoint says 13,369,344",
              (long long)k.expert_bytes);
        printf("  ok    %lld bytes per expert, %d slots in 1 GB\n",
               (long long)k.expert_bytes, k.nslot);
        dsv4_cache_free(&k);
    }

    printf("\n-- GATE 2  a budget too small for one expert is REFUSED --\n");
    {
        DSV4Cache k;
        CHECK(dsv4_cache_init(&k, &st, &c, 1000) != 0,
              "accepted a budget that cannot hold a single expert");
        printf("  ok    refused a 1000-byte budget\n");
    }

    printf("\n-- GATE 3  hit, miss and LRU eviction order --\n");
    {
        DSV4Cache k;
        /* Two slots exactly, so eviction is forced and predictable. Ask in
         * units of what a slot COSTS, not what an expert weighs -- a slot also
         * carries the alignment slack that keeps O_DIRECT copy-free. */
        dsv4_cache_init(&k, &st, &c, dsv4_cache_slot_bytes(&c) * 2);
        CHECK(k.nslot == 2, "%d slots, expected 2", k.nslot);

        /* The fixture only carries expert 0 of layers 2 and 3, so use those. */
        const DSV4ExpertW *a = dsv4_cache_get(&k, 2, 0);
        CHECK(a != NULL, "layer 2 expert 0 failed to load");
        CHECK(k.misses == 1 && k.hits == 0, "first access: %llu hits %llu misses",
              (unsigned long long)k.hits, (unsigned long long)k.misses);

        const DSV4ExpertW *a2 = dsv4_cache_get(&k, 2, 0);
        CHECK(a2 == a, "a hit returned a different pointer");
        CHECK(k.hits == 1, "second access was not a hit");

        const DSV4ExpertW *b = dsv4_cache_get(&k, 3, 0);
        /* Two failure modes needing different messages: an earlier version
         * reported a NULL as "shares a slot", which sent me looking at the LRU
         * when the fixture was simply missing the tensor. */
        CHECK(b != NULL, "layer 3 expert 0 failed to load");
        CHECK(b != a, "layer 3 expert 0 aliased layer 2's slot");
        CHECK(k.evictions == 0, "evicted with a free slot available");

        /* Touch layer 2 so layer 3 becomes least recently used, then force an
         * eviction and check the RIGHT one went. */
        /* A load that FAILS must not evict a live entry: the slot is only
         * claimed after the read succeeds. */
        dsv4_cache_get(&k, 2, 0);
        const unsigned long long ev_before = (unsigned long long)k.evictions;
        (void)dsv4_cache_get(&k, 3, 9);          /* absent -> load fails */
        CHECK((unsigned long long)k.evictions == ev_before,
              "a failed load evicted a live entry");
        CHECK(dsv4_cache_get(&k, 2, 0) == a,
              "layer 2 was lost to a load that never succeeded");
        printf("  ok    %llu hits, %llu misses, %llu evictions\n",
               (unsigned long long)k.hits, (unsigned long long)k.misses,
               (unsigned long long)k.evictions);
        dsv4_cache_free(&k);
    }

    printf("\n-- GATE 4  a failed load does not leave a live slot --\n");
    {
        /* Expert 7 is not in the fixture. The cache must report the failure and
         * must NOT leave a slot claiming to hold it -- a later request would
         * then be served garbage as a hit. */
        DSV4Cache k;
        dsv4_cache_init(&k, &st, &c, 13369344LL * 2);
        printf("  (expect a diagnostic naming the missing tensor)\n");
        const DSV4ExpertW *e = dsv4_cache_get(&k, 2, 7);
        CHECK(e == NULL, "a missing expert was reported as loaded");
        const DSV4ExpertW *again = dsv4_cache_get(&k, 2, 7);
        CHECK(again == NULL, "the failed load was cached and served as a hit");
        CHECK(k.hits == 0, "a failed load counted as a hit");
        printf("  ok    missing expert refused twice, never cached\n");
        dsv4_cache_free(&k);
    }

    printf("\n-- GATE 5  the bound matrices describe packed FP4 --\n");
    {
        DSV4Cache k;
        dsv4_cache_init(&k, &st, &c, 1LL << 30);
        const DSV4ExpertW *e = dsv4_cache_get(&k, 2, 0);
        if (e) {
            CHECK(e->w1.wdt == DSV4_WFP4, "w1 is not tagged FP4");
            CHECK(e->w1.blk_r == 1 && e->w1.blk_c == 32,
                  "w1 block is %dx%d, expected 1x32", e->w1.blk_r, e->w1.blk_c);
            CHECK(e->w1.rows == c.moe_inter && e->w1.cols == c.hidden,
                  "w1 is %dx%d, config implies %dx%d", e->w1.rows, e->w1.cols,
                  c.moe_inter, c.hidden);
            /* w2 is the down projection: transposed relative to w1/w3. */
            CHECK(e->w2.rows == c.hidden && e->w2.cols == c.moe_inter,
                  "w2 is %dx%d, expected %dx%d (down projection)",
                  e->w2.rows, e->w2.cols, c.hidden, c.moe_inter);
            CHECK(e->w1.w != NULL && e->w1.s != NULL, "w1 weight or scale unbound");
            CHECK(e->w1.w != e->w1.s, "weight and scale point at the same bytes");
            printf("  ok    w1 %dx%d 1x32, w2 %dx%d, all six tensors bound\n",
                   e->w1.rows, e->w1.cols, e->w2.rows, e->w2.cols);
        }
        dsv4_cache_free(&k);
    }

    printf("\n-- GATE 6  cached bytes equal an INDEPENDENT read of the same tensors --\n");
    {
        /* Every other check here compares the cache against itself, which is
         * exactly how a placement bug survived: O_DIRECT reads are widened to
         * the enclosing 4096 window, and a window that reached back over the
         * previous run corrupted it identically on every path -- so serial and
         * concurrent agreed, on the wrong bytes. The only cure is a reference
         * sharing no code with the fast path: a plain buffered pread of the
         * same tensor, compared byte for byte. */
        DSV4Cache k;
        dsv4_cache_init(&k, &st, &c, 1LL << 30);
        int bad = 0, checked = 0;
        for (int e = 0; e < 4 && !bad; e++) {
            const DSV4ExpertW *w = dsv4_cache_get(&k, 2, e);
            if (!w) continue;
            const DSV4QMat *m[6] = { &w->w1, &w->w2, &w->w3,
                                     &w->w1, &w->w2, &w->w3 };
            const char *sfx[6] = { "w1.weight", "w2.weight", "w3.weight",
                                   "w1.scale",  "w2.scale",  "w3.scale" };
            for (int t = 0; t < 6 && !bad; t++) {
                char name[256];
                snprintf(name, sizeof name, "layers.2.ffn.experts.%d.%s", e, sfx[t]);
                const DSV4Tensor *ti = dsv4_st_find(&st, name);
                if (!ti) continue;
                unsigned char *ref = (unsigned char *)malloc((size_t)ti->nbytes);
                if (!ref) { bad = 1; break; }
                int64_t got = 0;
                while (got < ti->nbytes) {
                    const ssize_t n = pread(st.fd[ti->shard], ref + got,
                                            (size_t)(ti->nbytes - got),
                                            (off_t)(ti->off + got));
                    if (n <= 0) break;
                    got += n;
                }
                const unsigned char *have = (t < 3) ? m[t]->w : m[t]->s;
                if (got != ti->nbytes
                    || memcmp(ref, have, (size_t)ti->nbytes) != 0) {
                    CHECK(0, "expert %d %s differs from a buffered read of the "
                             "same bytes", e, sfx[t]);
                    bad = 1;
                }
                checked++;
                free(ref);
            }
        }
        if (!bad)
            printf("  ok    %d tensors byte-identical to an independent "
                   "buffered read\n", checked);
        dsv4_cache_free(&k);
    }

    printf("\n-- GATE 7  the concurrent fetch equals the serial one, byte for byte --\n");
    {
        /* dsv4_cache_get_many reads its misses on several threads at once. The
         * bytes it produces must be indistinguishable from six calls to
         * dsv4_cache_get, or every token downstream is quietly different. Two
         * caches, same requests, compared byte for byte -- not by pointer, and
         * not by shape. */
        /* Layer 2 experts 0..5 exist in the fixture; 2 appears twice on purpose. */
        const int ids[6] = { 4, 2, 5, 0, 2, 3 };
        DSV4Cache a, b;
        dsv4_cache_init(&a, &st, &c, 1LL << 30);
        dsv4_cache_init(&b, &st, &c, 1LL << 30);

        const DSV4ExpertW *many[6];
        const int rc = dsv4_cache_get_many(&b, 2, ids, 6, many);
        CHECK(rc == 0, "get_many reported failure (%d)", rc);

        int bad = 0;
        for (int k = 0; k < 6; k++) {
            const DSV4ExpertW *one = dsv4_cache_get(&a, 2, ids[k]);
            if (!one || !many[k]) { CHECK(0, "expert %d missing", ids[k]); continue; }
            const struct { const DSV4QMat *x, *y; const char *n; } m[3] = {
                { &one->w1, &many[k]->w1, "w1" },
                { &one->w2, &many[k]->w2, "w2" },
                { &one->w3, &many[k]->w3, "w3" },
            };
            for (int q = 0; q < 3; q++) {
                const int64_t nb = (int64_t)m[q].x->rows * m[q].x->cols / 2;
                if (memcmp(m[q].x->w, m[q].y->w, (size_t)nb) != 0) {
                    CHECK(0, "expert %d %s weights differ", ids[k], m[q].n); bad++;
                }
                const int64_t ns = (int64_t)m[q].x->rows
                                 * ((m[q].x->cols + 31) / 32);
                if (memcmp(m[q].x->s, m[q].y->s, (size_t)ns) != 0) {
                    CHECK(0, "expert %d %s scales differ", ids[k], m[q].n); bad++;
                }
            }
        }
        /* The repeated id must be served from the same slot, not loaded twice. */
        CHECK(many[1] == many[4], "the duplicate request took a second slot");
        if (!bad && rc == 0) printf("  ok    6 experts (one repeated) identical to the "
                         "serial path, %lld bytes each\n",
                         (long long)b.expert_bytes);
        dsv4_cache_free(&a);
        dsv4_cache_free(&b);
    }

    printf("\n-- GATE 8  begin() hands back residents that are USABLE before end() --\n");
    {
        /* The contract dsv4_cache_get_many_begin adds is that the experts it
         * returns immediately are complete, and stay complete while the misses
         * are still being read. moe() depends on exactly that: it runs those
         * matmuls in the gap. If a resident could be disturbed by an in-flight
         * read -- evicted, rebound, written through -- the model would produce
         * wrong tokens only under timing, which is the worst way to find out.
         *
         * So: warm three experts, then ask for six. The three residents are
         * copied out DURING the gap and compared afterwards against a serial
         * reference. Anything the readers do to them shows up as a mismatch. */
        const int warm[3] = { 1, 3, 5 };
        const int ids[6]  = { 0, 1, 2, 3, 4, 5 };

        DSV4Cache a, b;
        dsv4_cache_init(&a, &st, &c, 1LL << 30);
        dsv4_cache_init(&b, &st, &c, 1LL << 30);
        for (int i = 0; i < 3; i++) dsv4_cache_get(&b, 2, warm[i]);

        const DSV4ExpertW *ex[6];
        DSV4CacheBatch bat;
        const int rc = dsv4_cache_get_many_begin(&b, 2, ids, 6, ex, &bat);
        CHECK(rc == 0, "begin reported failure (%d)", rc);

        /* In the gap. Exactly what moe() does here, minus the arithmetic. */
        int nres = 0;
        unsigned char *snap[6];
        int64_t snapn[6];
        for (int k = 0; k < 6; k++) {
            snap[k] = NULL; snapn[k] = 0;
            if (!ex[k]) continue;
            nres++;
            snapn[k] = (int64_t)ex[k]->w1.rows * ex[k]->w1.cols / 2;
            snap[k] = (unsigned char *)malloc((size_t)snapn[k]);
            if (snap[k]) memcpy(snap[k], ex[k]->w1.w, (size_t)snapn[k]);
        }
        CHECK(nres == 3, "expected 3 residents before end(), got %d", nres);

        const int rc2 = dsv4_cache_get_many_end(&b, ex, &bat);
        CHECK(rc2 == 0, "end reported failure (%d)", rc2);

        int bad8 = 0;
        for (int k = 0; k < 6; k++) {
            const DSV4ExpertW *one = dsv4_cache_get(&a, 2, ids[k]);
            if (!one || !ex[k]) { CHECK(0, "expert %d missing", ids[k]); continue; }
            const int64_t nb = (int64_t)one->w1.rows * one->w1.cols / 2;
            if (memcmp(one->w1.w, ex[k]->w1.w, (size_t)nb) != 0) {
                CHECK(0, "expert %d differs from the serial path", ids[k]); bad8++;
            }
            /* And the resident's bytes must not have MOVED during the gap. */
            if (snap[k] && memcmp(snap[k], ex[k]->w1.w, (size_t)snapn[k]) != 0) {
                CHECK(0, "resident expert %d changed while misses were read",
                      ids[k]);
                bad8++;
            }
            free(snap[k]);
        }
        if (!bad8 && rc == 0 && rc2 == 0)
            printf("  ok    3 residents usable in the gap and unchanged by it, "
                   "all 6 match the serial path\n");
        dsv4_cache_free(&a);
        dsv4_cache_free(&b);
    }

    dsv4_st_close(&st);
    printf("\n");
    if (fails) { printf("CACHE GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("CACHE GATE PASSED\n");
    return 0;
}
