/* SPDX-License-Identifier: Apache-2.0 */
/* test_trunk.c - the streamed trunk, against the shards it was packed from.
 *
 * THE CLAIM WORTH PROVING
 *   A layer bound from the streamed trunk must be byte-identical to the same
 *   layer bound from the shards. If it is not, the model's behaviour depends on
 *   how much memory the machine had, because pinned layers and streamed layers
 *   take different paths. That is the worst kind of bug: it reproduces on one
 *   machine and not another.
 *
 * RUNS ON THE REAL CHECKPOINT when one is present, and skips cleanly otherwise
 * so the suite stays green on a machine without 150 GB of weights. A skip is
 * printed loudly rather than silently passing.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include "dsv4_trunk.h"
#include "dsv4_cfg.h"

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

static int have(const char *p)
{
    struct stat st;
    return stat(p, &st) == 0;
}

int main(int argc, char **argv)
{
    const char *model = (argc > 1) ? argv[1] : getenv("DSV4_MODEL");
    const char *trunk = (argc > 2) ? argv[2] : getenv("DSV4_TRUNK");

    printf("DeepSeek-V4 trunk streamer gate\n");

    if (!model || !trunk || !have(model) || !have(trunk)) {
        printf("  SKIP  no packed trunk available.\n");
        printf("        Build one with:\n");
        printf("          python3 tools/pack_trunk.py <model_dir> <trunk_dir> --layers 6\n");
        printf("        then re-run with DSV4_MODEL and DSV4_TRUNK set.\n");
        printf("\nTRUNK GATE SKIPPED\n");
        return 0;
    }

    char cfgpath[1024];
    snprintf(cfgpath, sizeof cfgpath, "%s/config.json", model);

    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS, cfgpath)) {
        printf("  FAIL  could not load %s\n", cfgpath); return 1;
    }

    DSV4St st;
    if (dsv4_st_open(&st, model) != 0) {
        printf("  FAIL  could not open %s\n", model); return 1;
    }

    printf("\n-- GATE 1  open the packed trunk --\n");
    DSV4Trunk tr;
    /* The budget must pin SOME layers and leave a ring of at least two slots,
     * so both paths and the round-robin are exercised in one run. It is derived
     * from the actual layer sizes rather than guessed: a fixed figure silently
     * degenerates into "pins everything" on a small trunk or "pins nothing" on
     * a large one, and the gate then tests half of what it claims to.
     *
     * An earlier version hard-coded 400 MB, pinned everything, and asserted
     * `npin < n_layers || n_layers <= npin` -- a tautology that could not fail.
     */
    int64_t widest = 0, first = 0;
    {
        DSV4Trunk probe;
        if (dsv4_trunk_open(&probe, trunk, &c, 1LL << 40) != 0) {
            printf("  FAIL  trunk open failed\n"); dsv4_st_close(&st); return 1;
        }
        for (int L = 0; L < probe.n_layers; L++)
            if (probe.lay[L].nbytes > widest) widest = probe.lay[L].nbytes;
        first = probe.lay[0].nbytes;
        dsv4_trunk_close(&probe);
    }
    const int64_t budget = first + 3 * widest + (1LL << 20);

    if (dsv4_trunk_open(&tr, trunk, &c, budget) != 0) {
        printf("  FAIL  trunk open failed\n"); dsv4_st_close(&st); return 1;
    }
    printf("  ok    %d layers, %d pinned, %d ring slot(s) of %.1f MB "
           "(budget %.0f MB)\n",
           tr.n_layers, tr.npin, tr.nslot, (double)tr.slot_bytes / 1048576.0,
           (double)budget / 1048576.0);
    CHECK(tr.npin > 0, "nothing pinned; the pinned path would go untested");
    CHECK(tr.npin < tr.n_layers,
          "the budget pinned all %d layers; the ring would go untested",
          tr.n_layers);
    /* The ring is DELIBERATELY minimal: dsv4_trunk_open reserves one slot and
     * spends everything else on pins, because a pinned layer is a guaranteed
     * hit and a ring slot is not. So nslot is normally 1, and demanding more
     * would be testing a policy the engine does not have. What the design DOES
     * promise is gated below. */

    printf("\n-- GATE 2  streamed bind == shard bind, byte for byte --\n");
    {
        int checked = 0;
        for (int L = 0; L < tr.n_layers && L < 6; L++) {
            if (tr.lay[L].nt == 0) continue;

            DSV4LayerBind a, b;
            if (dsv4_bind_layer(&st, &c, L, &a) != 0) {
                CHECK(0, "shard bind failed for layer %d", L); continue;
            }
            if (dsv4_trunk_bind(&tr, &c, L, &b) != 0) {
                CHECK(0, "trunk bind failed for layer %d", L);
                dsv4_bind_free(&a); continue;
            }

            /* Wide tensors: identical floats. Narrow: identical bytes. */
            const struct { const char *what; const float *x, *y; int n; } wide[] = {
                { "attn_norm", a.w.attn_norm, b.w.attn_norm, c.hidden },
                { "ffn_norm",  a.w.ffn_norm,  b.w.ffn_norm,  c.hidden },
                { "sink",      a.w.attn.sink, b.w.attn.sink, c.n_heads },
                { "hc_attn.fn",a.w.hc_attn.fn,b.w.hc_attn.fn,
                  dsv4_mix_hc(&c) * dsv4_hc_dim(&c) },
            };
            for (unsigned i = 0; i < sizeof wide / sizeof wide[0]; i++)
                CHECK(wide[i].x && wide[i].y &&
                      memcmp(wide[i].x, wide[i].y,
                             (size_t)wide[i].n * sizeof(float)) == 0,
                      "layer %d %s differs between shard and trunk",
                      L, wide[i].what);

            const struct { const char *what; const DSV4QMat *x, *y; } nar[] = {
                { "wq_a", &a.w.attn.wq_a, &b.w.attn.wq_a },
                { "wq_b", &a.w.attn.wq_b, &b.w.attn.wq_b },
                { "wo_a", &a.w.attn.wo_a, &b.w.attn.wo_a },
                { "wo_b", &a.w.attn.wo_b, &b.w.attn.wo_b },
                { "gate", &a.w.moe.gate,  &b.w.moe.gate  },
                { "shared.w1", &a.w.moe.shared.w1, &b.w.moe.shared.w1 },
            };
            for (unsigned i = 0; i < sizeof nar / sizeof nar[0]; i++) {
                const DSV4QMat *x = nar[i].x, *y = nar[i].y;
                CHECK(x->wdt == y->wdt && x->rows == y->rows && x->cols == y->cols,
                      "layer %d %s geometry differs", L, nar[i].what);
                const size_t nb = (x->wdt == DSV4_WFP4)
                    ? (size_t)x->rows * (size_t)(x->cols / 2)
                    : (size_t)x->rows * (size_t)x->cols
                      * (size_t)(x->wdt == DSV4_WBF16 ? 2 : 1);
                CHECK(memcmp(x->w, y->w, nb) == 0,
                      "layer %d %s weight bytes differ", L, nar[i].what);
                if (x->s && y->s) {
                    const size_t ns =
                        (size_t)((x->rows + x->blk_r - 1) / x->blk_r) *
                        (size_t)((x->cols + x->blk_c - 1) / x->blk_c);
                    CHECK(memcmp(x->s, y->s, ns) == 0,
                          "layer %d %s scale bytes differ", L, nar[i].what);
                }
            }
            CHECK(a.w.has_idx == b.w.has_idx && a.w.has_comp == b.w.has_comp
                  && a.w.compress_ratio == b.w.compress_ratio,
                  "layer %d structural flags differ", L);
            dsv4_bind_free(&a);
            checked++;
        }
        printf("  ok    %d layers agree between the shard and trunk paths\n", checked);
        CHECK(checked > 0, "no layers were compared");
    }

    printf("\n-- GATE 3  re-binding a pinned layer does not re-read --\n");
    {
        const uint64_t before = tr.misses;
        DSV4LayerBind b;
        dsv4_trunk_bind(&tr, &c, 0, &b);
        CHECK(tr.misses == before, "layer 0 is pinned but was read again");
        printf("  ok    pinned layer served without a read\n");
    }

    printf("\n-- GATE 4  the hit rate is exactly npin/n_layers --\n");
    {
        /* THE PROPERTY THE WHOLE POLICY EXISTS FOR.
         *
         * The trunk is walked 0..N-1 on every token. Under LRU that cyclic scan
         * hits ZERO percent however much memory is added, because the next layer
         * needed is always the one just evicted. Pinning a prefix instead makes
         * the hit rate exactly npin/N, deterministically, so every extra
         * gigabyte buys its fair share.
         *
         * Two full sweeps: after the first, every pinned layer is resident, so
         * the second must hit on exactly those and miss on exactly the rest.
         * A regression to LRU, or a pin quietly falling out, shows up here as a
         * hit count that is not npin. */
        for (int L = 0; L < tr.n_layers; L++) {
            if (tr.lay[L].nt == 0) continue;
            DSV4LayerBind b; dsv4_trunk_bind(&tr, &c, L, &b);
        }
        const uint64_t h0 = tr.hits, m0 = tr.misses;
        int n_present = 0;
        for (int L = 0; L < tr.n_layers; L++) {
            if (tr.lay[L].nt == 0) continue;
            DSV4LayerBind b; dsv4_trunk_bind(&tr, &c, L, &b);
            n_present++;
        }
        const uint64_t hits = tr.hits - h0, misses = tr.misses - m0;
        CHECK(hits == (uint64_t)tr.npin,
              "second sweep hit %llu times, expected exactly npin = %d",
              (unsigned long long)hits, tr.npin);
        CHECK(misses == (uint64_t)(n_present - tr.npin),
              "second sweep missed %llu times, expected %d",
              (unsigned long long)misses, n_present - tr.npin);
        printf("  ok    %llu hits + %llu misses over %d layers = %.0f%%, "
               "npin/N = %.0f%%\n",
               (unsigned long long)hits, (unsigned long long)misses, n_present,
               100.0 * (double)hits / (double)n_present,
               100.0 * (double)tr.npin / (double)n_present);
    }

    printf("\n");
    dsv4_trunk_report(&tr, "after the gate");
    dsv4_trunk_close(&tr);
    dsv4_st_close(&st);

    printf("\n");
    if (fails) { printf("TRUNK GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("TRUNK GATE PASSED\n");
    return 0;
}
