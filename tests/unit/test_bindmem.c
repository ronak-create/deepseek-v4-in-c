/* SPDX-License-Identifier: Apache-2.0 */
/* test_bindmem.c - binding a layer from a trunk run must agree, exactly, with
 * binding the same layer from the shards.
 *
 * THIS IS THE GATE THAT KEEPS THE TWO PATHS HONEST. The engine binds from the
 * shards when a layer is pinned and from a streamed run when it is not, and
 * K3's whole reason for sharing one plan_layer() is that two independent name
 * lists would drift. Drift here does not crash: it produces a model whose
 * behaviour depends on how much memory the machine happened to have.
 */
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include "dsv4_bind.h"
#include "dsv4_cfg.h"

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

/* A "trunk run": the layer's tensors concatenated verbatim, each with its
 * offset. That is what pack_trunk.py will produce, except that it reads from N
 * source runs rather than one. */
typedef struct { char name[224]; int64_t off, nbytes; int dtype; } Ent;
typedef struct { Ent *e; int n; } Run;

static int run_find(void *ctx, const char *name, int64_t *off, int64_t *nb, int *dt)
{
    Run *r = (Run *)ctx;
    for (int i = 0; i < r->n; i++)
        if (!strcmp(r->e[i].name, name)) {
            *off = r->e[i].off; *nb = r->e[i].nbytes; *dt = r->e[i].dtype;
            return 0;
        }
    return -1;
}

int main(void)
{
    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    DSV4St st;

    printf("DeepSeek-V4 trunk-bind agreement gate\n");

    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS,
                            "tests/fixtures/cfg/dsv4_flash_config.json")) return 1;
    if (dsv4_st_open(&st, "tests/fixtures/st") != 0) {
        printf("  FAIL  fixture missing; run make st-fixtures\n"); return 1;
    }
    const DSV4Tensor *tid = dsv4_st_find(&st, "layers.2.ffn.gate.tid2eid");
    if (tid) c.vocab = (int)tid->shape[0];

    for (int L = 2; L <= 3; L++) {
        printf("\n-- layer %d (%s) --\n", L,
               L == 2 ? "ratio 4, indexed, hash-routed" : "ratio 128, scored");

        DSV4LayerBind a;
        if (dsv4_bind_layer(&st, &c, L, &a) != 0) {
            printf("  FAIL  shard bind failed\n"); fails++; continue;
        }

        Run run; run.n = 0;
        run.e = (Ent *)calloc(160, sizeof(Ent));
        int64_t total = 0;
        char pfx[32]; snprintf(pfx, sizeof pfx, "layers.%d.", L);
        for (int i = 0; i < st.nt; i++) {
            if (strncmp(st.t[i].name, pfx, strlen(pfx))) continue;
            if (strstr(st.t[i].name, ".ffn.experts.")) continue;   /* streamed */
            snprintf(run.e[run.n].name, sizeof run.e[run.n].name, "%s", st.t[i].name);
            run.e[run.n].off    = total;
            run.e[run.n].nbytes = st.t[i].nbytes;
            run.e[run.n].dtype  = (int)st.t[i].dtype;
            total = (total + st.t[i].nbytes + 7) & ~(int64_t)7;
            run.n++;
        }
        unsigned char *bytes = (unsigned char *)malloc((size_t)total);
        for (int i = 0; i < run.n; i++) {
            const DSV4Tensor *t = dsv4_st_find(&st, run.e[i].name);
            dsv4_st_read(&st, t, bytes + run.e[i].off);
        }

        const size_t wcap = dsv4_bind_widen_bytes(&c);
        unsigned char *widen = (unsigned char *)malloc(wcap);
        DSV4MemSrc src = { run_find, &run };
        DSV4LayerBind b;
        size_t used = 0;
        if (dsv4_bind_layer_mem(&c, L, &b, bytes, &src, widen, wcap, &used) != 0) {
            printf("  FAIL  memory bind failed\n"); fails++;
        } else {
            CHECK(used <= wcap, "widen used %zu of %zu", used, wcap);

            struct { const char *what; const float *x, *y; int n; } wide[] = {
                { "attn_norm",   a.w.attn_norm,    b.w.attn_norm,    c.hidden   },
                { "ffn_norm",    a.w.ffn_norm,     b.w.ffn_norm,     c.hidden   },
                { "sink",        a.w.attn.sink,    b.w.attn.sink,    c.n_heads  },
                { "q_norm",      a.w.attn.q_norm,  b.w.attn.q_norm,  c.q_lora   },
                { "kv_norm",     a.w.attn.kv_norm, b.w.attn.kv_norm, c.head_dim },
                { "hc_attn.fn",  a.w.hc_attn.fn,   b.w.hc_attn.fn,
                  dsv4_mix_hc(&c) * dsv4_hc_dim(&c) },
                { "hc_ffn.base", a.w.hc_ffn.base,  b.w.hc_ffn.base, dsv4_mix_hc(&c) },
            };
            for (unsigned i = 0; i < sizeof wide / sizeof wide[0]; i++) {
                if (!wide[i].x || !wide[i].y) {
                    CHECK(0, "%s unbound on one path", wide[i].what); continue;
                }
                CHECK(memcmp(wide[i].x, wide[i].y,
                             (size_t)wide[i].n * sizeof(float)) == 0,
                      "%s differs between the shard and trunk paths", wide[i].what);
            }

            struct { const char *what; const DSV4QMat *x, *y; } nar[] = {
                { "wq_a", &a.w.attn.wq_a, &b.w.attn.wq_a },
                { "wq_b", &a.w.attn.wq_b, &b.w.attn.wq_b },
                { "wkv",  &a.w.attn.wkv,  &b.w.attn.wkv  },
                { "wo_a", &a.w.attn.wo_a, &b.w.attn.wo_a },
                { "wo_b", &a.w.attn.wo_b, &b.w.attn.wo_b },
                { "gate", &a.w.moe.gate,  &b.w.moe.gate  },
                { "shared.w1", &a.w.moe.shared.w1, &b.w.moe.shared.w1 },
                { "shared.w2", &a.w.moe.shared.w2, &b.w.moe.shared.w2 },
            };
            for (unsigned i = 0; i < sizeof nar / sizeof nar[0]; i++) {
                const DSV4QMat *x = nar[i].x, *y = nar[i].y;
                CHECK(x->wdt == y->wdt && x->rows == y->rows && x->cols == y->cols
                      && x->blk_r == y->blk_r && x->blk_c == y->blk_c,
                      "%s geometry differs between paths", nar[i].what);
                size_t nb = (x->wdt == DSV4_WFP4)
                          ? (size_t)x->rows * (size_t)(x->cols / 2)
                          : (size_t)x->rows * (size_t)x->cols
                            * (size_t)(x->wdt == DSV4_WBF16 ? 2 : 1);
                CHECK(memcmp(x->w, y->w, nb) == 0,
                      "%s weight bytes differ between paths", nar[i].what);
                if (x->s && y->s) {
                    size_t ns = (size_t)((x->rows + x->blk_r - 1) / x->blk_r)
                              * (size_t)((x->cols + x->blk_c - 1) / x->blk_c);
                    CHECK(memcmp(x->s, y->s, ns) == 0,
                          "%s scale bytes differ between paths", nar[i].what);
                }
            }

            CHECK(a.w.has_idx == b.w.has_idx && a.w.has_comp == b.w.has_comp
                  && a.w.hash_routed == b.w.hash_routed
                  && a.w.compress_ratio == b.w.compress_ratio,
                  "structural flags differ between paths");

            /* The trunk path must BORROW, not copy: that is why streaming a
             * layer costs one read and no unpacking. */
            CHECK((const unsigned char *)b.w.attn.wq_a.w >= bytes &&
                  (const unsigned char *)b.w.attn.wq_a.w <  bytes + total,
                  "the trunk path copied a narrow matrix instead of borrowing it");

            printf("  ok    %d wide and %d narrow tensors agree; widen used "
                   "%zu of %zu bytes\n",
                   (int)(sizeof wide / sizeof wide[0]),
                   (int)(sizeof nar / sizeof nar[0]), used, wcap);
        }
        dsv4_bind_free(&a);
        free(bytes); free(widen); free(run.e);
    }

    dsv4_st_close(&st);
    printf("\n");
    if (fails) { printf("TRUNK-BIND GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("TRUNK-BIND GATE PASSED\n");
    return 0;
}
