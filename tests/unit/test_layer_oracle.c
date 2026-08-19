/* SPDX-License-Identifier: Apache-2.0 */
/* test_layer_oracle.c - a whole decoder block against the PyTorch reference.
 *
 * WHAT THIS CATCHES THAT test_oracle.c CANNOT
 *   Per-kernel agreement is necessary and not sufficient. Every kernel can match
 *   to 5e-7 while the block computes the wrong thing, because the remaining
 *   mistakes live BETWEEN kernels:
 *     - the FFN residual captured at the top of the block instead of after the
 *       attention half
 *     - attn_norm applied to the pre-hc_pre state rather than the reduced one
 *     - the q latent handed to wq_b un-normalised
 *     - heads mixed across output groups by flattening before wo_a
 *     - the KV ring written at the wrong slot, which only shows after `window`
 *       steps
 *   None of those are visible from inside a kernel. This runs SIX positions so
 *   the ring wraps.
 *
 * WEIGHTS ARE bf16 BIT PATTERNS, not floats. The fixture carries raw uint16 and
 * the C points at them directly, so both implementations widen identical bits.
 * Emitting f32 and reading bf16 would compare two different models and call the
 * gap "tolerance".
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "dsv4_layer.h"
#include "json.h"

void dsv4_rope_table(float *, float *, int, int, int, double, double, double, double);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

static char *slurp(const char *p)
{
    FILE *f = fopen(p, "rb"); if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = malloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    s[n] = 0; fclose(f); return s;
}

static jval *G;

static int nf(const char *k, float *out, int max)
{
    jval *a = json_get(G, k);
    if (!a || a->t != J_ARR) return 0;
    int n = a->len < max ? a->len : max;
    for (int i = 0; i < n; i++) out[i] = (float)a->kids[i]->num;
    return n;
}

/* bf16 bit patterns straight from the fixture: no widening, no conversion. */
static uint16_t *nb(const char *k, int *count)
{
    jval *a = json_get(G, k);
    if (!a || a->t != J_ARR) { *count = 0; return NULL; }
    uint16_t *w = malloc((size_t)a->len * sizeof(uint16_t));
    for (int i = 0; i < a->len; i++) w[i] = (uint16_t)a->kids[i]->num;
    *count = a->len;
    return w;
}

static double cnum(jval *cfg, const char *k)
{
    jval *v = json_get(cfg, k);
    return (v && v->t == J_NUM) ? v->num : 0.0;
}

static void mk(DSV4QMat *m, const char *key, int rows, int cols)
{
    int n = 0;
    uint16_t *w = nb(key, &n);
    memset(m, 0, sizeof *m);
    m->wdt = DSV4_WBF16; m->rows = rows; m->cols = cols; m->w = w;
    if (n != rows * cols)
        printf("  FAIL  %s has %d bf16 values, expected %d\n", key, n, rows * cols);
}

/* The fixture's experts, served through the same interface the streaming cache
 * uses, so the block does not know it is being tested. */
static DSV4ExpertW g_ex[64];
static const DSV4ExpertW *get_expert(void *ctx, int layer, int e)
{
    (void)ctx; (void)layer;
    return &g_ex[e];
}

static int run_case(const char *file, const char *label)
{
    printf("\n== %s ==\n", label);

    char *txt = slurp(file);
    if (!txt) { printf("  FAIL  %s missing; run make ref-fixtures\n", file);
                fails++; return 1; }
    char *arena = NULL;
    G = json_parse(txt, &arena);
    if (!G) { printf("  FAIL  fixture is not valid JSON\n"); return 1; }
    jval *jc = json_get(G, "cfg");

    DSV4Cfg c; memset(&c, 0, sizeof c);
    c.hidden      = (int)cnum(jc, "hidden");
    c.hc_mult     = (int)cnum(jc, "hc_mult");
    c.n_heads     = (int)cnum(jc, "n_heads");
    c.n_kv_heads  = 1;
    c.head_dim    = (int)cnum(jc, "head_dim");
    c.qk_rope     = (int)cnum(jc, "qk_rope");
    c.q_lora      = (int)cnum(jc, "q_lora");
    c.o_lora      = (int)cnum(jc, "o_lora");
    c.o_groups    = (int)cnum(jc, "o_groups");
    c.sliding_window = (int)cnum(jc, "window");
    c.n_experts   = (int)cnum(jc, "n_experts");
    c.topk        = (int)cnum(jc, "topk");
    c.moe_inter   = (int)cnum(jc, "moe_inter");
    c.n_shared    = (int)cnum(jc, "n_shared");
    c.routed_scale = (float)cnum(jc, "route_scale");
    c.swiglu_limit = (float)cnum(jc, "swiglu_limit");
    c.rms_eps     = (float)cnum(jc, "rms_eps");
    c.hc_eps      = (float)cnum(jc, "hc_eps");
    c.hc_sinkhorn_iters = (int)cnum(jc, "sinkhorn_iters");
    c.n_layers    = 1;
    c.num_hash_layers = 0;                 /* scored routing */
    c.index_n_heads  = (int)cnum(jc, "n_heads");
    c.index_head_dim = (int)cnum(jc, "head_dim");
    c.index_topk     = 1 << 20;            /* take every compressed row */

    /* compress_ratio comes from the fixture. The layer map is one entry long
     * because the fixture is one layer. */
    static int cr[1];
    cr[0] = (int)cnum(jc, "ratio");
    c.n_compress = 1;
    c.compress_ratios = cr;

    const int hc = c.hc_mult, d = c.hidden, H = c.n_heads, hd = c.head_dim;
    const int mix = (2 + hc) * hc;

    DSV4LayerW w; memset(&w, 0, sizeof w);
    w.layer = 0; w.hash_routed = 0;
    w.compress_ratio = cr[0];
    w.has_comp = (cr[0] != 0);
    w.has_idx  = 0;   /* the indexer is exercised separately; here every
                       * compressed row is taken, which is what index_topk
                       * above forces and what the reference does */
    w.attn.has_comp = w.has_comp;
    w.attn.has_idx  = 0;

    static float attn_norm[512], ffn_norm[512], q_norm[512], kv_norm[512], sink[64];
    static float hcaf[8192], hcff[8192], hcab[64], hcfb[64], hcas[8], hcfs[8];
    nf("attn_norm", attn_norm, 512); nf("ffn_norm", ffn_norm, 512);
    nf("q_norm", q_norm, 512);       nf("kv_norm", kv_norm, 512);
    nf("sink", sink, 64);
    nf("hc_attn_fn", hcaf, 8192);    nf("hc_ffn_fn", hcff, 8192);
    nf("hc_attn_base", hcab, 64);    nf("hc_ffn_base", hcfb, 64);
    nf("hc_attn_scale", hcas, 8);    nf("hc_ffn_scale", hcfs, 8);

    w.attn_norm = attn_norm; w.ffn_norm = ffn_norm;
    w.attn.q_norm = q_norm;  w.attn.kv_norm = kv_norm; w.attn.sink = sink;
    w.hc_attn.fn = hcaf; w.hc_attn.base = hcab; w.hc_attn.scale = hcas;
    w.hc_ffn.fn  = hcff; w.hc_ffn.base  = hcfb; w.hc_ffn.scale  = hcfs;
    (void)mix;

    mk(&w.attn.wq_a, "wq_a", c.q_lora, d);
    mk(&w.attn.wq_b, "wq_b", H * hd, c.q_lora);
    mk(&w.attn.wkv,  "wkv",  hd, d);
    mk(&w.attn.wo_a, "wo_a", c.o_groups * c.o_lora, H * hd / c.o_groups);
    mk(&w.attn.wo_b, "wo_b", d, c.o_groups * c.o_lora);
    mk(&w.moe.gate,  "gate", c.n_experts, d);

    static float c_ape[4096], c_norm[512];
    if (w.has_comp) {
        const int coff = (cr[0] == 4) ? 2 : 1;
        mk(&w.attn.comp.wkv,   "c_wkv",   coff * hd, d);
        mk(&w.attn.comp.wgate, "c_wgate", coff * hd, d);
        nf("c_ape", c_ape, 4096);
        nf("c_norm", c_norm, 512);
        w.attn.comp.ape  = c_ape;
        w.attn.comp.norm = c_norm;
        w.attn.comp.ratio = cr[0];
        w.attn.comp.coff  = coff;
    }

    static float gbias[256];
    nf("gate_bias", gbias, 256);
    w.moe.bias = gbias;
    w.moe.tid2eid = NULL;

    char key[32];
    for (int e = 0; e < c.n_experts; e++) {
        snprintf(key, sizeof key, "e%d_w1", e); mk(&g_ex[e].w1, key, c.moe_inter, d);
        snprintf(key, sizeof key, "e%d_w3", e); mk(&g_ex[e].w3, key, c.moe_inter, d);
        snprintf(key, sizeof key, "e%d_w2", e); mk(&g_ex[e].w2, key, d, c.moe_inter);
    }
    mk(&w.moe.shared.w1, "sh_w1", c.moe_inter, d);
    mk(&w.moe.shared.w3, "sh_w3", c.moe_inter, d);
    mk(&w.moe.shared.w2, "sh_w2", d, c.moe_inter);

    /* The fixture's reference uses one table for both kinds, so match it. In
     * the real model a ratio-0 layer uses theta 10000 with YaRN off and every
     * compressed layer theta 160000 with YaRN on -- that split is covered by
     * the rope gate, not here. */
    static float cs[128 * 64], sn[128 * 64];
    dsv4_rope_table(cs, sn, c.qk_rope, 128, 0, 10000.0, 16.0, 32.0, 1.0);

    DSV4Scratch s;
    if (dsv4_scratch_init(&s, &c, 128) != 0) { printf("  FAIL  scratch\n"); return 1; }
    DSV4ExpertSrc src = { get_expert, NULL, NULL };

    static float h[1024];
    DSV4LayerState lstate;
    if (dsv4_state_init(&lstate, &c, 0, 128) != 0) {
        printf("  FAIL  layer state allocation\n");
        return 1;
    }
    nf("h_in", h, 1024);
    const int steps = (int)cnum(G, "steps");
    const int token_id = (int)cnum(G, "token_id");
    jval *outs = json_get(G, "h_out");

    printf("\n-- %d positions, KV ring of %d (so it wraps) --\n",
           steps, c.sliding_window);
    for (int pos = 0; pos < steps; pos++) {
        dsv4_layer_forward(h, &w, &c, &s, &src, &lstate, cs, sn, pos, token_id);

        jval *row = outs->kids[pos];
        float worst = 0.0f; int at = -1;
        double sq = 0.0;
        for (int i = 0; i < hc * d; i++) {
            const double r = row->kids[i]->num;
            sq += r * r;
        }
        const float rms = (float)sqrt(sq / (hc * d));
        for (int i = 0; i < hc * d; i++) {
            const float r = (float)row->kids[i]->num;
            const float dd = fabsf(h[i] - r);
            const float sc = fmaxf(rms, fmaxf(fabsf(h[i]), fabsf(r)));
            const float rel = sc > 0 ? dd / sc : 0.0f;
            if (rel > worst) { worst = rel; at = i; }
        }
        /* bf16 weights and a different summation order; 1e-4 is what the
         * arithmetic allows over a whole block, and is far tighter than any
         * wiring error would produce. A misplaced residual moves this to O(1). */
        if (worst > 1e-4f)
            CHECK(0, "pos %d: worst rel %.3g at element %d (C %.6g vs torch %.6g)",
                  pos, (double)worst, at, (double)h[at],
                  (double)row->kids[at]->num);
        else
            printf("  ok    pos %d  worst rel err %.2g over %d values\n",
                   pos, (double)worst, hc * d);
    }

    /* The compressor must close the same number of windows as the reference.
     * A row emitted on the wrong step, or one missed at the boundary, changes
     * how much history attention can see without changing any single value. */
    if (w.has_comp) {
        const int want = (int)cnum(G, "n_compressed");
        CHECK(lstate.n_compressed == want,
              "produced %d compressed rows, reference produced %d",
              lstate.n_compressed, want);
        if (lstate.n_compressed == want)
            printf("  ok    %d compressed rows, matching the reference\n", want);
    }

    dsv4_state_free(&lstate);
    dsv4_scratch_free(&s);
    free(txt);
    return 0;
}

int main(void)
{
    printf("DeepSeek-V4 WHOLE-BLOCK oracle gate\n");

    /* All three layer kinds through one code path. The overlap case is the one
     * worth having: it splices the previous window's first half-channels with
     * the current window's second half, then slides. Nothing smaller catches a
     * mistake in that. */
    run_case("tests/fixtures/ref/layer_dense.json",
             "compress_ratio 0  (dense, no compressor)");
    run_case("tests/fixtures/ref/layer_c8.json",
             "compress_ratio 8  (HCA, non-overlap)");
    run_case("tests/fixtures/ref/layer_c4.json",
             "compress_ratio 4  (HCA, OVERLAP: two half-windows spliced)");

    printf("\n");
    if (fails) { printf("BLOCK ORACLE GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("BLOCK ORACLE GATE PASSED\n");
    return 0;
}
