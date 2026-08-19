/* SPDX-License-Identifier: Apache-2.0 */
/* test_layer.c - the decoder block, wired end to end on a tiny synthetic model.
 *
 * WHAT THIS GATE CAN AND CANNOT ESTABLISH
 *   It cannot establish numerical agreement with DeepSeek: that needs the
 *   PyTorch oracle and is Phase 3's job. What it CAN establish is that the
 *   plumbing is right, and the plumbing is where this architecture hides its
 *   mistakes -- a residual taken at the wrong point, a head group mixed with its
 *   neighbour, a token id that never reaches the router.
 *
 *   So every gate here is a STRUCTURAL claim with a counterfactual: something
 *   that must change when a specific input changes, or must not.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>
#include "dsv4_layer.h"

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

/* ---- a tiny model, small enough to reason about ---- */
#define HID   8
#define HC    2
#define HEADS 2
#define HDIM  4
#define QLORA 4
#define OLORA 2
#define OGRP  2
#define NEXP  4
#define TOPK  2
#define INTER 6
#define WIN   4
#define RD    2

static float *alloc_f(size_t n) { return (float *)calloc(n, sizeof(float)); }

/* Deterministic small values; a constant fill would let an indexing bug hide. */
static void fill(float *p, size_t n, unsigned seed, float amp)
{
    unsigned st = seed;
    for (size_t i = 0; i < n; i++) {
        st = st * 1103515245u + 12345u;
        p[i] = (float)((int)((st >> 16) & 0xFF) - 128) * amp;
    }
}

static void mk_bf16(DSV4QMat *m, int rows, int cols, unsigned seed, float amp)
{
    memset(m, 0, sizeof *m);
    m->wdt = DSV4_WBF16; m->rows = rows; m->cols = cols;
    uint16_t *w = (uint16_t *)calloc((size_t)rows * cols, sizeof(uint16_t));
    float *tmp = alloc_f((size_t)rows * cols);
    fill(tmp, (size_t)rows * cols, seed, amp);
    for (int i = 0; i < rows * cols; i++) {
        union { float f; uint32_t u; } v; v.f = tmp[i];
        w[i] = (uint16_t)(v.u >> 16);
    }
    free(tmp);
    m->w = w;
}

static DSV4ExpertW g_experts[NEXP];

static const DSV4ExpertW *get_expert(void *ctx, int layer, int e)
{
    (void)ctx; (void)layer;
    return (e >= 0 && e < NEXP) ? &g_experts[e] : NULL;
}

static void mk_expert(DSV4ExpertW *e, unsigned seed)
{
    mk_bf16(&e->w1, INTER, HID,   seed + 1, 0.01f);
    mk_bf16(&e->w3, INTER, HID,   seed + 2, 0.01f);
    mk_bf16(&e->w2, HID,   INTER, seed + 3, 0.01f);
}

static DSV4Cfg   g_cfg;
static DSV4LayerW g_w;
static float *g_hcfn_a, *g_hcfn_f;

static void build(int hash_routed)
{
    memset(&g_cfg, 0, sizeof g_cfg);
    g_cfg.hidden = HID; g_cfg.n_layers = 4; g_cfg.vocab = 16;
    g_cfg.rms_eps = 1e-6f; g_cfg.hc_eps = 1e-6f;
    g_cfg.n_heads = HEADS; g_cfg.n_kv_heads = 1; g_cfg.head_dim = HDIM;
    g_cfg.q_lora = QLORA; g_cfg.o_lora = OLORA; g_cfg.o_groups = OGRP;
    g_cfg.qk_rope = RD; g_cfg.sliding_window = WIN;
    g_cfg.index_topk = 0;
    g_cfg.hc_mult = HC; g_cfg.hc_sinkhorn_iters = 20;
    g_cfg.n_experts = NEXP; g_cfg.topk = TOPK; g_cfg.n_shared = 1;
    g_cfg.moe_inter = INTER; g_cfg.routed_scale = 1.0f; g_cfg.norm_topk = 1;
    g_cfg.swiglu_limit = 10.0f;
    g_cfg.num_hash_layers = hash_routed ? 4 : 0;

    memset(&g_w, 0, sizeof g_w);
    g_w.layer = 0;
    g_w.hash_routed = hash_routed;

    float *an = alloc_f(HID), *fn = alloc_f(HID);
    for (int i = 0; i < HID; i++) { an[i] = 1.0f; fn[i] = 1.0f; }
    g_w.attn_norm = an; g_w.ffn_norm = fn;

    float *sink = alloc_f(HEADS), *qn = alloc_f(QLORA), *kn = alloc_f(HDIM);
    for (int i = 0; i < HEADS; i++) sink[i] = -5.0f;
    for (int i = 0; i < QLORA; i++) qn[i] = 1.0f;
    for (int i = 0; i < HDIM; i++)  kn[i] = 1.0f;
    g_w.attn.sink = sink; g_w.attn.q_norm = qn; g_w.attn.kv_norm = kn;

    mk_bf16(&g_w.attn.wq_a, QLORA, HID, 11, 0.02f);
    mk_bf16(&g_w.attn.wq_b, HEADS * HDIM, QLORA, 12, 0.02f);
    mk_bf16(&g_w.attn.wkv,  HDIM, HID, 13, 0.02f);
    mk_bf16(&g_w.attn.wo_a, OGRP * OLORA, HEADS * HDIM / OGRP, 14, 0.02f);
    mk_bf16(&g_w.attn.wo_b, HID, OGRP * OLORA, 15, 0.02f);

    const int mix = (2 + HC) * HC, hcd = HC * HID;
    g_hcfn_a = alloc_f((size_t)mix * hcd);
    g_hcfn_f = alloc_f((size_t)mix * hcd);
    fill(g_hcfn_a, (size_t)mix * hcd, 21, 0.01f);
    fill(g_hcfn_f, (size_t)mix * hcd, 22, 0.01f);
    float *ba = alloc_f(mix), *bf = alloc_f(mix);
    float *sa = alloc_f(3), *sf = alloc_f(3);
    for (int i = 0; i < 3; i++) { sa[i] = 1.0f; sf[i] = 1.0f; }
    g_w.hc_attn.fn = g_hcfn_a; g_w.hc_attn.base = ba; g_w.hc_attn.scale = sa;
    g_w.hc_ffn.fn  = g_hcfn_f; g_w.hc_ffn.base  = bf; g_w.hc_ffn.scale  = sf;

    mk_bf16(&g_w.moe.gate, NEXP, HID, 31, 0.05f);
    if (hash_routed) {
        int64_t *t = (int64_t *)calloc((size_t)g_cfg.vocab * TOPK, sizeof(int64_t));
        for (int v = 0; v < g_cfg.vocab; v++)
            for (int k = 0; k < TOPK; k++) t[v * TOPK + k] = (v + k) % NEXP;
        g_w.moe.tid2eid = t;
    } else {
        float *b = alloc_f(NEXP);
        fill(b, NEXP, 41, 0.01f);
        g_w.moe.bias = b;
    }
    mk_expert(&g_w.moe.shared, 50);
    for (int e = 0; e < NEXP; e++) mk_expert(&g_experts[e], 60u + (unsigned)e * 10u);
}

int main(void)
{
    printf("DeepSeek-V4 decoder layer gate\n");

    const int hcd = HC * HID;
    float *cs = alloc_f((size_t)64 * (RD / 2));
    float *sn = alloc_f((size_t)64 * (RD / 2));
    for (int t = 0; t < 64; t++)
        for (int i = 0; i < RD / 2; i++) {
            cs[t * (RD / 2) + i] = cosf((float)t * 0.1f);
            sn[t * (RD / 2) + i] = sinf((float)t * 0.1f);
        }

    DSV4ExpertSrc src = { get_expert, NULL, NULL };

    printf("\n-- GATE 1  a block runs and keeps the stream finite --\n");
    {
        build(0);
        DSV4Scratch s;
        CHECK(dsv4_scratch_init(&s, &g_cfg, 64) == 0, "scratch allocation failed");
        float *h = alloc_f(hcd);
        DSV4LayerState lstate;
        dsv4_state_init(&lstate, &g_cfg, 0, 64);
        float *before = alloc_f(hcd);
        fill(h, hcd, 77, 0.1f);
        memcpy(before, h, (size_t)hcd * sizeof(float));

        dsv4_layer_forward(h, &g_w, &g_cfg, &s, &src, &lstate, cs, sn, 0, 3);

        int finite = 0, changed = 0;
        for (int i = 0; i < hcd; i++) {
            if (isfinite(h[i])) finite++;
            if (h[i] != before[i]) changed = 1;
        }
        CHECK(finite == hcd, "%d of %d outputs finite", finite, hcd);
        CHECK(changed, "the block left the residual stream untouched");
        printf("  ok    %d values, all finite, stream advanced\n", hcd);
        free(h); free(before); dsv4_state_free(&lstate); dsv4_scratch_free(&s);
    }

    printf("\n-- GATE 2  the token id reaches the router on a HASH layer --\n");
    {
        /* Same hidden state, different token. On a hash-routed layer the expert
         * set is a function of the token alone, so the output MUST differ. If
         * token_id were dropped somewhere between the block and the gate, this
         * is the only gate that would notice. */
        build(1);
        DSV4Scratch s; dsv4_scratch_init(&s, &g_cfg, 64);
        float *h1 = alloc_f(hcd), *h2 = alloc_f(hcd);
        DSV4LayerState lstate;
        fill(h1, hcd, 88, 0.1f);
        memcpy(h2, h1, (size_t)hcd * sizeof(float));

        dsv4_state_init(&lstate, &g_cfg, 0, 64);
        dsv4_layer_forward(h1, &g_w, &g_cfg, &s, &src, &lstate, cs, sn, 0, 1);
        dsv4_state_free(&lstate);
        dsv4_state_init(&lstate, &g_cfg, 0, 64);
        dsv4_layer_forward(h2, &g_w, &g_cfg, &s, &src, &lstate, cs, sn, 0, 2);

        CHECK(memcmp(h1, h2, (size_t)hcd * sizeof(float)) != 0,
              "tokens 1 and 2 gave identical output; token_id is not reaching "
              "the hash router");
        printf("  ok    token 1 and token 2 route differently\n");
        free(h1); free(h2); dsv4_state_free(&lstate); dsv4_scratch_free(&s);
    }

    printf("\n-- GATE 3  on a SCORED layer the token id is irrelevant --\n");
    {
        /* The mirror of GATE 2. A scored layer routes on the hidden state only,
         * so passing a different token must change nothing. If it does, the
         * token has leaked into a path that should not see it. */
        build(0);
        DSV4Scratch s; dsv4_scratch_init(&s, &g_cfg, 64);
        float *h1 = alloc_f(hcd), *h2 = alloc_f(hcd);
        DSV4LayerState lstate;
        fill(h1, hcd, 88, 0.1f);
        memcpy(h2, h1, (size_t)hcd * sizeof(float));

        dsv4_state_init(&lstate, &g_cfg, 0, 64);
        dsv4_layer_forward(h1, &g_w, &g_cfg, &s, &src, &lstate, cs, sn, 0, 1);
        dsv4_state_free(&lstate);
        dsv4_state_init(&lstate, &g_cfg, 0, 64);
        dsv4_layer_forward(h2, &g_w, &g_cfg, &s, &src, &lstate, cs, sn, 0, 9);

        CHECK(memcmp(h1, h2, (size_t)hcd * sizeof(float)) == 0,
              "a scored layer's output depends on token_id; it must not");
        printf("  ok    tokens 1 and 9 give identical output\n");
        free(h1); free(h2); dsv4_state_free(&lstate); dsv4_scratch_free(&s);
    }

    printf("\n-- GATE 4  position advances the KV ring and changes the output --\n");
    {
        build(0);
        DSV4Scratch s; dsv4_scratch_init(&s, &g_cfg, 64);
        float *h = alloc_f(hcd);
        DSV4LayerState lstate;
        dsv4_state_init(&lstate, &g_cfg, 0, 64);
        float prev[HC * HID];
        fill(h, hcd, 55, 0.1f);

        int distinct = 0;
        for (int pos = 0; pos < WIN + 2; pos++) {
            memcpy(prev, h, (size_t)hcd * sizeof(float));
        dsv4_layer_forward(h, &g_w, &g_cfg, &s, &src, &lstate, cs, sn, pos, 3);
            for (int i = 0; i < hcd; i++)
                if (!isfinite(h[i])) { CHECK(0, "non-finite at pos %d", pos); break; }
            if (memcmp(prev, h, (size_t)hcd * sizeof(float)) != 0) distinct++;
        }
        CHECK(distinct == WIN + 2, "only %d of %d steps changed the stream",
              distinct, WIN + 2);
        /* The ring must be fully written after `win` steps. */
        int nonzero_rows = 0;
        for (int r = 0; r < WIN; r++) {
            int nz = 0;
            for (int c = 0; c < HDIM; c++) if (lstate.kv_cache[r * HDIM + c] != 0.0f) nz = 1;
            nonzero_rows += nz;
        }
        CHECK(nonzero_rows == WIN, "%d of %d ring rows written", nonzero_rows, WIN);
        printf("  ok    %d steps, all finite, all %d ring rows written\n",
               WIN + 2, WIN);
        free(h); dsv4_state_free(&lstate); dsv4_scratch_free(&s);
    }

    printf("\n-- GATE 5  the FFN residual is taken AFTER the attention half --\n");
    {
        /* Counterfactual: perturb ONLY the attention output path (by changing
         * wo_b) and check the final result moves. If the FFN half re-used the
         * block's input as its residual, the attention contribution would enter
         * only through hc_post's `x` term and a change here would still show --
         * so instead assert the stronger property: with the attention output
         * forced to zero, the result must STILL differ from the input, because
         * the residual carries it. */
        build(0);
        DSV4Scratch s; dsv4_scratch_init(&s, &g_cfg, 64);
        float *h = alloc_f(hcd);
        DSV4LayerState lstate;
        dsv4_state_init(&lstate, &g_cfg, 0, 64);
        float *in = alloc_f(hcd);
        fill(in, hcd, 99, 0.1f);

        memcpy(h, in, (size_t)hcd * sizeof(float));
        dsv4_layer_forward(h, &g_w, &g_cfg, &s, &src, &lstate, cs, sn, 0, 3);
        float ref[HC * HID]; memcpy(ref, h, sizeof ref);

        /* Zero wo_b so attention contributes nothing to x, leaving only the
         * residual path through both hc_post calls. */
        DSV4QMat saved = g_w.attn.wo_b;
        uint16_t *zeros = (uint16_t *)calloc((size_t)HID * OGRP * OLORA, 2);
        g_w.attn.wo_b.w = zeros;
        memcpy(h, in, (size_t)hcd * sizeof(float));
        dsv4_state_free(&lstate); dsv4_state_init(&lstate, &g_cfg, 0, 64);
        dsv4_layer_forward(h, &g_w, &g_cfg, &s, &src, &lstate, cs, sn, 0, 3);

        CHECK(memcmp(ref, h, sizeof ref) != 0,
              "zeroing the attention output changed nothing; attention is not "
              "reaching the stream");
        int still_moved = 0;
        for (int i = 0; i < hcd; i++) if (h[i] != in[i]) still_moved = 1;
        CHECK(still_moved, "with attention zeroed the block became the identity; "
              "the FFN half is not seeing its own residual");
        g_w.attn.wo_b = saved; free(zeros);
        printf("  ok    attention reaches the stream, and the FFN half still "
               "advances without it\n");
        free(h); free(in); dsv4_state_free(&lstate); dsv4_scratch_free(&s);
    }

    printf("\n");
    if (fails) { printf("LAYER GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("LAYER GATE PASSED\n");
    return 0;
}
