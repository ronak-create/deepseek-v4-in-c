/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_layer.c - one decoder layer, decode path.
 *
 * Source: model.py Block.forward, Attention.forward, MoE.forward.
 *
 * THE RESIDUAL STREAM IS hc_mult VECTORS WIDE, NOT ONE. Every layer boundary
 * carries [hc_mult][hidden], and a block is
 *
 *     residual = x
 *     x, post, comb = hc_pre(x, hc_attn)     hc_mult -> 1
 *     x = attn(attn_norm(x))
 *     x = hc_post(x, residual, post, comb)   1 -> hc_mult
 *     residual = x
 *     x, post, comb = hc_pre(x, hc_ffn)
 *     x = ffn(ffn_norm(x))
 *     x = hc_post(x, residual, post, comb)
 *
 * Note the SECOND residual is taken AFTER the attention half, not at the top of
 * the block. Reusing the block's input for both halves is the obvious reading
 * and quietly removes the attention contribution from the FFN's residual path.
 *
 * THREE THINGS INSIDE ATTENTION THAT ARE NOT OBVIOUS
 *
 * 1. q IS NORMALISED TWICE, and the second one has NO WEIGHT:
 *        qr = q_norm(wq_a(x))                <- learned RMSNorm on q_lora
 *        q  = wq_b(qr) -> [n_heads, head_dim]
 *        q *= rsqrt(q.square().mean(-1) + eps)   <- per head, UNWEIGHTED
 *    The second is easy to read as a repeat of q_norm and is not: it has no
 *    parameter and it runs per head over head_dim, not over q_lora.
 *
 * 2. qr IS REUSED BY THE INDEXER. The indexer takes the pre-wq_b latent, not x
 *    and not q, which is why it shares q_lora_rank with attention.
 *
 * 3. THE OUTPUT PROJECTION IS GROUPED AND LOW-RANK. o is viewed as o_groups
 *    slices, each multiplied by its own slice of wo_a, and only then does wo_b
 *    map back to hidden. Flattening o and applying wo_a as one matrix has the
 *    right shapes and mixes heads that must stay in separate groups.
 */
#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4.h"
#include "dsv4_bind.h"
#include "dsv4_layer.h"
#include "dsv4_quant.h"

/* Kernels this file wires together. */
void dsv4_rmsnorm(float *, const float *, const float *, int, float);
void dsv4_swiglu(float *, const float *, const float *, int, float);
void dsv4_route(int *, float *, float *, float *, const float *,
                const int64_t *, int, int, int, float);
void dsv4_mmq(float *, const float *, const DSV4QMat *);
void dsv4_rope_apply(float *, int, int, const float *, const float *, int, int, int);
void dsv4_sparse_attn(float *, const float *, const float *, const float *,
                      const int *, int, int, int, float, float *);
int  dsv4_window_idxs(int *, int, int);
int  dsv4_compress_step(float *, const float *, const float *, const float *,
                        float *, float *, int, int, int);
int  dsv4_compress_rope_pos(int, int);
void dsv4_indexer_score(float *, const float *, const float *, const float *,
                        int, int, int);
int  dsv4_topk(int *, const float *, int, int);
void dsv4_indexer_offset(int *, int, int);
int  dsv4_indexer_navail(int, int);
void dsv4_hc_pre(float *, float *, float *, float *, const float *,
                 const DSV4HcW *, int, int, int, float, float);
void dsv4_hc_post(float *, const float *, const float *, const float *,
                  const float *, int, int);





/* ------------------------------------------------------------ layer state --- */

int dsv4_state_init(DSV4LayerState *st, const DSV4Cfg *c, int layer, int max_pos)
{
    memset(st, 0, sizeof *st);
    /* dsv4_compress_ratio returns -1 for a layer outside the map, and -1 is
     * TRUTHY. Writing `ratio ? ... : 0` therefore took the compressed branch and
     * computed max_pos / -1 + 1, a negative count that became an enormous
     * size_t. Clamp to 0 here so "no compressor" has exactly one spelling. */
    int ratio = dsv4_compress_ratio(c, layer);
    if (ratio < 0) ratio = 0;
    const int coff  = (ratio == 4) ? 2 : 1;
    const int hd    = c->head_dim;

    const size_t ncomp = ratio ? (size_t)(max_pos / ratio + 1) : 0;
    const size_t kvn   = ((size_t)c->sliding_window + ncomp) * (size_t)hd;
    const size_t cn    = ratio ? (size_t)coff * ratio * coff * hd : 0;

    float *a = (float *)calloc(kvn + 2 * cn, sizeof(float));
    if (!a) return -1;
    st->arena = a;
    st->kv_cache = a;
    if (ratio) {
        st->comp_kv    = a + kvn;
        st->comp_score = a + kvn + cn;
        /* -INFINITY, not zero: an unfilled slot must contribute nothing to the
         * pooling softmax. Zeros would give it uniform weight instead. */
        for (size_t i = 0; i < cn; i++) st->comp_score[i] = -INFINITY;
    }
    return 0;
}

void dsv4_state_free(DSV4LayerState *st)
{
    if (!st) return;
    free(st->arena);
    memset(st, 0, sizeof *st);
}

/* ---------------------------------------------------------------- scratch --- */

int dsv4_scratch_init(DSV4Scratch *s, const DSV4Cfg *c)
{
    memset(s, 0, sizeof *s);
    const size_t hc    = (size_t)c->hc_mult;
    const size_t d     = (size_t)c->hidden;
    const size_t heads = (size_t)c->n_heads;
    const size_t hd    = (size_t)c->head_dim;
    const size_t inter = (size_t)c->moe_inter * (size_t)c->n_shared;
    /* The window plus every compressed row a ratio-4 layer can select. */
    const size_t nidx  = (size_t)c->sliding_window + (size_t)c->index_topk;

    const size_t nf =
          d                 /* x1     */
        + hc * d            /* resid  */
        + hc                /* post   */
        + hc * hc           /* comb   */
        + (2 + hc) * hc     /* mixes  */
        + d                 /* qr (also reused as the layer output buffer) */
        + heads * hd        /* q      */
        + hd                /* kv     */
        + heads * hd        /* o      */
        + (size_t)c->o_groups * (size_t)c->o_lora   /* ogrp */
        + (size_t)c->n_experts * 2                  /* gate scores + orig  */
        + inter * 3                                 /* expert gate/up/out  */
        + (size_t)c->n_heads * nidx                 /* attn scratch, PER HEAD */
        + d                                         /* expert_acc          */
        + 4u * (size_t)c->head_dim                  /* comp kv/score in    */
        + (size_t)c->index_n_heads * (size_t)c->index_head_dim  /* idx q  */
        + (size_t)c->index_n_heads                  /* idx head weights    */
        + (size_t)(c->max_pos ? 4096 : 4096);       /* idx scores          */

    /* qr must hold max(q_lora, hidden): it carries the q latent and is reused
     * for the block's hidden-width output. Sizing it to q_lora alone overflows
     * on any model where hidden > q_lora, which is both released ones. */
    const size_t qr_n = (d > (size_t)c->q_lora) ? d : (size_t)c->q_lora;

    float *a = (float *)calloc(nf + qr_n, sizeof(float));
    if (!a) return -1;
    s->arena = a;

    float *p = a;
    s->x1 = p;           p += d;
    s->resid = p;        p += hc * d;
    s->post = p;         p += hc;
    s->comb = p;         p += hc * hc;
    s->mixes = p;        p += (2 + hc) * hc;
    s->qr = p;           p += qr_n;
    s->q = p;            p += heads * hd;
    s->kv = p;           p += hd;
    s->o = p;            p += heads * hd;
    s->ogrp = p;         p += (size_t)c->o_groups * (size_t)c->o_lora;
    s->gate_scores = p;  p += c->n_experts;
    s->gate_orig = p;    p += c->n_experts;
    s->expert_gate = p;  p += inter;
    s->expert_up = p;    p += inter;
    s->expert_out = p;   p += inter;
    s->attn_scratch = p; p += nidx;
    s->expert_acc = p;   p += d;
    s->comp_kv_in = p;   p += 2u * (size_t)c->head_dim;
    s->comp_sc_in = p;   p += 2u * (size_t)c->head_dim;
    s->idx_q = p;        p += (size_t)c->index_n_heads * (size_t)c->index_head_dim;
    s->idx_w = p;        p += (size_t)c->index_n_heads;
    s->idx_scores = p;   p += 4096;

    s->idxs = (int *)calloc(nidx, sizeof(int));
    if (!s->idxs) { free(a); s->arena = NULL; return -1; }
    return 0;
}

void dsv4_scratch_free(DSV4Scratch *s)
{
    if (!s) return;
    free(s->arena);
    free(s->idxs);
    memset(s, 0, sizeof *s);
}

/* SwiGLU expert: w2( silu(clamp(w1 x)) * clamp(w3 x) ). */
static void expert_forward(float *out, const float *x, const DSV4ExpertW *e,
                           DSV4Scratch *s, int hidden, int inter, float limit)
{
    dsv4_mmq(s->expert_gate, x, &e->w1);
    dsv4_mmq(s->expert_up,   x, &e->w3);
    dsv4_swiglu(s->expert_out, s->expert_gate, s->expert_up, inter, limit);
    dsv4_mmq(out, s->expert_out, &e->w2);
}

/* Attention for one token at absolute position `pos`.
 *
 * kv_cache is the layer's ring: `window` rows of head_dim, followed by the
 * compressed rows when the layer has a compressor. idxs must already hold the
 * window slots and any compressed selections, in cache coordinates. */
static void attention(float *out, const float *x, const DSV4LayerW *w,
                      const DSV4Cfg *c, DSV4Scratch *s, DSV4LayerState *st,
                      const float *cs, const float *sn, int pos, int n_idx)
{
    const int hd = c->head_dim, rd = c->qk_rope, H = c->n_heads;
    float *kv_cache = st->kv_cache;

    /* q: learned norm on the latent, then the per-head unweighted norm. */
    dsv4_mmq(s->qr, x, &w->attn.wq_a);
    dsv4_rmsnorm(s->qr, s->qr, w->attn.q_norm, c->q_lora, c->rms_eps);
    dsv4_mmq(s->q, s->qr, &w->attn.wq_b);
    for (int h = 0; h < H; h++) {
        float *qh = s->q + (size_t)h * hd;
        double sq = 0.0;
        for (int i = 0; i < hd; i++) sq += (double)qh[i] * qh[i];
        const float inv = (float)(1.0 / sqrt(sq / (double)hd + (double)c->rms_eps));
        for (int i = 0; i < hd; i++) qh[i] *= inv;
        dsv4_rope_apply(qh, hd, rd, cs, sn, pos, rd / 2, 0);
    }

    /* kv: one head, written into the sliding ring. */
    dsv4_mmq(s->kv, x, &w->attn.wkv);
    dsv4_rmsnorm(s->kv, s->kv, w->attn.kv_norm, hd, c->rms_eps);
    dsv4_rope_apply(s->kv, hd, rd, cs, sn, pos, rd / 2, 0);
    memcpy(kv_cache + (size_t)(pos % c->sliding_window) * hd, s->kv,
           (size_t)hd * sizeof(float));

    /* ---- HCA: pool `ratio` tokens into one compressed row ----
     * The compressor sees the SAME normed x attention does (model.py calls it
     * from inside Attention.forward, after attn_norm), and writes into the tail
     * of kv_cache, past the sliding window. */
    if (w->has_comp) {
        const int ratio = w->compress_ratio;
        const int coff  = (ratio == 4) ? 2 : 1;
        const int width = coff * hd;
        dsv4_mmq(s->comp_kv_in, x, &w->attn.comp.wkv);
        dsv4_mmq(s->comp_sc_in, x, &w->attn.comp.wgate);
        if (dsv4_compress_step(s->kv, s->comp_kv_in, s->comp_sc_in,
                               w->attn.comp.ape, st->comp_kv, st->comp_score,
                               pos, ratio, hd)) {
            dsv4_rmsnorm(s->kv, s->kv, w->attn.comp.norm, hd, c->rms_eps);
            /* Stamped with the FIRST position of the window it summarises. */
            dsv4_rope_apply(s->kv, hd, rd, cs, sn,
                            dsv4_compress_rope_pos(pos, ratio), rd / 2, 0);
            memcpy(kv_cache + (size_t)(c->sliding_window + st->n_compressed) * hd,
                   s->kv, (size_t)hd * sizeof(float));
            st->n_compressed++;
        }
        (void)width;

        /* ---- which compressed rows may attention see? ----
         * ratio 4 selects index_topk of them with the CSA indexer; every other
         * ratio takes them all, in order. Both then offset into the cache by
         * sliding_window, because the window occupies the front. */
        const int avail = st->n_compressed;
        int take = avail;
        if (avail > 0) {
            if (w->has_idx) {
                dsv4_mmq(s->idx_q, s->qr, &w->attn.idx.wq_b);
                for (int h = 0; h < c->index_n_heads; h++)
                    dsv4_rope_apply(s->idx_q + (size_t)h * c->index_head_dim,
                                    c->index_head_dim, rd, cs, sn, pos, rd / 2, 0);
                dsv4_mmq(s->idx_w, x, &w->attn.idx.weights_proj);
                dsv4_indexer_score(s->idx_scores, s->idx_q,
                                   kv_cache + (size_t)c->sliding_window * hd,
                                   s->idx_w, c->index_n_heads,
                                   c->index_head_dim, avail);
                take = avail < c->index_topk ? avail : c->index_topk;
                dsv4_topk(s->idxs + n_idx, s->idx_scores, avail, take);
            } else {
                for (int i = 0; i < take; i++) s->idxs[n_idx + i] = i;
            }
            dsv4_indexer_offset(s->idxs + n_idx, take, c->sliding_window);
            n_idx += take;
        }
    }

    dsv4_sparse_attn(s->o, s->q, kv_cache, w->attn.sink, s->idxs,
                     H, hd, n_idx, 1.0f / sqrtf((float)hd), s->attn_scratch);

    /* De-rotate, then the GROUPED low-rank output projection. */
    for (int h = 0; h < H; h++)
        dsv4_rope_apply(s->o + (size_t)h * hd, hd, rd, cs, sn, pos, rd / 2, 1);

    const int gw = H * hd / c->o_groups;       /* channels per group  */
    const int gr = c->o_lora;                  /* rank per group      */
    /* Stored bytes per weight for this matrix. BF16 is two, FP8 and packed FP4
     * are one -- and for FP4 a row is only cols/2 bytes wide.
     *
     * This is not incidental. The first version advanced the slice pointer by
     * g*gr*gw BYTES regardless of dtype, which is correct only for a 1-byte
     * format. With BF16 weights every group after the first read from halfway
     * into the wrong rows. Per-kernel gates cannot see it: every matmul was
     * right, the caller handed one of them the wrong pointer. The whole-block
     * oracle caught it at 14% divergence on the very first token. */
    const size_t esz = (w->attn.wo_a.wdt == DSV4_WBF16) ? 2u : 1u;
    const size_t row_bytes = (w->attn.wo_a.wdt == DSV4_WFP4)
                           ? (size_t)gw / 2u : (size_t)gw * esz;
    for (int g = 0; g < c->o_groups; g++) {
        DSV4QMat slice = w->attn.wo_a;
        /* wo_a is [o_groups*o_lora][gw]; group g owns rows [g*gr, (g+1)*gr). */
        slice.rows = gr;
        slice.cols = gw;
        slice.w = (const unsigned char *)w->attn.wo_a.w + (size_t)g * gr * row_bytes;
        if (slice.s) {
            /* The scale grid is sliced the same way, in scale-grid rows. */
            const size_t sc_row = (size_t)((gw + slice.blk_c - 1) / slice.blk_c);
            slice.s = (const unsigned char *)w->attn.wo_a.s
                    + (size_t)g * (size_t)(gr / (slice.blk_r ? slice.blk_r : 1)) * sc_row;
        }
        dsv4_mmq(s->ogrp + (size_t)g * gr, s->o + (size_t)g * gw, &slice);
    }
    dsv4_mmq(out, s->ogrp, &w->attn.wo_b);
}

/* MoE for one token. `token_id` is only read on hash-routed layers. */
static void moe(float *out, const float *x, const DSV4LayerW *w,
                const DSV4Cfg *c, DSV4Scratch *s, const DSV4ExpertSrc *src,
                int layer, int token_id)
{
    dsv4_mmq(s->gate_scores, x, &w->moe.gate);
    dsv4_route(s->topk_idx, s->topk_w, s->gate_scores, s->gate_orig,
               w->moe.bias, w->moe.tid2eid, token_id,
               c->n_experts, c->topk, c->routed_scale);

    for (int i = 0; i < c->hidden; i++) out[i] = 0.0f;

    /* expert_acc, NOT x1. x1 IS the MoE input `x` at every call site, so
     * writing an expert's output there corrupts the input for every expert
     * after the first and for the shared expert too.
     *
     * That was a live bug. Per-kernel gates could not see it: every matmul and
     * every activation was correct, and the caller handed one of them a buffer
     * that aliased its own input. The whole-block oracle caught it. */
    for (int k = 0; k < c->topk; k++) {
        const DSV4ExpertW *e = src->get(src->ctx, layer, s->topk_idx[k]);
        if (!e) continue;                    /* caller-reported; see header */
        expert_forward(s->expert_acc, x, e, s, c->hidden, c->moe_inter,
                       c->swiglu_limit);
        const float wk = s->topk_w[k];
        for (int i = 0; i < c->hidden; i++) out[i] += wk * s->expert_acc[i];
    }

    /* The shared expert is added UNWEIGHTED, after the routed sum. */
    expert_forward(s->expert_acc, x, &w->moe.shared, s, c->hidden,
                   c->moe_inter * c->n_shared, c->swiglu_limit);
    for (int i = 0; i < c->hidden; i++) out[i] += s->expert_acc[i];
}

/* One decoder layer. `h` is [hc_mult][hidden] in and out. */
void dsv4_layer_forward(float *h, const DSV4LayerW *w, const DSV4Cfg *c,
                        DSV4Scratch *s, const DSV4ExpertSrc *src,
                        DSV4LayerState *st, const float *cs, const float *sn,
                        int pos, int token_id)
{
    const int hc = c->hc_mult, d = c->hidden;
    const size_t wide = (size_t)hc * d;

    /* ---- attention half ---- */
    memcpy(s->resid, h, wide * sizeof(float));
    dsv4_hc_pre(s->x1, s->post, s->comb, s->mixes, h, &w->hc_attn,
                hc, d, c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
    dsv4_rmsnorm(s->x1, s->x1, w->attn_norm, d, c->rms_eps);

    const int n_idx = dsv4_window_idxs(s->idxs, pos, c->sliding_window);
    attention(s->qr, s->x1, w, c, s, st, cs, sn, pos, n_idx);

    dsv4_hc_post(h, s->qr, s->resid, s->post, s->comb, hc, d);

    /* ---- FFN half. The residual is taken HERE, not at the top. ---- */
    memcpy(s->resid, h, wide * sizeof(float));
    dsv4_hc_pre(s->x1, s->post, s->comb, s->mixes, h, &w->hc_ffn,
                hc, d, c->hc_sinkhorn_iters, c->rms_eps, c->hc_eps);
    dsv4_rmsnorm(s->x1, s->x1, w->ffn_norm, d, c->rms_eps);

    moe(s->qr, s->x1, w, c, s, src, w->layer, token_id);

    dsv4_hc_post(h, s->qr, s->resid, s->post, s->comb, hc, d);
}
