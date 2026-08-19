/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_layer.h - one decoder layer, decode path. See dsv4_layer.c. */
#ifndef DSV4_LAYER_H
#define DSV4_LAYER_H

#include "dsv4.h"
#include "dsv4_bind.h"

/* Per-layer scratch, allocated once and reused. A decode step touches every
 * buffer here once per layer per token, so churning them would dominate. */
typedef struct {
    float *x1, *resid, *post, *comb, *mixes;
    float *qr, *q, *kv, *o, *ogrp;
    float *gate_scores, *gate_orig, *expert_gate, *expert_up, *expert_out;
    float *expert_acc;   /* one expert's hidden-width output; MUST NOT
                          * alias the MoE input -- see dsv4_layer.c */
    float *comp_kv_in, *comp_sc_in;   /* compressor projections, one token */
    float *idx_q, *idx_w, *idx_scores; /* CSA indexer working set          */
    float *attn_scratch;
    int   *idxs;
    int    topk_idx[DSV4_MAX_TOPK];
    float  topk_w[DSV4_MAX_TOPK];
    void  *arena;
} DSV4Scratch;

/* A source of routed experts.
 *
 * Routed experts are streamed, not resident: 13,369,344 bytes each, 256 per
 * layer, 137 GB for Flash. The layer asks for one at a time and never holds
 * more than topk. Returning NULL means the expert could not be provided, which
 * is a correctness failure the caller must surface -- silently skipping it
 * leaves a token routed through fewer experts than the model specifies. */
typedef struct {
    const DSV4ExpertW *(*get)(void *ctx, int layer, int expert);
    void *ctx;
} DSV4ExpertSrc;

int  dsv4_scratch_init(DSV4Scratch *s, const DSV4Cfg *c);
void dsv4_scratch_free(DSV4Scratch *s);

/* PER-LAYER state that persists across tokens.
 *
 * Distinct from DSV4Scratch, which is reused by every layer within one token.
 * These buffers belong to one layer for the whole sequence, so a streaming
 * engine that evicts a layer's weights must NOT evict this.
 *
 *   kv_cache    [sliding_window + max_compressed][head_dim]
 *               The window ring first, then the compressed rows. That ordering
 *               is why the indexer offsets its selections by sliding_window.
 *   comp_kv,
 *   comp_score  [coff*ratio][coff*head_dim], the compressor's open window.
 *               score_state MUST start at -INFINITY: unfilled slots have to
 *               contribute exactly zero to the pooling softmax, and zero-filled
 *               scores would instead give them uniform weight.
 */
typedef struct {
    float *kv_cache;
    float *comp_kv, *comp_score;
    int    n_compressed;      /* rows written so far */
    void  *arena;
} DSV4LayerState;

int  dsv4_state_init(DSV4LayerState *st, const DSV4Cfg *c, int layer,
                     int max_pos);
void dsv4_state_free(DSV4LayerState *st);

/* `h` is [hc_mult][hidden] in and out. kv_cache is the layer's ring:
 * sliding_window rows of head_dim, then the compressed rows if any. */
/* `cs`/`sn` are the layer's RoPE table. A compressed layer needs a SECOND
 * table: model.py builds freqs_cis with compress_rope_theta and YaRN enabled for
 * any compress_ratio != 0, and with rope_theta and YaRN disabled for ratio 0.
 * The compressor rotates with the same table as the layer it belongs to. */
void dsv4_layer_forward(float *h, const DSV4LayerW *w, const DSV4Cfg *c,
                        DSV4Scratch *s, const DSV4ExpertSrc *src,
                        DSV4LayerState *st, const float *cs, const float *sn,
                        int pos, int token_id);

#endif /* DSV4_LAYER_H */
