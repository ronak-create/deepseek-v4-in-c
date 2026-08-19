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

/* `h` is [hc_mult][hidden] in and out. kv_cache is the layer's ring:
 * sliding_window rows of head_dim, then the compressed rows if any. */
void dsv4_layer_forward(float *h, const DSV4LayerW *w, const DSV4Cfg *c,
                        DSV4Scratch *s, const DSV4ExpertSrc *src,
                        float *kv_cache, const float *cs, const float *sn,
                        int pos, int token_id);

#endif /* DSV4_LAYER_H */
