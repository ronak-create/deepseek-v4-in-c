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
    /* ONE ACCUMULATOR PER k, not one shared.
     *
     * With overlapped fetching, expert k's matmul runs as soon as expert k is
     * available -- which is not k order, because the residents go first and the
     * streamed ones land when the drive says so. The weighted sum into `out`
     * must still happen in k order or the result stops being bit-identical, so
     * each k needs somewhere of its own to leave its output until then.
     *
     * expert_acc[k] must also never alias the MoE input: x1 IS the MoE input at
     * every call site, and writing an expert's output there corrupted the input
     * for every later expert AND the shared expert. Per-kernel gates could not
     * see it; the whole-block oracle caught it. */
    float *expert_acc;   /* topk buffers of `hidden`, contiguous */
    int    expert_acc_stride;
    float *comp_kv_in, *comp_sc_in;   /* compressor projections, one token */
    float *idx_q, *idx_w, *idx_scores; /* CSA indexer working set          */
    int    idx_cap;                    /* how many scores idx_scores holds  */
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
    /* Optional. Fetch a whole layer's top-k at once so the misses can be read
     * concurrently -- serialised, they run at queue depth one and miss the
     * drive's plateau by better than 2x. May be NULL, in which case the layer
     * falls back to calling get() in a loop, which is correct and slower. The
     * accumulation order does not change either way, so neither does the
     * output. */
    int (*get_many)(void *ctx, int layer, const int *experts, int n,
                    const DSV4ExpertW **out);

    /* Optional overlapped pair, and the reason moe() is shaped the way it is.
     *
     * begin() does the whole decide pass and returns immediately, filling
     * out[k] for every expert already resident and leaving NULL for every one
     * that has to be read. end() waits for those reads and fills in the rest.
     * Between the two, the caller owns the CPU -- so the resident experts'
     * matmuls run while the streamed ones are still on the wire.
     *
     * Both or neither. When either is NULL the layer uses get_many, which is
     * correct and does not overlap. Output is identical either way, which is
     * the property the whole design is arranged around: an expert's matmul may
     * run in any order, but the weighted sum runs in k order regardless. */
    int (*begin)(void *ctx, int layer, const int *experts, int n,
                 const DSV4ExpertW **out);
    int (*end)(void *ctx, const DSV4ExpertW **out);

    void *ctx;
} DSV4ExpertSrc;

/* max_pos must be the SAME bound passed to dsv4_state_init. The indexer scores
 * one candidate per compressed row, and the compressed row count is a function
 * of max_pos -- so a scratch sized without it is sized by guesswork. It was:
 * idx_scores held a literal 4096 floats, which happened to suffice only because
 * the CLI passes sliding_window + 4096 and the smallest indexed ratio is 4.
 * Raise the context and that becomes a heap overflow with no warning. */
int  dsv4_scratch_init(DSV4Scratch *s, const DSV4Cfg *c, int max_pos);
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
/* WHERE THE TIME GOES, split so the CPU/GPU question can be answered with data.
 *
 * The distinction that matters is expert_io vs expert_mm. A streamed MoE reads
 * routed experts off NVMe and then multiplies by them; those two costs live in
 * the same loop but have opposite remedies. Only expert_mm can move to a GPU --
 * expert_io is a disk read, and putting it behind PCIe would make it worse.
 *
 * Seconds, accumulated across the whole run. Always on: a handful of
 * clock_gettime calls per layer is nothing beside a 12.75 MB read. */
typedef struct {
    double hc;          /* mHC pre/post, Sinkhorn, the two RMSNorms   */
    double attn;        /* CSA + HCA + RoPE + the output projection   */
    double gate;        /* router matmul and top-k selection          */
    double expert_io;   /* waiting for the expert cache               */
    double expert_mm;   /* routed-expert FP4 matmuls                  */
    double shared;      /* the shared expert                          */
} DSV4Prof;
extern DSV4Prof dsv4_prof;
void dsv4_prof_report(double wall, int passes);

void dsv4_layer_forward(float *h, const DSV4LayerW *w, const DSV4Cfg *c,
                        DSV4Scratch *s, const DSV4ExpertSrc *src,
                        DSV4LayerState *st, const float *cs, const float *sn,
                        int pos, int token_id);

#endif /* DSV4_LAYER_H */
