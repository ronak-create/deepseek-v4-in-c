/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4.h - DeepSeek-V4 inference in portable C99.
 *
 * Derived from kimi-k3-in-c (Fareed Khan, Apache-2.0). The I/O, cache and CLI
 * subsystems are that project's; the model is not. DeepSeek-V4 shares none of
 * K3's kernels: no KDA, no MLA-NoPE, no SiTU-GLU, no MXFP4.
 *
 * WHAT THIS ARCHITECTURE IS  (config model_type "deepseek_v4")
 *   Attention   MLA-shaped q/o LoRA with ONE kv head, plus two things K3 has no
 *               analogue for:
 *                 CSA  a "lightning indexer" that scores all positions with its
 *                      own 64-head projection and keeps only index_topk of them.
 *                 HCA  per-layer KV compression. compress_ratios[L] is 0, 4 or
 *                      128; 0 means the layer runs uncompressed sliding-window
 *                      attention and has NO indexer at all.
 *   Residual    mHC. Not one residual stream but hc_mult=4 of them, mixed each
 *               block by a Sinkhorn-normalised (doubly stochastic) matrix.
 *   MoE         top-6 of n_routed_experts, sqrtsoftplus scores, noaux_tc bias,
 *               plus ONE shared expert. No dense-layer prefix: every layer is MoE.
 *   Weights     experts FP4 (float4_e2m1fn_x2), everything else FP8 e4m3 with
 *               ue8m0 scales over 128x128 blocks.
 *
 * THE INVARIANTS  (each is a place where a plausible implementation runs, emits
 * fluent text, and is wrong, with no crash and no NaN)
 *
 *   1. compress_ratios is PER LAYER, its zeros are load-bearing, and it is
 *      LONGER than the decoder. It holds num_hidden_layers +
 *      num_nextn_predict_layers entries, because it covers the MTP module too,
 *      and that trailing entry is 0 in both released models. Read it as if it
 *      were n_layers long and you invent a dense-attention layer at the end that
 *      does not exist. Measured from the released files:
 *
 *        Flash  44 entries, 43 decoder + 1 MTP -> 2 dense (layers 0,1),
 *               21 indexed, 20 compressed-128
 *        Pro    62 entries, 61 decoder + 1 MTP -> 0 dense,
 *               30 indexed, 31 compressed-128
 *
 *      Pro has NO dense-attention layer at all. A reader that takes the first
 *      element, or treats 0 as "unset", silently runs every layer down the wrong
 *      attention path.
 *   2. The indexer runs ONLY where compress_ratio == 4. Running it everywhere
 *      costs accuracy and time; running it nowhere costs accuracy alone.
 *   3. scoring_func and topk_method are NOT defaultable. sqrtsoftplus and
 *      sigmoid produce similarly-scaled routing weights, so substituting one
 *      reorders top-k on a minority of rows and degrades quality without ever
 *      looking broken.
 *   4. THERE ARE TWO ROUTING MODES. model.py:556 is
 *        self.hash = layer_id < args.n_hash_layers
 *      and num_hash_layers is 3 in both released models, so layers 0-2 pick
 *      their experts by a static token-id lookup (ffn.gate.tid2eid, I64
 *      [vocab, topk]) while layers 3+ use scored top-k with ffn.gate.bias
 *      (F32 [n_experts]). VERIFIED in the checkpoint: layer 2 carries tid2eid
 *      and no bias; layer 3 carries bias and no tid2eid.
 *
 *      A binder that assumes one mode everywhere fails on the other half. Note
 *      also that model.py still computes scores from gate.weight on hash
 *      layers -- only the INDICES bypass top-k, the combining weights do not.
 *
 *      Consequence worth exploiting: on layers 0-2 the six experts are known
 *      the instant the token id is, before any compute, so their reads can be
 *      issued at token start and fully overlapped.
 */
#ifndef DSV4_H
#define DSV4_H

#include <stdint.h>
#include <stddef.h>

/* Bounds the fixed-size routing arrays. Both released models use top-6. */
#define DSV4_MAX_TOPK      16

/* Prompt tokens processed together by the batched (GEMM) prefill path.
 *
 * A cap and not a tuning parameter. It bounds a per-thread stack array in the
 * batched matmuls, and it bounds the KV rows a single batched attention step
 * has to reason about. Prompts longer than this are prefilled in chunks of
 * this size, which changes nothing about the result -- the reuse a GEMM gets
 * is already near its ceiling well before 64 (measured 7.99x at a 207-token
 * prompt, and the curve is a logarithm, not a line).
 *
 * Raising it costs stack in every matmul thread and widens the batched
 * activation buffers; it does not unlock proportional speed. */
#define DSV4_MAX_BATCH     64
#define DSV4_MAX_LAYERS    128
#define DSV4_MAX_HC_MULT   8

/* Router score function. Never defaulted: see invariant 3. */
enum { DSV4_SCORE_SQRTSOFTPLUS = 1, DSV4_SCORE_SIGMOID = 2, DSV4_SCORE_SOFTMAX = 3 };
/* Top-k selection method. noaux_tc = aux-loss-free bias steers SELECTION ONLY. */
enum { DSV4_TOPK_NOAUX_TC = 1, DSV4_TOPK_GREEDY = 2 };
/* Weight storage. DSV4_WF32 is zero so a memset struct keeps fp32 behaviour. */
enum { DSV4_WF32 = 0, DSV4_WBF16 = 1, DSV4_WFP8 = 2, DSV4_WFP4 = 3 };

typedef struct {
    int   hidden;             /* 7168 Pro / 4096 Flash                          */
    int   n_layers;           /* 61 / 43                                        */
    int   vocab;              /* 129280                                         */
    float rms_eps;            /* 1e-6                                           */
    int   max_pos;            /* 1048576                                        */

    /* ---- attention: MLA-shaped, ONE kv head, q and o both low-rank ---- */
    int   n_heads;            /* 128 / 64                                       */
    int   n_kv_heads;         /* 1                                              */
    int   head_dim;           /* 512                                            */
    int   q_lora;             /* 1536 / 1024                                    */
    int   o_lora;             /* 1024                                           */
    int   o_groups;           /* 16 / 8                                         */
    int   qk_rope;            /* 64                                             */
    int   sliding_window;     /* 128                                            */

    /* ---- CSA: the lightning indexer ---- */
    int   index_n_heads;      /* 64                                             */
    int   index_head_dim;     /* 128                                            */
    int   index_topk;         /* 1024 Pro / 512 Flash                           */

    /* ---- HCA: per-layer KV compression. See invariant 1. ---- */
    int   n_compress;         /* MUST equal n_layers                            */
    int  *compress_ratios;    /* 0 | 4 | 128, one per layer                     */
    float compress_rope_theta;/* 160000                                         */

    /* ---- mHC: manifold-constrained hyper-connections ---- */
    int   hc_mult;            /* 4 parallel residual streams                    */
    int   hc_sinkhorn_iters;  /* 20                                             */
    float hc_eps;             /* 1e-6                                           */

    /* ---- MoE: every layer, no dense prefix ---- */
    int   n_experts;          /* 384 / 256                                      */
    int   topk;               /* 6                                              */
    int   n_shared;           /* 1                                              */
    int   moe_inter;          /* 3072 / 2048                                    */
    float routed_scale;       /* 2.5 / 1.5                                      */
    int   norm_topk;          /* 1                                              */
    int   scoring_func;       /* DSV4_SCORE_*                                   */
    int   topk_method;        /* DSV4_TOPK_*                                    */

    /* ---- activation ---- */
    float swiglu_limit;       /* 10.0, a CLAMP not a scale                      */

    /* ---- RoPE with YaRN scaling ---- */
    float rope_theta;         /* 10000                                          */
    float yarn_factor;        /* 16                                             */
    float yarn_beta_fast;     /* 32                                             */
    float yarn_beta_slow;     /* 1                                              */
    int   yarn_orig_ctx;      /* 65536                                          */

    /* ---- quantisation ---- */
    int   wblock_m, wblock_n; /* 128 x 128                                      */

    /* ---- routing and MTP ---- */
    /* num_hash_layers IS implemented: layers below it route by tid2eid lookup
     * (dsv4_ops.c:112) and the whole-model oracle exercises one. Only the MTP
     * module is skipped -- it predicts extra tokens and is not part of the
     * single-token forward pass. */
    int   num_hash_layers;    /* 3                                              */
    int   num_nextn_predict;  /* 1: MTP module, skipped at inference            */
} DSV4Cfg;

/* Layer predicates. compress_ratios is per layer and ZERO-based here. */
static inline int dsv4_compress_ratio(const DSV4Cfg *c, int layer)
{
    if (layer < 0 || layer >= c->n_compress) return -1;
    return c->compress_ratios[layer];
}
/* Uncompressed sliding-window attention, no indexer. See invariant 1. */
static inline int dsv4_is_dense_attn(const DSV4Cfg *c, int layer)
{
    return dsv4_compress_ratio(c, layer) == 0;
}
/* The indexer runs here and ONLY here. See invariant 2. */
static inline int dsv4_has_indexer(const DSV4Cfg *c, int layer)
{
    return dsv4_compress_ratio(c, layer) == 4;
}

/* Routing mode. See invariant 4 and model.py:556. A hash-routed layer carries
 * ffn.gate.tid2eid and NO ffn.gate.bias; a scored layer is the reverse. Both
 * carry ffn.gate.weight, because scores are computed either way. */
static inline int dsv4_is_hash_routed(const DSV4Cfg *c, int layer)
{
    return layer >= 0 && layer < c->n_layers && layer < c->num_hash_layers;
}

/* ---------------------------------------------------------------- weights ---
 * Every shape below was read out of the released DeepSeek-V4-Flash checkpoint
 * headers and is annotated with both the literal Flash value and the config
 * expression it comes from, so Pro is the same code with different numbers.
 *
 * mHC geometry, from model.py:663-667 -- NOT guessed:
 *     mix_hc = (2 + hc_mult) * hc_mult      Flash: (2+4)*4 = 24
 *     hc_dim = hc_mult * hidden             Flash: 4*4096  = 16384
 * so hc_attn_fn / hc_ffn_fn are [24, 16384] and the head's is [hc_mult, hc_dim].
 */
static inline int dsv4_mix_hc(const DSV4Cfg *c) { return (2 + c->hc_mult) * c->hc_mult; }
static inline int dsv4_hc_dim(const DSV4Cfg *c) { return c->hc_mult * c->hidden; }

/* A matrix that carries its own block scales.
 *
 * THE TWO BLOCK GEOMETRIES ARE NOT INTERCHANGEABLE and this struct exists to
 * stop them being confused:
 *   DSV4_WFP8  F8_E4M3 weights, F8_E8M0 scales, 128 x 128 blocks. `rows`/`cols`
 *              are element counts and equal the stored byte counts.
 *   DSV4_WFP4  I8 storage holding TWO 4-bit values per byte, F8_E8M0 scales,
 *              1 x 32 blocks. `cols` is the LOGICAL element count, so the
 *              stored row is cols/2 bytes wide. Getting this backwards reads
 *              half a row and still produces finite numbers.
 * A plain BF16/F32 matrix sets s = NULL and leaves blk_* at 0.
 */
typedef struct {
    const void *w;          /* packed weight bytes, never widened            */
    const void *s;          /* F8_E8M0 block scales, or NULL                 */
    int         wdt;        /* DSV4_WF32 | DSV4_WBF16 | DSV4_WFP8 | DSV4_WFP4 */
    int         rows, cols; /* LOGICAL element dims                          */
    int         blk_r, blk_c;
} DSV4QMat;

/* HCA compressor. Present only where compress_ratio != 0 (model.py:466).
 * coff = 2 when the ratio is 4 (overlapped), else 1; it widens the projection,
 * which is why layer 2's wkv is [1024, 4096] and layer 3's is [512, 4096]. */
typedef struct {
    const float *ape;       /* F32 [ratio, coff*head_dim]                    */
    const float *norm;      /* widened from BF16 [head_dim]                  */
    DSV4QMat     wkv;       /* BF16 [coff*head_dim, hidden]                  */
    DSV4QMat     wgate;     /* BF16 [coff*head_dim, hidden]                  */
    int          ratio, coff;
} DSV4CompressorW;

/* CSA indexer. Present ONLY where compress_ratio == 4 -- invariant 2, and
 * confirmed in the checkpoint: layer 2 has these, layer 3 does not. */
typedef struct {
    DSV4QMat        wq_b;         /* FP8 [index_n_heads*index_head_dim, q_lora] = [8192,1024] */
    DSV4QMat        weights_proj; /* BF16 [index_n_heads, hidden] = [64, 4096]  */
    DSV4CompressorW comp;         /* its own compressor, narrower than the main one */
} DSV4IndexerW;

typedef struct {
    const float *sink;      /* F32 [n_heads], attention sinks                */
    const float *q_norm;    /* widened from BF16 [q_lora]                    */
    const float *kv_norm;   /* widened from BF16 [head_dim]                  */
    DSV4QMat     wq_a;      /* FP8 [q_lora, hidden]            = [1024, 4096] */
    DSV4QMat     wq_b;      /* FP8 [n_heads*head_dim, q_lora]  = [32768,1024] */
    DSV4QMat     wkv;       /* FP8 [head_dim, hidden]          = [512,  4096] */
    DSV4QMat     wo_a;      /* FP8 [o_lora*o_groups, n_heads*head_dim/o_groups] = [8192,4096] */
    DSV4QMat     wo_b;      /* FP8 [hidden, o_lora*o_groups]   = [4096, 8192] */
    DSV4CompressorW comp;   /* valid when has_comp                            */
    DSV4IndexerW    idx;    /* valid when has_idx                             */
    int has_comp, has_idx;
} DSV4AttnW;

/* One routed expert, as it sits in the cache: still packed FP4, never widened.
 * Measured: 13,369,344 B on disk per expert for Flash. Widening to f32 would be
 * 8x that, and a token touches 6 per layer across 43 layers. */
typedef struct {
    DSV4QMat w1, w3;        /* FP4 [moe_inter, hidden]                       */
    DSV4QMat w2;            /* FP4 [hidden, moe_inter]                       */
} DSV4ExpertW;

typedef struct {
    DSV4QMat       gate;    /* BF16 [n_experts, hidden]; scores ALWAYS computed */
    const float   *bias;    /* F32 [n_experts] on scored layers, else NULL   */
    const int64_t *tid2eid; /* I64 [vocab, topk] on hash layers, else NULL   */
    DSV4ExpertW    shared;  /* the one shared expert, FP8 not FP4, resident  */
    /* Routed experts are streamed per token and are deliberately absent here. */
} DSV4MoeW;

/* mHC mixing parameters. Two sets per block, one for attention, one for FFN. */
typedef struct {
    const float *fn;        /* F32 [mix_hc, hc_dim]  = [24, 16384]           */
    const float *base;      /* F32 [mix_hc]          = [24]                  */
    const float *scale;     /* F32 [3] on a block, [1] on the head           */
} DSV4HcW;

typedef struct {
    const float *attn_norm; /* widened from BF16 [hidden]                    */
    const float *ffn_norm;  /* widened from BF16 [hidden]                    */
    DSV4AttnW    attn;
    DSV4MoeW     moe;
    DSV4HcW      hc_attn, hc_ffn;
    int          layer;
    int          compress_ratio;
    int          hash_routed;   /* layer < num_hash_layers: tid2eid, no bias */
    int          has_comp;      /* compress_ratio != 0                       */
    int          has_idx;       /* compress_ratio == 4                       */
} DSV4LayerW;

/* Model-level weights. All six live outside layers.* and mtp.*:
 * embed.weight (shard 1) and norm/head/hc_head_* (shard 45). Both embed and
 * head are BF16 [vocab, hidden] = 0.986 GB each; tie_word_embeddings is false,
 * so they are genuinely two tensors. */
typedef struct {
    const void  *embed;     /* BF16 [vocab, hidden]                          */
    const void  *head;      /* BF16 [vocab, hidden]                          */
    const float *norm;      /* widened from BF16 [hidden]                    */
    DSV4HcW      hc_head;   /* fn is [hc_mult, hc_dim], scale is [1]         */
    int          wdt;
} DSV4ModelW;

/* Gather one embedding row, widening if the table is BF16. The table is INDEXED,
 * not multiplied, so it cannot go through a matmul dispatch: a memcpy with a
 * float stride would read half a row of the wrong values. */
static inline void dsv4_embed_row(float *dst, const void *table, int wdt,
                                  int64_t row, int hidden)
{
    if (wdt == DSV4_WBF16) {
        const uint16_t *p = (const uint16_t *)table + row * hidden;
        for (int i = 0; i < hidden; i++) {
            union { uint32_t u; float f; } v;
            v.u = (uint32_t)p[i] << 16;
            dst[i] = v.f;
        }
    } else {
        const float *p = (const float *)table + row * hidden;
        for (int i = 0; i < hidden; i++) dst[i] = p[i];
    }
}

#endif /* DSV4_H */
