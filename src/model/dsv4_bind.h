/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind.h - bind real checkpoint tensors into the engine's weight structs.
 *
 * Ported in structure from kimi-k3-in-c/src/model/k3_bind.c (Apache-2.0). The
 * central idea is that project's and is kept: ONE plan_layer() declares what a
 * layer needs, and every consumer -- binding from shards, and later binding from
 * a streamed trunk run -- walks that same plan. Two independent name lists would
 * drift, and the drift would show up as a transposed or truncated matrix rather
 * than as an error.
 *
 * WHAT IS DIFFERENT HERE
 *
 * 1. THREE STORAGE CLASSES, NOT TWO. K3 chose between "keep bf16" and "widen to
 *    fp32". DeepSeek-V4 adds quantised pairs: an F8_E4M3 or packed-FP4 weight is
 *    meaningless without its F8_E8M0 scale, which is a SEPARATE TENSOR. So the
 *    unit of binding for those is a DSV4QMat holding both, requested together by
 *    reqq(), and the two can never be bound independently or half-bound.
 *
 * 2. LAYERS ARE NOT ALL THE SAME SHAPE. Verified in the released checkpoint:
 *
 *      compress_ratio 0    no compressor, no indexer     (Flash layers 0,1)
 *      compress_ratio 4    compressor + INDEXER          (21 of Flash's layers)
 *      compress_ratio 128  compressor, no indexer        (20 of Flash's layers)
 *      layer < num_hash_layers   ffn.gate.tid2eid, NO ffn.gate.bias
 *      layer >= num_hash_layers  ffn.gate.bias, NO tid2eid
 *
 *    A plan that requests every tensor unconditionally fails on four of these
 *    five layer kinds. plan_layer() therefore branches on dsv4_compress_ratio(),
 *    dsv4_has_indexer() and dsv4_is_hash_routed(), and requesting a tensor that
 *    should be absent is as much a bug as omitting one that should be present.
 *
 * 3. ROUTED EXPERTS ARE NEVER BOUND. They are streamed per token: 13,369,344
 *    bytes each, 256 per layer, 137 GB in total for Flash. Only the ONE shared
 *    expert is resident, and it is FP8 rather than FP4.
 *
 * 4. THE MTP BLOCK IS SKIPPED. mtp.0.* is a complete extra layer (1,575 tensors,
 *    with its own e_proj/h_proj/enorm/hnorm/norm) used for multi-token
 *    prediction. This engine does not implement speculative decoding, so nothing
 *    here binds it. compress_ratios covers it, which is why dsv4_cfg narrows the
 *    layer map to n_layers -- see invariant 1.
 */
#ifndef DSV4_BIND_H
#define DSV4_BIND_H

#include "dsv4.h"
#include "dsv4_st.h"

typedef struct {
    void      *blob;      /* one allocation holding every resident weight of this
                           * layer, MIXED narrow and fp32, each tensor 8-aligned */
    size_t     nbytes;
    int        layer;
    DSV4LayerW w;
} DSV4LayerBind;

/* Bind one decoder layer. Returns 0 on success.
 * Routed expert pointers are deliberately absent from DSV4LayerW: those are
 * streamed per token, not resident. */
int  dsv4_bind_layer(const DSV4St *s, const DSV4Cfg *c, int layer, DSV4LayerBind *b);
void dsv4_bind_free(DSV4LayerBind *b);

/* Bytes one layer needs, without reading any. For sizing and reporting. */
int64_t dsv4_bind_layer_bytes(const DSV4St *s, const DSV4Cfg *c, int layer);

typedef struct {
    void      *blob;
    size_t     nbytes;
    DSV4ModelW w;
} DSV4ModelBind;

/* Model-level weights: embed, final norm, head, and the head's mHC parameters.
 * embed and head are 0.986 GB each at BF16 and tie_word_embeddings is false, so
 * they are two distinct tensors. Pass want_head = 0 when only the trunk is being
 * exercised. */
int  dsv4_bind_model(const DSV4St *s, const DSV4Cfg *c, int want_head,
                     DSV4ModelBind *m);
void dsv4_bind_model_free(DSV4ModelBind *m);

#endif /* DSV4_BIND_H */
