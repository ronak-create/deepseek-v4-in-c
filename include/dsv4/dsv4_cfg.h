/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_cfg.h - build a DSV4Cfg from config.json, without ever guessing.
 *
 * THE ONE RULE, INHERITED FROM kimi-k3-in-c: AN ABSENT FIELD IS AN ERROR,
 * NEVER A DEFAULT.
 *
 * It is worth restating for this architecture because the failure is worse here
 * than it was for K3. DeepSeek-V4's config is FLAT, so a reader written against
 * a nested schema does not fail loudly, it misses every key. If each miss took a
 * plausible default:
 *
 *   - swiglu_limit would default to 10.0, which is CORRECT, so the activation
 *     looks right and inspires confidence in everything read after it;
 *   - compress_ratios would come back empty, so dsv4_is_dense_attn() answers
 *     true for every layer and the entire model runs uncompressed attention with
 *     no indexer -- a DIFFERENT model that still emits fluent English;
 *   - hc_mult would default to 1, collapsing mHC to an ordinary residual stream,
 *     which is exactly what a reader author would guess and exactly wrong.
 *
 * Nothing crashes. No number looks out of place. The output is simply not this
 * model's. So a missing key fails the load, and every missing key is collected
 * and reported together rather than one per run.
 *
 * STRING ENUMS ARE VALIDATED, NOT MAPPED. scoring_func, topk_method and the
 * quantisation format names are compared against the values this engine actually
 * implements. An unrecognised value is a hard failure, because DeepSeek shipping
 * a "sigmoid" variant must stop the load rather than quietly run sqrtsoftplus.
 */
#ifndef DSV4_CFG_H
#define DSV4_CFG_H

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "json.h"
#include "dsv4.h"

typedef struct {
    jval       *root;      /* the config object; this schema is flat        */
    jval       *quant;     /* quantization_config                           */
    jval       *rope;      /* rope_scaling                                  */
    const char *missing[48];
    int         nmissing;
} DSV4CfgSrc;

static inline void dsv4cfg_miss(DSV4CfgSrc *s, const char *name)
{
    if (s->nmissing < (int)(sizeof s->missing / sizeof s->missing[0]))
        s->missing[s->nmissing] = name;
    s->nmissing++;
}

/* Look up `name` in `obj`, falling back to the root. Passing NULL for obj means
 * "root only", which is the common case in this flat schema. */
static inline jval *dsv4cfg_find(DSV4CfgSrc *s, jval *obj, const char *name)
{
    jval *v = obj ? json_get(obj, name) : NULL;
    return v ? v : json_get(s->root, name);
}

static inline int dsv4cfg_i(DSV4CfgSrc *s, jval *obj, const char *name)
{
    jval *v = dsv4cfg_find(s, obj, name);
    if (!v || v->t != J_NUM) { dsv4cfg_miss(s, name); return 0; }
    return (int)v->num;
}

static inline float dsv4cfg_f(DSV4CfgSrc *s, jval *obj, const char *name)
{
    jval *v = dsv4cfg_find(s, obj, name);
    if (!v || v->t != J_NUM) { dsv4cfg_miss(s, name); return 0.0f; }
    return (float)v->num;
}

/* Booleans only. A numeric 0/1 is accepted because hand-edited configs use it. */
static inline int dsv4cfg_b(DSV4CfgSrc *s, jval *obj, const char *name)
{
    jval *v = dsv4cfg_find(s, obj, name);
    if (!v) { dsv4cfg_miss(s, name); return 0; }
    if (v->t == J_BOOL) return v->boolean ? 1 : 0;
    if (v->t == J_NUM)  return v->num != 0.0;
    dsv4cfg_miss(s, name);
    return 0;
}

/* A string field matched against the values this engine implements. `table` is
 * NULL-terminated pairs of {spelling, code}. An unknown spelling returns -1 and
 * names what IS supported, because that is the actionable part. */
typedef struct { const char *name; int code; } DSV4Enum;

static inline int dsv4cfg_enum(DSV4CfgSrc *s, jval *obj, const char *name,
                               const DSV4Enum *table, const char *whence)
{
    jval *v = dsv4cfg_find(s, obj, name);
    if (!v || v->t != J_STR) { dsv4cfg_miss(s, name); return 0; }
    for (int i = 0; table[i].name; i++)
        if (!strcmp(v->str, table[i].name)) return table[i].code;

    fprintf(stderr, "dsv4_cfg: %s has %s = \"%s\", which this engine does not "
                    "implement.\n  supported:", whence, name, v->str);
    for (int i = 0; table[i].name; i++) fprintf(stderr, " %s", table[i].name);
    fprintf(stderr, "\n  refusing to substitute a different one: it would run a "
                    "DIFFERENT model that still emits fluent text.\n");
    return -1;
}

/* Load a DSV4Cfg. `cr` receives the per-layer compression ratios and must hold
 * at least cr_max entries. Returns 1 on success, 0 on any failure.
 *
 * Callers MUST NOT proceed on 0. A half-filled DSV4Cfg is precisely the
 * silently-wrong-model case this file exists to prevent. */
static inline int dsv4_cfg_load(DSV4Cfg *c, int *cr, int cr_max,
                                jval *root, const char *whence)
{
    /* THESE TABLES LIST WHAT HAS A KERNEL, NOT WHAT THE FORMAT CAN EXPRESS.
     * A value belongs here only once something implements it. Listing the wider
     * space -- sigmoid and softmax scoring, greedy top-k -- would let a config
     * naming one of them load cleanly and then run sqrtsoftplus anyway, which is
     * the exact silent substitution this file exists to prevent. Add a row in
     * the same commit that adds its kernel, never before. */
    static const DSV4Enum score_tbl[] = {
        { "sqrtsoftplus", DSV4_SCORE_SQRTSOFTPLUS },
        { NULL, 0 }
    };
    static const DSV4Enum topk_tbl[] = {
        { "noaux_tc", DSV4_TOPK_NOAUX_TC },
        { NULL, 0 }
    };
    /* Only these storage formats have kernels. Anything else must stop the load
     * rather than be read as the format that happens to share its width. */
    static const DSV4Enum expert_dt_tbl[] = { { "fp4",   DSV4_WFP4 }, { NULL, 0 } };
    static const DSV4Enum quant_fmt_tbl[] = { { "e4m3",  1 },         { NULL, 0 } };
    static const DSV4Enum scale_fmt_tbl[] = { { "ue8m0", 1 },         { NULL, 0 } };

    memset(c, 0, sizeof *c);

    DSV4CfgSrc s; memset(&s, 0, sizeof s);
    s.root  = root;
    s.quant = json_get(root, "quantization_config");
    s.rope  = json_get(root, "rope_scaling");

    /* Reject a config for a different architecture before reading anything from
     * it. Field names collide across DeepSeek generations; model_type does not. */
    jval *mt = json_get(root, "model_type");
    if (!mt || mt->t != J_STR || strcmp(mt->str, "deepseek_v4")) {
        fprintf(stderr, "dsv4_cfg: %s has model_type \"%s\", expected "
                        "\"deepseek_v4\"\n", whence,
                (mt && mt->t == J_STR) ? mt->str : "(absent)");
        return 0;
    }

    c->hidden    = dsv4cfg_i(&s, NULL, "hidden_size");
    c->n_layers  = dsv4cfg_i(&s, NULL, "num_hidden_layers");
    c->vocab     = dsv4cfg_i(&s, NULL, "vocab_size");
    c->rms_eps   = dsv4cfg_f(&s, NULL, "rms_norm_eps");
    c->max_pos   = dsv4cfg_i(&s, NULL, "max_position_embeddings");

    /* Attention */
    c->n_heads        = dsv4cfg_i(&s, NULL, "num_attention_heads");
    c->n_kv_heads     = dsv4cfg_i(&s, NULL, "num_key_value_heads");
    c->head_dim       = dsv4cfg_i(&s, NULL, "head_dim");
    c->q_lora         = dsv4cfg_i(&s, NULL, "q_lora_rank");
    c->o_lora         = dsv4cfg_i(&s, NULL, "o_lora_rank");
    c->o_groups       = dsv4cfg_i(&s, NULL, "o_groups");
    c->qk_rope        = dsv4cfg_i(&s, NULL, "qk_rope_head_dim");
    c->sliding_window = dsv4cfg_i(&s, NULL, "sliding_window");

    /* CSA */
    c->index_n_heads  = dsv4cfg_i(&s, NULL, "index_n_heads");
    c->index_head_dim = dsv4cfg_i(&s, NULL, "index_head_dim");
    c->index_topk     = dsv4cfg_i(&s, NULL, "index_topk");

    /* HCA */
    c->compress_rope_theta = dsv4cfg_f(&s, NULL, "compress_rope_theta");

    /* mHC */
    c->hc_mult           = dsv4cfg_i(&s, NULL, "hc_mult");
    c->hc_sinkhorn_iters = dsv4cfg_i(&s, NULL, "hc_sinkhorn_iters");
    c->hc_eps            = dsv4cfg_f(&s, NULL, "hc_eps");

    /* MoE */
    c->n_experts    = dsv4cfg_i(&s, NULL, "n_routed_experts");
    c->topk         = dsv4cfg_i(&s, NULL, "num_experts_per_tok");
    c->n_shared     = dsv4cfg_i(&s, NULL, "n_shared_experts");
    c->moe_inter    = dsv4cfg_i(&s, NULL, "moe_intermediate_size");
    c->routed_scale = dsv4cfg_f(&s, NULL, "routed_scaling_factor");
    c->norm_topk    = dsv4cfg_b(&s, NULL, "norm_topk_prob");

    c->swiglu_limit = dsv4cfg_f(&s, NULL, "swiglu_limit");

    /* RoPE. The YaRN parameters live under rope_scaling; type is checked. */
    c->rope_theta = dsv4cfg_f(&s, NULL, "rope_theta");
    if (!s.rope) {
        dsv4cfg_miss(&s, "rope_scaling");
    } else {
        jval *ty = json_get(s.rope, "type");
        if (!ty || ty->t != J_STR || strcmp(ty->str, "yarn")) {
            fprintf(stderr, "dsv4_cfg: %s rope_scaling.type is \"%s\", expected "
                            "\"yarn\"\n", whence,
                    (ty && ty->t == J_STR) ? ty->str : "(absent)");
            return 0;
        }
        c->yarn_factor    = dsv4cfg_f(&s, s.rope, "factor");
        c->yarn_beta_fast = dsv4cfg_f(&s, s.rope, "beta_fast");
        c->yarn_beta_slow = dsv4cfg_f(&s, s.rope, "beta_slow");
        c->yarn_orig_ctx  = dsv4cfg_i(&s, s.rope, "original_max_position_embeddings");
    }

    c->num_hash_layers   = dsv4cfg_i(&s, NULL, "num_hash_layers");
    c->num_nextn_predict = dsv4cfg_i(&s, NULL, "num_nextn_predict_layers");

    /* String enums. A -1 means "recognised the key, refused the value", which
     * has already printed its own diagnostic. */
    c->scoring_func = dsv4cfg_enum(&s, NULL, "scoring_func", score_tbl, whence);
    c->topk_method  = dsv4cfg_enum(&s, NULL, "topk_method",  topk_tbl,  whence);
    if (c->scoring_func < 0 || c->topk_method < 0) return 0;

    if (dsv4cfg_enum(&s, NULL, "expert_dtype", expert_dt_tbl, whence) < 0) return 0;

    /* Quantisation block. weight_block_size is [m, n]. */
    if (!s.quant) {
        dsv4cfg_miss(&s, "quantization_config");
    } else {
        if (dsv4cfg_enum(&s, s.quant, "fmt",       quant_fmt_tbl, whence) < 0) return 0;
        if (dsv4cfg_enum(&s, s.quant, "scale_fmt", scale_fmt_tbl, whence) < 0) return 0;
        jval *wb = json_get(s.quant, "weight_block_size");
        if (!wb || wb->t != J_ARR || wb->len != 2 ||
            wb->kids[0]->t != J_NUM || wb->kids[1]->t != J_NUM) {
            dsv4cfg_miss(&s, "quantization_config.weight_block_size");
        } else {
            c->wblock_m = (int)wb->kids[0]->num;
            c->wblock_n = (int)wb->kids[1]->num;
        }
    }

    /* compress_ratios. Per layer, and its zeros are load-bearing -- see
     * invariant 1 in dsv4.h. NOTE the length: the list covers the MTP module as
     * well as the decoder, so it holds n_layers + num_nextn_predict_layers
     * entries, not n_layers. Both released models carry exactly one trailing
     * entry, and that entry is 0 in both -- which is why reading it as a decoder
     * layer invents a dense-attention layer that does not exist. */
    jval *crj = json_get(root, "compress_ratios");
    if (!crj || crj->t != J_ARR || crj->len == 0) {
        dsv4cfg_miss(&s, "compress_ratios");
    } else if (crj->len > cr_max) {
        fprintf(stderr, "dsv4_cfg: %s lists %d compress ratios, buffer holds %d\n",
                whence, crj->len, cr_max);
        return 0;
    } else {
        for (int i = 0; i < crj->len; i++) {
            if (crj->kids[i]->t != J_NUM) {
                fprintf(stderr, "dsv4_cfg: %s compress_ratios[%d] is not a number\n",
                        whence, i);
                return 0;
            }
            cr[i] = (int)crj->kids[i]->num;
        }
        c->n_compress      = crj->len;
        c->compress_ratios = cr;
    }

    if (s.nmissing) {
        fprintf(stderr, "dsv4_cfg: %s is missing %d required field(s):\n",
                whence, s.nmissing);
        int cap = (int)(sizeof s.missing / sizeof s.missing[0]);
        int shown = s.nmissing < cap ? s.nmissing : cap;
        for (int i = 0; i < shown; i++) fprintf(stderr, "    %s\n", s.missing[i]);
        if (s.nmissing > shown)
            fprintf(stderr, "    ... and %d more\n", s.nmissing - shown);
        fprintf(stderr, "  refusing to substitute defaults: a config this reader "
                        "cannot\n  fully understand would silently produce a "
                        "DIFFERENT model.\n");
        return 0;
    }

    /* ---- structural checks ----
     * A config can parse cleanly and still be unable to describe this model. */
    if (c->n_layers <= 0 || c->hidden <= 0 || c->vocab <= 0) {
        fprintf(stderr, "dsv4_cfg: %s has non-positive layers/hidden/vocab\n", whence);
        return 0;
    }
    if (c->n_layers > DSV4_MAX_LAYERS) {
        fprintf(stderr, "dsv4_cfg: %s has %d layers, this build supports %d\n"
                        "  (DSV4_MAX_LAYERS in dsv4.h)\n",
                whence, c->n_layers, DSV4_MAX_LAYERS);
        return 0;
    }
    /* Invariant 1: one ratio per decoder layer PLUS one per MTP layer, no more
     * and no fewer. A short list is the single most likely way to run most
     * layers down the wrong path; a list read as if it were n_layers long
     * invents a trailing dense-attention layer, because the MTP entry is 0. */
    if (c->num_nextn_predict < 0) {
        fprintf(stderr, "dsv4_cfg: %s has num_nextn_predict_layers %d\n",
                whence, c->num_nextn_predict);
        return 0;
    }
    if (c->n_compress != c->n_layers + c->num_nextn_predict) {
        fprintf(stderr, "dsv4_cfg: %s has %d compress_ratios for %d decoder "
                        "layers + %d MTP layer(s); expected %d\n",
                whence, c->n_compress, c->n_layers, c->num_nextn_predict,
                c->n_layers + c->num_nextn_predict);
        return 0;
    }
    /* WHAT THE RATIO ACTUALLY SELECTS. This once rejected anything but 0, 4 and
     * 128, on the reasoning that those are the only values the two released
     * checkpoints use. That conflated "values I have seen" with "values that
     * have a kernel" -- the one thing this validator exists not to do. Grepping
     * for the constant found 128 in the error message and nowhere else:
     * dsv4_compress.c pools over `ratio` tokens for any ratio, and dsv4_bind.c
     * sizes the ape as ratio*coff*head_dim. Only three cases are structurally
     * distinct, and two of them are exact values:
     *
     *     0        no Compressor at all             (model.py:466)
     *     4        Compressor + Indexer, overlapped (model.py:290)
     *     other    Compressor alone, non-overlapped
     *
     * A negative ratio stays an error: dsv4_compress_ratio() returns it verbatim
     * and callers divide by it. */
    for (int i = 0; i < c->n_compress; i++) {
        if (cr[i] < 0) {
            fprintf(stderr, "dsv4_cfg: %s compress_ratios[%d] = %d; a ratio is a"
                            " token count and cannot be negative\n",
                    whence, i, cr[i]);
            return 0;
        }
    }
    /* Having validated the whole list, narrow the layer map to the decoder. The
     * MTP entries stay in `cr` but out of reach of dsv4_compress_ratio(), so no
     * caller can mistake the MTP block for layer n_layers. */
    c->n_compress = c->n_layers;

    if (c->topk > DSV4_MAX_TOPK) {
        fprintf(stderr, "dsv4_cfg: %s selects top-%d, but this build supports at "
                        "most %d\n  (DSV4_MAX_TOPK in dsv4.h bounds the "
                        "fixed-size routing arrays)\n",
                whence, c->topk, DSV4_MAX_TOPK);
        return 0;
    }
    if (c->topk > c->n_experts || c->topk <= 0) {
        fprintf(stderr, "dsv4_cfg: %s selects %d of %d experts\n",
                whence, c->topk, c->n_experts);
        return 0;
    }
    if (c->hc_mult < 1 || c->hc_mult > DSV4_MAX_HC_MULT) {
        fprintf(stderr, "dsv4_cfg: %s has hc_mult %d, outside 1..%d\n",
                whence, c->hc_mult, DSV4_MAX_HC_MULT);
        return 0;
    }
    if (c->hc_sinkhorn_iters < 1) {
        fprintf(stderr, "dsv4_cfg: %s has hc_sinkhorn_iters %d; the mixing "
                        "matrix would never be normalised\n",
                whence, c->hc_sinkhorn_iters);
        return 0;
    }
    if (c->head_dim <= 0 || c->n_heads <= 0 || c->n_kv_heads <= 0) {
        fprintf(stderr, "dsv4_cfg: %s has non-positive head geometry\n", whence);
        return 0;
    }
    if (c->o_groups <= 0 || c->n_heads % c->o_groups != 0) {
        fprintf(stderr, "dsv4_cfg: %s has %d heads in %d output groups; heads "
                        "must divide evenly\n", whence, c->n_heads, c->o_groups);
        return 0;
    }
    /* Invariant 4: hash routing covers a PREFIX of the layers and cannot cover
     * all of them. A model with no scored layer would carry no ffn.gate.bias
     * anywhere, which is a different architecture, not this one. */
    if (c->num_hash_layers < 0 || c->num_hash_layers >= c->n_layers) {
        fprintf(stderr, "dsv4_cfg: %s has num_hash_layers %d for %d layers; it "
                        "must be a proper prefix (0 <= n < n_layers)\n",
                whence, c->num_hash_layers, c->n_layers);
        return 0;
    }
    if (c->swiglu_limit <= 0.0f) {
        fprintf(stderr, "dsv4_cfg: %s has swiglu_limit %g; it is a CLAMP and "
                        "must be positive\n", whence, (double)c->swiglu_limit);
        return 0;
    }
    if (c->wblock_m <= 0 || c->wblock_n <= 0) {
        fprintf(stderr, "dsv4_cfg: %s has weight_block_size [%d, %d]\n",
                whence, c->wblock_m, c->wblock_n);
        return 0;
    }

    int n_dense = 0, n_idx = 0, n_cmp = 0, cmp_ratio = 0;
    for (int i = 0; i < c->n_compress; i++) {
        if      (cr[i] == 0) n_dense++;
        else if (cr[i] == 4) n_idx++;
        else               { n_cmp++; cmp_ratio = cr[i]; }
    }

    printf("config: %s | hidden=%d layers=%d vocab=%d | heads=%d kv=%d dim=%d\n"
           "        attention: %d dense + %d indexed(top%d) + %d compressed(1:%d)\n"
           "        routing  : %d hash (tid2eid) + %d scored (gate.bias)\n"
           "        experts %d top%d shared%d inter=%d | mHC x%d (%d sinkhorn iters)\n",
           whence, c->hidden, c->n_layers, c->vocab,
           c->n_heads, c->n_kv_heads, c->head_dim,
           n_dense, n_idx, c->index_topk, n_cmp, cmp_ratio,
           c->num_hash_layers, c->n_layers - c->num_hash_layers,
           c->n_experts, c->topk, c->n_shared, c->moe_inter,
           c->hc_mult, c->hc_sinkhorn_iters);
    return 1;
}

/* Read, parse and load a config file. The arena is left allocated because
 * DSV4Cfg does not copy the strings it does not own. */
static inline int dsv4_cfg_load_file(DSV4Cfg *c, int *cr, int cr_max,
                                     const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) { perror(path); return 0; }
    if (fseek(f, 0, SEEK_END) != 0) { perror(path); fclose(f); return 0; }
    long n = ftell(f);
    if (n < 0 || n > (1L << 28)) {
        fprintf(stderr, "%s: implausible config size %ld\n", path, n);
        fclose(f); return 0;
    }
    if (fseek(f, 0, SEEK_SET) != 0) { perror(path); fclose(f); return 0; }
    char *txt = (char *)malloc((size_t)n + 1);
    if (!txt) { fprintf(stderr, "OOM reading %s\n", path); fclose(f); return 0; }
    size_t got = fread(txt, 1, (size_t)n, f);
    fclose(f);
    txt[got] = 0;

    char *arena = NULL;
    jval *root = json_parse(txt, &arena);
    if (!root) { fprintf(stderr, "%s: not valid JSON\n", path); free(txt); return 0; }
    return dsv4_cfg_load(c, cr, cr_max, root, path);
}

#endif /* DSV4_CFG_H */
