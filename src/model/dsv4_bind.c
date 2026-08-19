/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_bind.c - see dsv4_bind.h for the design and for what differs from K3.
 *
 * THE CHECK THAT EARNS ITS KEEP is in plan_resolve: every tensor's element count
 * is compared against what the CONFIG implies before a single byte is read. The
 * expectations are expressions of config fields, never literals, so pointing the
 * engine at Pro exercises the same code with different numbers. A checkpoint that
 * disagrees with its own config then fails at load, loudly, instead of feeding
 * every kernel downstream the wrong strides while producing plausible output.
 */
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "dsv4_bind.h"

#define MAXB 96      /* the widest layer, a ratio-4 indexed one, needs 46 */

typedef struct {
    char              name[224];
    const DSV4Tensor *t;
    int64_t           want;    /* STORED elements expected; -1 accepts anything */
    int               narrow;  /* 1 = keep checkpoint bytes, 0 = widen to fp32   */
    const void      **dest;
    size_t            off;     /* byte offset into the blob, filled while sizing */
} Req;

typedef struct {
    Req r[MAXB];
    int n;
    int bad;
} Plan;

static void req_(Plan *p, const void **dest, int narrow, int64_t want,
                 const char *fmt, va_list ap)
{
    if (p->n >= MAXB) { fprintf(stderr, "dsv4_bind: too many tensors\n"); p->bad++; return; }
    Req *q = &p->r[p->n];
    vsnprintf(q->name, sizeof q->name, fmt, ap);
    q->dest = dest; q->want = want; q->narrow = narrow;
    q->t = NULL; q->off = 0;
    p->n++;
}

/* WIDE: widened to fp32. Used for the small vectors that kernels index
 * ELEMENTWISE (norms, sinks, gate bias, the mHC parameters), where a silently
 * wrong element type is not a crash but a different model. */
static void reqw(Plan *p, const float **dest, int64_t want, const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    req_(p, (const void **)dest, 0, want, fmt, ap);
    va_end(ap);
}

/* NARROW: kept as the checkpoint's own bytes, widened inside the matmul. */
static void reqn(Plan *p, const void **dest, int64_t want, const char *fmt, ...)
{
    va_list ap; va_start(ap, fmt);
    req_(p, dest, 1, want, fmt, ap);
    va_end(ap);
}

static size_t align8(size_t x) { return (x + 7u) & ~(size_t)7u; }
static int64_t ceil_div(int64_t a, int64_t b) { return (a + b - 1) / b; }

/* QUANTISED PAIR. A weight and its block scales are requested together and land
 * in one DSV4QMat, because binding either alone is never correct.
 *
 * `rows` and `cols` are LOGICAL element dimensions. The stored width differs by
 * class and that difference is the whole point of this function:
 *
 *   DSV4_WFP8   stored [rows, cols] bytes,  scale [rows/128, cols/128]
 *   DSV4_WFP4   stored [rows, cols/2] BYTES (two values per byte),
 *                                          scale [rows, cols/32]
 *   DSV4_WBF16  stored [rows, cols],        no scale tensor at all
 *
 * Verified against the released checkpoint: wq_a is F8_E4M3 [1024,4096] with
 * scale [8,32] (= 1024/128, 4096/128), and an expert w1 is I8 [2048,2048] with
 * scale [2048,128] (= 2048/1, 4096/32).
 */
/* NOTE reqq is varargs like reqw and reqn, deliberately. An earlier version took
 * a plain `const char *base` while its siblings took format strings, and callers
 * passing "%s.wkv" got a tensor literally named "%s.wkv.weight". Keeping all
 * three request functions to one calling convention removes that whole class. */
static void reqq(Plan *p, DSV4QMat *m, int wdt, int64_t rows, int64_t cols,
                 const char *fmt, ...)
{
    char base[192];
    va_list ap; va_start(ap, fmt);
    vsnprintf(base, sizeof base, fmt, ap);
    va_end(ap);

    memset(m, 0, sizeof *m);
    m->wdt = wdt; m->rows = (int)rows; m->cols = (int)cols;

    int64_t stored;
    switch (wdt) {
    case DSV4_WFP8:
        m->blk_r = 128; m->blk_c = 128;
        stored = rows * cols;
        break;
    case DSV4_WFP4:
        /* 1x32 blocks, NOT the 128x128 in quantization_config.weight_block_size.
         * That field describes the FP8 tensors; applying it here misreads every
         * scale while the byte counts still add up. */
        m->blk_r = 1; m->blk_c = 32;
        stored = rows * (cols / 2);
        break;
    default:                      /* BF16 or F32: no scale partner */
        m->blk_r = 0; m->blk_c = 0;
        stored = rows * cols;
        break;
    }

    reqn(p, &m->w, stored, "%s.weight", base);
    if (m->blk_c) {
        const int64_t ns = ceil_div(rows, m->blk_r) * ceil_div(cols, m->blk_c);
        reqn(p, &m->s, ns, "%s.scale", base);
    }
}

/* ------------------------------------------------------------------ plan ---- */

/* The compressor's projections widen by coff, which is 2 on the overlapped
 * ratio-4 path and 1 otherwise (model.py:290, self.overlap = ratio == 4).
 * Confirmed in the checkpoint: layer 2 (ratio 4) has wkv [1024,4096] and layer 3
 * (ratio 128) has [512,4096], against head_dim 512. */
static int comp_coff(int ratio) { return ratio == 4 ? 2 : 1; }

static void plan_compressor(Plan *p, const DSV4Cfg *c, DSV4CompressorW *w,
                            int ratio, int head_dim, const char *base)
{
    const int coff = comp_coff(ratio);
    w->ratio = ratio;
    w->coff  = coff;
    reqw(p, &w->ape,  (int64_t)ratio * coff * head_dim, "%s.ape",         base);
    reqw(p, &w->norm, head_dim,                         "%s.norm.weight", base);
    reqq(p, &w->wkv,   DSV4_WBF16, (int64_t)coff * head_dim, c->hidden, "%s.wkv",   base);
    reqq(p, &w->wgate, DSV4_WBF16, (int64_t)coff * head_dim, c->hidden, "%s.wgate", base);
}

static void plan_hc(Plan *p, const DSV4Cfg *c, DSV4HcW *w, int rows, int nscale,
                    const char *base)
{
    reqw(p, &w->fn,    (int64_t)rows * dsv4_hc_dim(c), "%s_fn",    base);
    reqw(p, &w->base,  rows,                           "%s_base",  base);
    reqw(p, &w->scale, nscale,                         "%s_scale", base);
}

/* Everything one decoder layer needs, and NOTHING it must not have. */
static void plan_layer(Plan *p, const DSV4Cfg *c, int L, DSV4LayerW *w)
{
    const int ratio = dsv4_compress_ratio(c, L);
    const int hash  = dsv4_is_hash_routed(c, L);

    memset(w, 0, sizeof *w);
    w->layer = L;
    w->compress_ratio = ratio;
    w->hash_routed = hash;
    w->has_comp = (ratio != 0);
    w->has_idx  = dsv4_has_indexer(c, L);

    reqw(p, &w->attn_norm, c->hidden, "layers.%d.attn_norm.weight", L);
    reqw(p, &w->ffn_norm,  c->hidden, "layers.%d.ffn_norm.weight",  L);

    /* ---- attention ---- */
    DSV4AttnW *a = &w->attn;
    a->has_comp = w->has_comp;
    a->has_idx  = w->has_idx;
    reqw(p, &a->sink,    c->n_heads,  "layers.%d.attn.attn_sink",      L);
    reqw(p, &a->q_norm,  c->q_lora,   "layers.%d.attn.q_norm.weight",  L);
    reqw(p, &a->kv_norm, c->head_dim, "layers.%d.attn.kv_norm.weight", L);

    reqq(p, &a->wq_a, DSV4_WFP8, c->q_lora, c->hidden, "layers.%d.attn.wq_a", L);
    reqq(p, &a->wq_b, DSV4_WFP8, (int64_t)c->n_heads * c->head_dim, c->q_lora,
         "layers.%d.attn.wq_b", L);
    reqq(p, &a->wkv,  DSV4_WFP8, c->head_dim, c->hidden, "layers.%d.attn.wkv", L);
    /* The output projection is low-rank AND grouped: o_lora_rank per group,
     * o_groups groups. Flash: 1024 x 8 = 8192, matching wo_a [8192, 4096]. */
    reqq(p, &a->wo_a, DSV4_WFP8, (int64_t)c->o_lora * c->o_groups, c->hidden,
         "layers.%d.attn.wo_a", L);
    reqq(p, &a->wo_b, DSV4_WFP8, c->hidden, (int64_t)c->o_lora * c->o_groups,
         "layers.%d.attn.wo_b", L);

    /* HCA. Absent on ratio-0 layers -- model.py:466 constructs no Compressor. */
    if (w->has_comp) {
        char nm[192];
        snprintf(nm, sizeof nm, "layers.%d.attn.compressor", L);
        plan_compressor(p, c, &a->comp, ratio, c->head_dim, nm);
    }

    /* CSA. Present ONLY where the ratio is 4. Its compressor is narrower: it
     * works in index_head_dim, not head_dim. */
    if (w->has_idx) {
        char nm[192];
        reqq(p, &a->idx.wq_b, DSV4_WFP8,
             (int64_t)c->index_n_heads * c->index_head_dim, c->q_lora,
             "layers.%d.attn.indexer.wq_b", L);
        reqq(p, &a->idx.weights_proj, DSV4_WBF16, c->index_n_heads, c->hidden,
             "layers.%d.attn.indexer.weights_proj", L);
        snprintf(nm, sizeof nm, "layers.%d.attn.indexer.compressor", L);
        plan_compressor(p, c, &a->idx.comp, ratio, c->index_head_dim, nm);
    }

    /* ---- mHC. mix_hc rows for a block, 3 scales (pre, post, comb). ---- */
    {
        char nm[192];
        snprintf(nm, sizeof nm, "layers.%d.hc_attn", L);
        plan_hc(p, c, &w->hc_attn, dsv4_mix_hc(c), 3, nm);
        snprintf(nm, sizeof nm, "layers.%d.hc_ffn", L);
        plan_hc(p, c, &w->hc_ffn,  dsv4_mix_hc(c), 3, nm);
    }

    /* ---- MoE ---- */
    DSV4MoeW *mo = &w->moe;
    reqq(p, &mo->gate, DSV4_WBF16, c->n_experts, c->hidden, "layers.%d.ffn.gate", L);
    /* Exactly one of these two exists, decided by layer < num_hash_layers.
     * Requesting both fails on every layer; requesting neither loses the routing. */
    if (hash)
        reqn(p, (const void **)&mo->tid2eid, (int64_t)c->vocab * c->topk,
             "layers.%d.ffn.gate.tid2eid", L);
    else
        reqw(p, &mo->bias, c->n_experts, "layers.%d.ffn.gate.bias", L);

    /* The shared expert is FP8 and resident. The routed ones are FP4 and streamed. */
    for (int i = 1; i <= 3; i++) {
        DSV4QMat *q = (i == 1) ? &mo->shared.w1 : (i == 3) ? &mo->shared.w3 : &mo->shared.w2;
        const int64_t inter = (int64_t)c->moe_inter * c->n_shared;
        const char *f = "layers.%d.ffn.shared_experts.w%d";
        if (i == 2) reqq(p, q, DSV4_WFP8, c->hidden, inter, f, L, i);  /* down    */
        else        reqq(p, q, DSV4_WFP8, inter, c->hidden, f, L, i);  /* gate/up */
    }
}

static void plan_model(Plan *p, const DSV4Cfg *c, int want_head, DSV4ModelW *w)
{
    memset(w, 0, sizeof *w);
    w->wdt = DSV4_WBF16;
    reqn(p, &w->embed, (int64_t)c->vocab * c->hidden, "embed.weight");
    reqw(p, &w->norm,  c->hidden,                     "norm.weight");
    if (want_head)
        reqn(p, &w->head, (int64_t)c->vocab * c->hidden, "head.weight");
    /* The head's mHC has hc_mult rows and ONE scale, against a block's mix_hc
     * rows and three (model.py:750-751). */
    plan_hc(p, c, &w->hc_head, c->hc_mult, 1, "hc_head");
}

/* --------------------------------------------------------------- resolve ---- */

/* Resolve and validate everything, and lay out the blob, before reading a byte. */
static int64_t plan_resolve(Plan *p, const DSV4St *s)
{
    size_t off = 0;
    for (int i = 0; i < p->n; i++) {
        Req *q = &p->r[i];
        q->t = dsv4_st_find(s, q->name);
        if (!q->t) {
            fprintf(stderr, "dsv4_bind: missing tensor %s\n", q->name);
            p->bad++;
            continue;
        }
        const int64_t have = dsv4_st_numel(q->t);
        if (q->want >= 0 && have != q->want) {
            fprintf(stderr, "dsv4_bind: %s has %lld elements, config implies %lld\n",
                    q->name, (long long)have, (long long)q->want);
            p->bad++;
            continue;
        }
        /* A widened tensor must actually be widenable. Asking to widen an FP8
         * weight is a planning bug: it has no meaning without its scale. */
        if (!q->narrow && dsv4_st_elemsize(q->t->dtype) > 0 &&
            (q->t->dtype == DSV4_DT_F8_E4M3 || q->t->dtype == DSV4_DT_F8_E8M0 ||
             q->t->dtype == DSV4_DT_I8 || q->t->dtype == DSV4_DT_I64)) {
            fprintf(stderr, "dsv4_bind: %s was planned WIDE but is a quantised or "
                            "index dtype; it must be requested narrow\n", q->name);
            p->bad++;
            continue;
        }
        const size_t bytes = q->narrow
            ? (size_t)q->t->nbytes
            : (size_t)have * sizeof(float);
        q->off = off;
        off = align8(off + bytes);
    }
    return p->bad ? -1 : (int64_t)off;
}

static int plan_load(Plan *p, const DSV4St *s, unsigned char *blob)
{
    for (int i = 0; i < p->n; i++) {
        Req *q = &p->r[i];
        unsigned char *dst = blob + q->off;
        if (q->narrow) {
            if (dsv4_st_read(s, q->t, dst) != q->t->nbytes) {
                fprintf(stderr, "dsv4_bind: short read on %s\n", q->name);
                return -1;
            }
        } else {
            const int64_t n = dsv4_st_numel(q->t);
            if (dsv4_st_read_f32(s, q->t, (float *)dst) != n) {
                fprintf(stderr, "dsv4_bind: short widen on %s\n", q->name);
                return -1;
            }
        }
        *q->dest = dst;
    }
    return 0;
}

/* ------------------------------------------------------------------ api ----- */

int64_t dsv4_bind_layer_bytes(const DSV4St *s, const DSV4Cfg *c, int L)
{
    Plan p; memset(&p, 0, sizeof p);
    DSV4LayerW tmp;
    plan_layer(&p, c, L, &tmp);
    return plan_resolve(&p, s);
}

int dsv4_bind_layer(const DSV4St *s, const DSV4Cfg *c, int L, DSV4LayerBind *b)
{
    memset(b, 0, sizeof *b);
    b->layer = L;

    Plan p; memset(&p, 0, sizeof p);
    plan_layer(&p, c, L, &b->w);
    const int64_t need = plan_resolve(&p, s);
    if (need < 0) {
        fprintf(stderr, "dsv4_bind: layer %d could not be planned\n", L);
        return -1;
    }

    b->blob = malloc((size_t)need);
    if (!b->blob) {
        fprintf(stderr, "dsv4_bind: OOM allocating %lld bytes for layer %d\n",
                (long long)need, L);
        return -1;
    }
    b->nbytes = (size_t)need;

    if (plan_load(&p, s, (unsigned char *)b->blob) != 0) {
        free(b->blob); b->blob = NULL; b->nbytes = 0;
        return -1;
    }
    return 0;
}

void dsv4_bind_free(DSV4LayerBind *b)
{
    if (!b) return;
    free(b->blob);
    memset(b, 0, sizeof *b);
}

int dsv4_bind_model(const DSV4St *s, const DSV4Cfg *c, int want_head,
                    DSV4ModelBind *m)
{
    memset(m, 0, sizeof *m);

    Plan p; memset(&p, 0, sizeof p);
    plan_model(&p, c, want_head, &m->w);
    const int64_t need = plan_resolve(&p, s);
    if (need < 0) { fprintf(stderr, "dsv4_bind: model could not be planned\n"); return -1; }

    m->blob = malloc((size_t)need);
    if (!m->blob) {
        fprintf(stderr, "dsv4_bind: OOM allocating %lld bytes for the model\n",
                (long long)need);
        return -1;
    }
    m->nbytes = (size_t)need;

    if (plan_load(&p, s, (unsigned char *)m->blob) != 0) {
        free(m->blob); m->blob = NULL; m->nbytes = 0;
        return -1;
    }
    return 0;
}

void dsv4_bind_model_free(DSV4ModelBind *m)
{
    if (!m) return;
    free(m->blob);
    memset(m, 0, sizeof *m);
}
