/* SPDX-License-Identifier: Apache-2.0 */
/* test_model_oracle.c - the WHOLE MODEL against the PyTorch reference.
 *
 * WHAT THIS CATCHES THAT NOTHING ELSE DOES
 *   test_oracle.c proves each kernel matches. test_layer_oracle.c proves a block
 *   matches. Neither touches the model-level path, which runs once per token and
 *   had never been compared against anything:
 *
 *     - the embedding gather
 *     - the expansion of one vector into hc_mult identical copies
 *     - carrying hc_mult streams ACROSS layer boundaries
 *     - two different RoPE tables selected per layer by compress_ratio
 *     - per-layer KV and compressor state that must not leak between layers
 *     - the head's own hc reduction, which is NOT the block's
 *     - the final norm and the lm_head
 *
 *   It also runs the real loader: config reader, safetensors reader, binder,
 *   trunk streamer and expert cache, on a checkpoint written in the released
 *   format. So a mistake anywhere from "open the file" to "emit a logit" shows
 *   up here as a number that disagrees.
 *
 * THE FIXTURE IS EXACT BY CONSTRUCTION. Every FP8 and FP4 weight is drawn from
 * the set its format represents exactly, so the reference and the engine see
 * byte-identical values and any disagreement is real rather than rounding.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#include "dsv4_cfg.h"
#include "dsv4_trunk.h"
#include "dsv4_cache.h"
#include "dsv4_layer.h"
#include "json.h"

void dsv4_rmsnorm(float *, const float *, const float *, int, float);
void dsv4_mmq(float *, const float *, const DSV4QMat *);
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

static DSV4Cache *g_cache;
static const DSV4ExpertW *ge(void *ctx, int L, int e)
{
    (void)ctx; return dsv4_cache_get(g_cache, L, e);
}

/* The overlapped pair, wired in so this gate actually covers it.
 *
 * It did not, before: the source here was built with .get alone, so both the
 * per-token path and the batched one took their non-overlapped branch and the
 * bit-exactness claim was never checked against the code that really runs. The
 * engine has been bitten by exactly that shape of gap before -- a gate that
 * cannot reach the path it claims to protect. */
static DSV4CacheBatch g_obatch;

static int ge_begin(void *ctx, int L, const int *e, int n,
                    const DSV4ExpertW **out)
{
    (void)ctx;
    return dsv4_cache_get_many_begin(g_cache, L, e, n, out, &g_obatch);
}

static int ge_end(void *ctx, const DSV4ExpertW **out)
{
    (void)ctx;
    return dsv4_cache_get_many_end(g_cache, out, &g_obatch);
}

/* The head's hc reduction: a sigmoid gate, NO Sinkhorn and no comb matrix, and
 * a single scalar hc_scale where a block has three. Duplicated from the CLI on
 * purpose -- if the two ever diverge this gate keeps measuring the reference
 * rather than whatever the CLI happens to do. */
static void hc_head(float *y, const float *x, const DSV4HcW *w,
                    int hc, int hidden, float norm_eps, float hc_eps)
{
    const int hcdim = hc * hidden;
    double sq = 0.0;
    for (int i = 0; i < hcdim; i++) sq += (double)x[i] * x[i];
    const float rs = (float)(1.0 / sqrt(sq / (double)hcdim + (double)norm_eps));
    float pre[DSV4_MAX_HC_MULT];
    for (int j = 0; j < hc; j++) {
        const float *row = w->fn + (size_t)j * hcdim;
        double acc = 0.0;
        for (int i = 0; i < hcdim; i++) acc = fma((double)row[i], (double)x[i], acc);
        pre[j] = 1.0f / (1.0f + expf(-((float)acc * rs * w->scale[0] + w->base[j])))
               + hc_eps;
    }
    for (int i = 0; i < hidden; i++) {
        double acc = 0.0;
        for (int j = 0; j < hc; j++)
            acc += (double)pre[j] * (double)x[(size_t)j * hidden + i];
        y[i] = (float)acc;
    }
}

/* Compare one stage of position 0 against the reference and say so. Returns 1
 * on agreement. Reporting the FIRST stage that drifts is the whole point: once
 * a layer is wrong every later number is wrong too, and a list of twelve failing
 * logits says nothing about which of forty steps caused them. */
static int stage(jval *trace, const char *name, const float *got, int n)
{
    if (!trace) return 1;
    jval *want = json_get(trace, name);
    if (!want) return 1;
    if (want->len != n) {
        printf("  FAIL  %s: reference has %d values, C produced %d\n",
               name, want->len, n);
        fails++; return 0;
    }
    double sq = 0.0;
    for (int i = 0; i < n; i++) sq += want->kids[i]->num * want->kids[i]->num;
    const float rms = (float)sqrt(sq / n);
    float worst = 0.0f; int at = 0;
    for (int i = 0; i < n; i++) {
        const float r = (float)want->kids[i]->num;
        const float sc = fmaxf(rms, fmaxf(fabsf(got[i]), fabsf(r)));
        const float rel = sc > 0 ? fabsf(got[i] - r) / sc : 0.0f;
        if (rel > worst) { worst = rel; at = i; }
    }
    if (worst > 5e-6f) {
        printf("  FAIL  %-8s diverges: rel %.3g at [%d]  C %.6g vs torch %.6g\n",
               name, (double)worst, at, (double)got[at], (double)want->kids[at]->num);
        fails++; return 0;
    }
    printf("  ok    %-8s %3d values, worst rel %.2g\n", name, n, (double)worst);
    return 1;
}

int main(int argc, char **argv)
{
    const char *dir   = (argc > 1) ? argv[1] : getenv("DSV4_TINY");
    const char *trunk = (argc > 2) ? argv[2] : getenv("DSV4_TINY_TRUNK");
    struct stat sb;

    printf("DeepSeek-V4 WHOLE-MODEL oracle gate\n");
    if (!dir || !trunk || stat(dir, &sb) != 0 || stat(trunk, &sb) != 0) {
        printf("  SKIP  no tiny checkpoint. Build one with:\n");
        printf("          ~/venv-cuda/bin/python tools/make_tiny_checkpoint.py ~/dsv4-tiny\n");
        printf("          python3 tools/pack_trunk.py ~/dsv4-tiny ~/dsv4-tiny-trunk\n");
        printf("\nMODEL ORACLE GATE SKIPPED\n");
        return 0;
    }

    char p[1024];
    DSV4Cfg c; int cr[DSV4_MAX_LAYERS];
    snprintf(p, sizeof p, "%s/config.json", dir);
    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS, p)) return 1;

    DSV4St st;
    if (dsv4_st_open(&st, dir) != 0) { printf("  FAIL  st_open\n"); return 1; }

    snprintf(p, sizeof p, "%s/expected.json", dir);
    char *txt = slurp(p);
    if (!txt) { printf("  FAIL  expected.json missing\n"); return 1; }
    char *arena = NULL;
    jval *G = json_parse(txt, &arena);
    jval *toks = json_get(G, "tokens");
    jval *trace = json_get(G, "trace");
    jval *lg   = json_get(G, "logits");

    DSV4Trunk tr;
    if (dsv4_trunk_open(&tr, trunk, &c, 64LL << 20) != 0) {
        printf("  FAIL  trunk_open\n"); return 1;
    }
    static DSV4Cache cache;
    g_cache = &cache;
    if (dsv4_cache_init(&cache, &st, &c, 64LL << 20) != 0) {
        printf("  FAIL  cache_init\n"); return 1;
    }

    DSV4ModelBind mb;
    if (dsv4_bind_model(&st, &c, 1, &mb) != 0) { printf("  FAIL  bind_model\n"); return 1; }

    const int maxpos = 256;
    const int half = c.qk_rope / 2;
    float *cs_d = malloc((size_t)maxpos * half * 4), *sn_d = malloc((size_t)maxpos * half * 4);
    float *cs_c = malloc((size_t)maxpos * half * 4), *sn_c = malloc((size_t)maxpos * half * 4);
    dsv4_rope_table(cs_d, sn_d, c.qk_rope, maxpos, 0, c.rope_theta,
                    c.yarn_factor, c.yarn_beta_fast, c.yarn_beta_slow);
    dsv4_rope_table(cs_c, sn_c, c.qk_rope, maxpos, c.yarn_orig_ctx,
                    c.compress_rope_theta, c.yarn_factor,
                    c.yarn_beta_fast, c.yarn_beta_slow);

    DSV4Scratch s;
    if (dsv4_scratch_init(&s, &c, maxpos) != 0) { printf("  FAIL  scratch\n"); return 1; }
    DSV4LayerState *ls = calloc((size_t)c.n_layers, sizeof(DSV4LayerState));
    for (int L = 0; L < c.n_layers; L++) dsv4_state_init(&ls[L], &c, L, maxpos);
    /* Named, so adding a member to DSV4ExpertSrc cannot silently
     * shift what this gate is testing. get() only: these gates check
     * the arithmetic, and the fetch path has its own. */
    DSV4ExpertSrc src = { .get = ge, .begin = ge_begin, .end = ge_end };

    const int hc = c.hc_mult, d = c.hidden;
    float *h = calloc((size_t)hc * d, sizeof(float));
    float *y = calloc((size_t)d, sizeof(float));
    float *logits = calloc((size_t)c.vocab, sizeof(float));
    /* Every position's logits, kept so the batched prefill below can be
     * compared against the run that was just validated against torch. */
    float *ref_logits = calloc((size_t)toks->len * c.vocab, sizeof(float));

    printf("\n-- %d positions through %d layers (%s) --\n",
           toks->len, c.n_layers, "config, safetensors, binder, trunk, cache, all real");

    for (int pos = 0; pos < toks->len; pos++) {
        const int tid = (int)toks->kids[pos]->num;

        dsv4_embed_row(y, mb.w.embed, mb.w.wdt, tid, d);
        for (int j = 0; j < hc; j++)
            memcpy(h + (size_t)j * d, y, (size_t)d * sizeof(float));
        if (pos == 0 && !stage(trace, "embed", h, hc * d)) goto done;

        for (int L = 0; L < c.n_layers; L++) {
            DSV4LayerBind lb;
            if (dsv4_trunk_bind(&tr, &c, L, &lb) != 0) {
                CHECK(0, "trunk_bind failed on layer %d", L); goto done;
            }
            const int comp = dsv4_compress_ratio(&c, L) != 0;
            dsv4_layer_forward(h, &lb.w, &c, &s, &src, &ls[L],
                               comp ? cs_c : cs_d, comp ? sn_c : sn_d, pos, tid);
            if (pos == 0) {
                char nm[32]; snprintf(nm, sizeof nm, "layer%d", L);
                if (!stage(trace, nm, h, hc * d)) goto done;
            }
        }

        hc_head(y, h, &mb.w.hc_head, hc, d, c.rms_eps, c.hc_eps);
        if (pos == 0 && !stage(trace, "hc_head", y, d)) goto done;
        dsv4_rmsnorm(y, y, mb.w.norm, d, c.rms_eps);
        if (pos == 0 && !stage(trace, "norm", y, d)) goto done;
        DSV4QMat head = { mb.w.head, NULL, DSV4_WBF16, c.vocab, d, 0, 0 };
        dsv4_mmq(logits, y, &head);
        memcpy(ref_logits + (size_t)pos * c.vocab, logits,
               (size_t)c.vocab * sizeof(float));

        /* Compare against the reference, scaled by the vector rather than by
         * each element -- see test_oracle.c for why. */
        jval *row = lg->kids[pos];
        double sq = 0.0;
        for (int v = 0; v < c.vocab; v++) sq += row->kids[v]->num * row->kids[v]->num;
        const float rms = (float)sqrt(sq / c.vocab);
        float worst = 0.0f; int at = -1;
        for (int v = 0; v < c.vocab; v++) {
            const float r = (float)row->kids[v]->num;
            const float dd = fabsf(logits[v] - r);
            const float sc = fmaxf(rms, fmaxf(fabsf(logits[v]), fabsf(r)));
            const float rel = sc > 0 ? dd / sc : 0.0f;
            if (rel > worst) { worst = rel; at = v; }
        }

        int c_arg = 0, r_arg = 0;
        for (int v = 1; v < c.vocab; v++) {
            if (logits[v] > logits[c_arg]) c_arg = v;
            if (row->kids[v]->num > row->kids[r_arg]->num) r_arg = v;
        }
        /* THE ARGMAX IS THE CLAIM THAT MATTERS. A logit off by 1e-4 changes
         * nothing; a different argmax is a different generated token, and every
         * token after it diverges. */
        CHECK(c_arg == r_arg,
              "pos %d: argmax %d, reference %d -- a DIFFERENT TOKEN", pos, c_arg, r_arg);
        if (worst > 2e-3f)
            CHECK(0, "pos %d: worst rel %.3g at vocab %d (C %.6g vs torch %.6g)",
                  pos, (double)worst, at, (double)logits[at],
                  (double)row->kids[at]->num);
        else if (c_arg == r_arg)
            printf("  ok    pos %d  argmax %-3d  worst rel err %.2g over %d logits\n",
                   pos, c_arg, (double)worst, c.vocab);
    }

    /* ---------------------------------------------------------------------
     * BATCHED PREFILL, against the run that was just checked against torch.
     *
     * This is the gate for dsv4_layer_forward_n, and it is deliberately the
     * strictest kind available: bit equality against the per-token path, on the
     * real loader and the real tiny checkpoint, after the per-token path has
     * itself been validated against a PyTorch reference. So "batched agrees
     * with per-token" is not agreement between two things that could be wrong
     * together -- one end of it is anchored.
     *
     * Several chunk sizes on purpose. Chunk 1 must land on the same code as
     * decode. Chunks that do not divide the prompt length exercise the ragged
     * last chunk, and chunk boundaries are exactly where a position-ordered
     * bug in the sliding ring or the compressor would hide: get pos0 wrong and
     * only the first chunk is right.
     */
    if (!fails) {
        printf("\n-- batched prefill vs per-token, bit for bit --\n");
        const int chunks[4] = { 1, 2, 3, toks->len };
        float *hb = calloc((size_t)toks->len * hc * d, sizeof(float));
        float *yb = calloc((size_t)d, sizeof(float));
        float *lb = calloc((size_t)c.vocab, sizeof(float));
        int32_t *tid_buf = calloc((size_t)toks->len, sizeof(int32_t));
        DSV4Batch bat;

        for (int ci = 0; ci < 4 && hb && yb && lb && tid_buf; ci++) {
            const int nchunk = chunks[ci];
            if (nchunk < 1 || nchunk > DSV4_MAX_BATCH) continue;
            if (ci > 0 && nchunk == chunks[ci - 1]) continue;

            if (dsv4_batch_init(&bat, &c, nchunk) != 0) {
                CHECK(0, "batch_init failed at chunk %d", nchunk); break;
            }
            /* Fresh layer state: the sliding ring and the compressor carry
             * position-ordered history, and reusing the per-token run's would
             * make this compare two different sequences. */
            for (int L = 0; L < c.n_layers; L++) {
                dsv4_state_free(&ls[L]);
                dsv4_state_init(&ls[L], &c, L, maxpos);
            }

            int bad = 0;
            for (int p0 = 0; p0 < toks->len && !bad; p0 += nchunk) {
                const int nt = (toks->len - p0 < nchunk) ? toks->len - p0
                                                         : nchunk;
                for (int t = 0; t < nt; t++) {
                    tid_buf[t] = (int32_t)toks->kids[p0 + t]->num;
                    dsv4_embed_row(yb, mb.w.embed, mb.w.wdt,
                                   (int)tid_buf[t], d);
                    for (int j = 0; j < hc; j++)
                        memcpy(hb + (size_t)t * hc * d + (size_t)j * d, yb,
                               (size_t)d * sizeof(float));
                }

                for (int L = 0; L < c.n_layers; L++) {
                    DSV4LayerBind lb2;
                    if (dsv4_trunk_bind(&tr, &c, L, &lb2) != 0) {
                        CHECK(0, "trunk_bind failed on layer %d", L);
                        bad = 1; break;
                    }
                    const int comp = dsv4_compress_ratio(&c, L) != 0;
                    dsv4_layer_forward_n(hb, &lb2.w, &c, &s, &bat, &src, &ls[L],
                                         comp ? cs_c : cs_d,
                                         comp ? sn_c : sn_d,
                                         p0, tid_buf, nt);
                }
                if (bad) break;

                for (int t = 0; t < nt; t++) {
                    hc_head(yb, hb + (size_t)t * hc * d, &mb.w.hc_head,
                            hc, d, c.rms_eps, c.hc_eps);
                    dsv4_rmsnorm(yb, yb, mb.w.norm, d, c.rms_eps);
                    DSV4QMat head2 = { mb.w.head, NULL, DSV4_WBF16,
                                       c.vocab, d, 0, 0 };
                    dsv4_mmq(lb, yb, &head2);
                    const float *ref = ref_logits + (size_t)(p0 + t) * c.vocab;
                    if (memcmp(lb, ref, (size_t)c.vocab * sizeof(float)) != 0) {
                        int at = 0;
                        for (int v = 0; v < c.vocab; v++)
                            if (lb[v] != ref[v]) { at = v; break; }
                        CHECK(0, "chunk %d, pos %d: logit %d is %.17g batched "
                                 "vs %.17g per-token", nchunk, p0 + t, at,
                              (double)lb[at], (double)ref[at]);
                        bad = 1; break;
                    }
                }
            }
            if (!bad)
                printf("  ok    chunk %-3d: all %d positions bit-identical to "
                       "the per-token path\n", nchunk, toks->len);
            dsv4_batch_free(&bat);
        }
        free(hb); free(yb); free(lb); free(tid_buf);
    }

done:
    for (int L = 0; L < c.n_layers; L++) dsv4_state_free(&ls[L]);
    free(ls); free(h); free(y); free(logits); free(ref_logits);
    free(cs_d); free(sn_d); free(cs_c); free(sn_c);
    dsv4_scratch_free(&s); dsv4_bind_model_free(&mb);
    dsv4_cache_free(&cache); dsv4_trunk_close(&tr); dsv4_st_close(&st);

    printf("\n");
    if (fails) { printf("MODEL ORACLE GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("MODEL ORACLE GATE PASSED\n");
    return 0;
}
