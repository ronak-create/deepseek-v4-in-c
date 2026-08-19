/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_run.c - the dsv4 binary: load, prompt, decode, report.
 *
 * SCOPE, stated plainly rather than discovered later
 *   Greedy decoding only. No temperature, no top-p, no chat template. Greedy is
 *   what makes the output identical across memory budgets, which every gate in
 *   this repo depends on, and DeepSeek-V4-Flash ships no Jinja template anyway:
 *   the released encoding/ folder is the message-formatting path. So this
 *   produces base-model continuations, not replies.
 *
 *   No chunked prefill. The prompt is fed one token at a time through the same
 *   decode path the generated tokens use, which is correct and not fast.
 *
 * TWO ROPE TABLES, NOT ONE
 *   model.py Attention.__init__ builds freqs_cis with compress_rope_theta and
 *   YaRN enabled for any compress_ratio != 0, and with rope_theta and YaRN
 *   DISABLED for ratio 0. One table for the whole model is wrong on one set of
 *   layers or the other, silently.
 */
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <time.h>

#include "dsv4_cfg.h"
#include "dsv4_bind.h"
#include "dsv4_trunk.h"
#include "dsv4_cache.h"
#include "dsv4_layer.h"
#include "dsv4_tok.h"

void dsv4_rmsnorm(float *, const float *, const float *, int, float);
void dsv4_mmq(float *, const float *, const DSV4QMat *);
void dsv4_rope_table(float *, float *, int, int, int, double, double, double, double);
int  dsv4_tok_load(DSV4Tok *, const char *);
void dsv4_tok_free(DSV4Tok *);
int  dsv4_tok_encode(const DSV4Tok *, const char *, int, int32_t *, int);

/* The expert source every layer is handed. The cache travels through ctx rather
 * than a file-scope global, so two models could run in one process. */
static const DSV4ExpertW *dsv4_run_get_expert(void *ctx, int layer, int e)
{
    return dsv4_cache_get((DSV4Cache *)ctx, layer, e);
}

static double now_s(void)
{
    struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + 1e-9 * (double)ts.tv_nsec;
}

static double peak_rss_gb(void)
{
    struct rusage ru;
    getrusage(RUSAGE_SELF, &ru);
    return (double)ru.ru_maxrss / 1048576.0;   /* Linux reports KB */
}

/* The head's own hc reduction. NOT Block.hc_pre: there is no Sinkhorn here and
 * no comb matrix, just a sigmoid gate. hc_scale is a single scalar applied to
 * every mix, where a block's is three. */
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
        const float m = (float)acc * rs;
        pre[j] = 1.0f / (1.0f + expf(-(m * w->scale[0] + w->base[j]))) + hc_eps;
    }
    for (int d = 0; d < hidden; d++) {
        double acc = 0.0;
        for (int j = 0; j < hc; j++)
            acc += (double)pre[j] * (double)x[(size_t)j * hidden + d];
        y[d] = (float)acc;
    }
}

static void usage(const char *me)
{
    printf("usage: %s <model_dir> --trunk <dir> --tok <file> "
           "[--prompt TEXT] [--gen N] [--budget GB]\n", me);
}

int main(int argc, char **argv)
{
    const char *model = NULL, *trunkdir = NULL, *tokfile = NULL;
    const char *prompt = "The capital of France is";
    int ngen = 8;
    double budget_gb = 8.0;

    for (int i = 1; i < argc; i++) {
        if (argv[i][0] != '-' && !model)      { model = argv[i]; continue; }
        if (!strcmp(argv[i], "--trunk")  && i + 1 < argc) trunkdir = argv[++i];
        else if (!strcmp(argv[i], "--tok")   && i + 1 < argc) tokfile = argv[++i];
        else if (!strcmp(argv[i], "--prompt")&& i + 1 < argc) prompt  = argv[++i];
        else if (!strcmp(argv[i], "--gen")   && i + 1 < argc) ngen    = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--budget")&& i + 1 < argc) budget_gb = atof(argv[++i]);
        else if (!strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
    }
    if (!model || !trunkdir || !tokfile) { usage(argv[0]); return 2; }

    char path[1024];
    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    snprintf(path, sizeof path, "%s/config.json", model);
    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS, path)) return 1;

    DSV4St st;
    if (dsv4_st_open(&st, model) != 0) return 1;

    DSV4Tok tok;
    if (dsv4_tok_load(&tok, tokfile) != 0) return 1;
    if ((int)tok.n_vocab != c.vocab) {
        fprintf(stderr, "tokenizer has %u ids, config says %d; refusing to run "
                        "a mismatched pair\n", tok.n_vocab, c.vocab);
        return 1;
    }

    const int64_t budget = (int64_t)(budget_gb * 1073741824.0);
    /* Split the budget: the trunk gets the larger share because a pinned layer
     * is a guaranteed hit, while an expert slot only helps if that expert is
     * routed to again. */
    DSV4Trunk tr;
    if (dsv4_trunk_open(&tr, trunkdir, &c, budget * 3 / 4) != 0) return 1;
    DSV4Cache cache;
    if (dsv4_cache_init(&cache, &st, &c, budget / 4) != 0) return 1;

    DSV4ModelBind mb;
    if (dsv4_bind_model(&st, &c, 1, &mb) != 0) return 1;

    /* Two tables: see the note at the top. */
    const int maxpos = c.sliding_window + 4096;
    float *cs_d = malloc((size_t)maxpos * (c.qk_rope / 2) * sizeof(float));
    float *sn_d = malloc((size_t)maxpos * (c.qk_rope / 2) * sizeof(float));
    float *cs_c = malloc((size_t)maxpos * (c.qk_rope / 2) * sizeof(float));
    float *sn_c = malloc((size_t)maxpos * (c.qk_rope / 2) * sizeof(float));
    dsv4_rope_table(cs_d, sn_d, c.qk_rope, maxpos, 0, c.rope_theta,
                    c.yarn_factor, c.yarn_beta_fast, c.yarn_beta_slow);
    dsv4_rope_table(cs_c, sn_c, c.qk_rope, maxpos, c.yarn_orig_ctx,
                    c.compress_rope_theta, c.yarn_factor,
                    c.yarn_beta_fast, c.yarn_beta_slow);

    DSV4Scratch scratch;
    if (dsv4_scratch_init(&scratch, &c) != 0) return 1;

    DSV4LayerState *lst = calloc((size_t)c.n_layers, sizeof(DSV4LayerState));
    for (int L = 0; L < c.n_layers; L++)
        if (dsv4_state_init(&lst[L], &c, L, maxpos) != 0) return 1;

    DSV4ExpertSrc src = { dsv4_run_get_expert, &cache };

    int32_t ids[8192];
    const int nprompt = dsv4_tok_encode(&tok, prompt, (int)strlen(prompt),
                                        ids, 8192);
    printf("\nprompt: \"%s\"  -> %d tokens\n", prompt, nprompt);
    if (nprompt == 0) { fprintf(stderr, "empty prompt\n"); return 1; }

    const int hc = c.hc_mult, d = c.hidden;
    float *h = calloc((size_t)hc * d, sizeof(float));
    float *xh = calloc((size_t)d, sizeof(float));
    float *logits = calloc((size_t)c.vocab, sizeof(float));

    printf("--- generated ids ---\n");
    const double t0 = now_s();
    int ntok = 0;

    for (int pos = 0; pos < nprompt + ngen; pos++) {
        const int32_t tid = (pos < nprompt) ? ids[pos] : ids[pos];

        /* embed, then expand to hc_mult identical copies (model.py Transformer:
         * h.unsqueeze(2).repeat(1,1,hc_mult,1)) */
        dsv4_embed_row(xh, mb.w.embed, mb.w.wdt, tid, d);
        for (int j = 0; j < hc; j++) memcpy(h + (size_t)j * d, xh, (size_t)d * sizeof(float));

        for (int L = 0; L < c.n_layers; L++) {
            DSV4LayerBind lb;
            if (dsv4_trunk_bind(&tr, &c, L, &lb) != 0) return 1;
            const int comp = dsv4_compress_ratio(&c, L) != 0;
            dsv4_layer_forward(h, &lb.w, &c, &scratch, &src, &lst[L],
                               comp ? cs_c : cs_d, comp ? sn_c : sn_d,
                               pos, tid);
        }

        if (pos + 1 >= nprompt) {
            hc_head(xh, h, &mb.w.hc_head, hc, d, c.rms_eps, c.hc_eps);
            dsv4_rmsnorm(xh, xh, mb.w.norm, d, c.rms_eps);
            DSV4QMat head = { mb.w.head, NULL, DSV4_WBF16, c.vocab, d, 0, 0 };
            dsv4_mmq(logits, xh, &head);

            int best = 0;
            for (int v = 1; v < c.vocab; v++) if (logits[v] > logits[best]) best = v;
            if (pos + 1 < nprompt + ngen) ids[pos + 1] = best;
            printf(" %d", best);
            fflush(stdout);
            ntok++;
        }
    }
    const double dt = now_s() - t0;

    printf("\n\n%d tokens in %.1f s, %.2f s/token\n", ntok, dt,
           ntok ? dt / ntok : 0.0);
    dsv4_trunk_report(&tr, "run");
    dsv4_cache_report(&cache, "run");
    printf("PEAK RSS for the whole run: %.2f GB\n", peak_rss_gb());

    for (int L = 0; L < c.n_layers; L++) dsv4_state_free(&lst[L]);
    free(lst); free(h); free(xh); free(logits);
    free(cs_d); free(sn_d); free(cs_c); free(sn_c);
    dsv4_scratch_free(&scratch);
    dsv4_bind_model_free(&mb);
    dsv4_cache_free(&cache);
    dsv4_trunk_close(&tr);
    dsv4_tok_free(&tok);
    dsv4_st_close(&st);
    return 0;
}
