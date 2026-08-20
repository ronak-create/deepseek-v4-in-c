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
#include "dsv4_cuda.h"
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

static int dsv4_run_get_experts(void *ctx, int layer, const int *e, int n,
                                const DSV4ExpertW **out)
{
    return dsv4_cache_get_many((DSV4Cache *)ctx, layer, e, n, out);
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
           "[--prompt TEXT] [--gen N] [--budget GB] [--route-log FILE] [--gpu]\n", me);
}

int main(int argc, char **argv)
{
    const char *model = NULL, *trunkdir = NULL, *tokfile = NULL;
    const char *routelog = NULL;
    int use_gpu = 0;
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
        else if (!strcmp(argv[i], "--route-log") && i + 1 < argc) routelog = argv[++i];
        else if (!strcmp(argv[i], "--gpu")) use_gpu = 1;
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
    /* SPLIT THE BUDGET BY WHAT EACH PART CAN USE, NOT BY A FIXED RATIO.
     *
     * This used to hand the trunk 3/4 and the cache 1/4, reasoning that a
     * pinned layer is a guaranteed hit while an expert slot only pays off on a
     * re-route. True as far as it goes, and it produced a cache that could not
     * work: the trunk SATURATES. Once all 43 layers are pinned there is nothing
     * left to put in a ring slot, so trunk budget beyond ~6.4 GB buys exactly
     * nothing -- while the expert cache was held at 1/4 of 8 GB = 2.0 GB, below
     * the 3.29 GB one forward pass touches. A measured run: 2,580 expert
     * requests, ZERO hits.
     *
     * So: let the trunk take what it can use, and give the cache the rest. */
    DSV4Trunk tr;
    if (dsv4_trunk_open(&tr, trunkdir, &c, budget) != 0) return 1;
    int64_t left = budget - dsv4_trunk_resident_bytes(&tr);
    DSV4Cache cache;
    if (dsv4_cache_init(&cache, &st, &c, left) != 0) return 1;
    if (routelog) {
        cache.route_log = fopen(routelog, "w");
        if (!cache.route_log) { perror(routelog); return 1; }
    }

    DSV4ModelBind mb;
    if (dsv4_bind_model(&st, &c, 1, &mb) != 0) return 1;

    /* Two tables: see the note at the top. */
    /* ---- optional GPU residency for the dense trunk ------------------
     *
     * ONLY PINNED LAYERS, and that restriction is load-bearing rather than
     * conservative. Device matrices are keyed by their HOST weight pointer,
     * which is sound only while that pointer means one thing. A pinned layer
     * owns its allocation for the life of the run. A ring-slot layer does not:
     * the same address is reused for a different layer when the slot turns
     * over, so an upload keyed on it would go on serving the layer it was
     * first filled from -- silently, with plausible numbers. At --budget 16
     * all 43 layers pin and the question does not arise; at a smaller budget
     * the tail of the model simply stays on the CPU, which is correct and
     * slower rather than fast and wrong. */
    int gpu_mats = 0;
    if (use_gpu) {
        if (!dsv4_cuda_available()) {
            fprintf(stderr, "--gpu: no CUDA device (or built without CUDA); "
                            "running on the CPU\n");
        } else if (dsv4_cuda_init() != 0) {
            fprintf(stderr, "--gpu: device init failed; running on the CPU\n");
        } else {
            const size_t before = dsv4_cuda_free_vram();
            for (int L = 0; L < tr.npin; L++) {
                DSV4LayerBind lb;
                if (dsv4_trunk_bind(&tr, &c, L, &lb) != 0) break;
                const DSV4QMat *m[9] = {
                    &lb.w.attn.wq_a, &lb.w.attn.wq_b, &lb.w.attn.wkv,
                    &lb.w.attn.wo_a, &lb.w.attn.wo_b,
                    &lb.w.moe.shared.w1, &lb.w.moe.shared.w2,
                    &lb.w.moe.shared.w3,
                    lb.w.attn.has_idx ? &lb.w.attn.idx.wq_b : NULL,
                };
                for (int k = 0; k < 9; k++)
                    if (m[k] && m[k]->wdt == DSV4_WFP8
                        && dsv4_cuda_upload(m[k]) == 0) gpu_mats++;
            }
            const size_t after = dsv4_cuda_free_vram();
            printf("gpu: %d FP8 matrices resident over %d pinned layer(s), "
                   "%.2f GB of VRAM used, %.2f GB free\n",
                   gpu_mats, tr.npin,
                   (double)(before - after) / 1073741824.0,
                   (double)after / 1073741824.0);
            if (gpu_mats == 0)
                fprintf(stderr, "--gpu: nothing fitted; running on the CPU\n");
        }
    }

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
    if (dsv4_scratch_init(&scratch, &c, maxpos) != 0) return 1;

    DSV4LayerState *lst = calloc((size_t)c.n_layers, sizeof(DSV4LayerState));
    for (int L = 0; L < c.n_layers; L++)
        if (dsv4_state_init(&lst[L], &c, L, maxpos) != 0) return 1;

    DSV4ExpertSrc src = { dsv4_run_get_expert, dsv4_run_get_experts, &cache };

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

    /* PROMPT TOKENS GO THROUGH ONE AT A TIME, and that is not an oversight.
     *
     * The obvious improvement is layer-major prefill: bind layer L once, run
     * every prompt token through it, then move to L+1, so a layer's 165 MB of
     * weights is touched once per prompt instead of once per token. It was
     * built and measured on 2026-08-20. It is bit-exact -- all 26 generated ids
     * identical -- and it is NOT faster: 73.6 s against a 66.0-69.9 s baseline.
     *
     * It saves nothing because the trunk is fully PINNED, so there is no disk
     * read to avoid, and 165 MB per layer dwarfs the ~24 MB L3, so keeping a
     * layer resident across 15 tokens buys no cache reuse either. Each token
     * still performs the same GEMV against the same bytes.
     *
     * What would win is a batched GEMM -- one weight row loaded once and used
     * for all N tokens' dot products, taking arithmetic intensity from ~1
     * flop/byte to ~N -- which needs a batched variant of every matmul and of
     * attention. That is a different change, and layer-major is only its
     * scaffolding. See task notes. */
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
    dsv4_prof_report(dt, ntok);
    printf("PEAK RSS for the whole run: %.2f GB\n", peak_rss_gb());

    for (int L = 0; L < c.n_layers; L++) dsv4_state_free(&lst[L]);
    free(lst); free(h); free(xh); free(logits);
    free(cs_d); free(sn_d); free(cs_c); free(sn_c);
    dsv4_scratch_free(&scratch);
    dsv4_bind_model_free(&mb);
    if (gpu_mats) dsv4_cuda_shutdown();
    dsv4_cache_free(&cache);
    dsv4_trunk_close(&tr);
    dsv4_tok_free(&tok);
    dsv4_st_close(&st);
    return 0;
}
