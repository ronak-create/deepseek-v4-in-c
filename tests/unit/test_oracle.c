/* SPDX-License-Identifier: Apache-2.0 */
/* test_oracle.c - every kernel against the PyTorch reference.
 *
 * This is the gate the whole port has been working toward. Until now every
 * check was internal consistency or a structural counterfactual: they prove the
 * C does what its author intended, not that the intent matched DeepSeek. Roughly
 * sixteen traps were found by reading model.py and kernel.py, and every one of
 * those corrections was REASONED. This file measures them.
 *
 * The fixtures come from tools/dsv4_ref.py, a pure-PyTorch reimplementation
 * written from the reference sources and deliberately NOT from src/. Two
 * implementations that agree because one was transcribed from the other prove
 * nothing.
 *
 * TOLERANCES
 *   The reference computes in f32 with a different summation order, so bit
 *   equality is not available and would be the wrong thing to demand. Each
 *   check states a relative tolerance sized to the kernel: elementwise maps get
 *   a tight one, reductions over many terms a looser one. A tolerance loose
 *   enough to hide a wrong constant would defeat the file, so they are kept as
 *   small as the arithmetic allows.
 */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdint.h>
#include "dsv4.h"
#include "json.h"

void  dsv4_rmsnorm(float *, const float *, const float *, int, float);
void  dsv4_swiglu(float *, const float *, const float *, int, float);
void  dsv4_route(int *, float *, float *, float *, const float *,
                 const int64_t *, int, int, int, float);
void  dsv4_rope_table(float *, float *, int, int, int, double, double, double, double);
void  dsv4_rope_apply(float *, int, int, const float *, const float *, int, int, int);
void  dsv4_hc_split_sinkhorn(float *, float *, float *, const float *,
                             const float *, const float *, int, int, float);
void  dsv4_hc_post(float *, const float *, const float *, const float *,
                   const float *, int, int);
void  dsv4_sparse_attn(float *, const float *, const float *, const float *,
                       const int *, int, int, int, float, float *);
void  dsv4_indexer_score(float *, const float *, const float *, const float *,
                         int, int, int);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

/* ---- fixture loading ---------------------------------------------------- */

static char *slurp(const char *path)
{
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    char *s = (char *)malloc((size_t)n + 1);
    if (fread(s, 1, (size_t)n, f) != (size_t)n) { fclose(f); free(s); return NULL; }
    s[n] = 0; fclose(f);
    return s;
}

static jval *load(const char *name, char **txt)
{
    char p[256];
    snprintf(p, sizeof p, "tests/fixtures/ref/%s.json", name);
    *txt = slurp(p);
    if (!*txt) { printf("  FAIL  missing %s (run make ref-fixtures)\n", p); fails++; return NULL; }
    char *arena = NULL;
    return json_parse(*txt, &arena);
}

/* NaN in the fixture marks a masked slot: JSON has no -inf, so the emitter
 * writes null and this turns it back into -INFINITY. */
static int arr(jval *o, const char *k, float *out, int max)
{
    jval *a = json_get(o, k);
    if (!a || a->t != J_ARR) return -1;
    const int n = a->len < max ? a->len : max;
    for (int i = 0; i < n; i++)
        out[i] = (a->kids[i]->t == J_NUM) ? (float)a->kids[i]->num : -INFINITY;
    return n;
}

static int iarr(jval *o, const char *k, int *out, int max)
{
    jval *a = json_get(o, k);
    if (!a || a->t != J_ARR) return -1;
    const int n = a->len < max ? a->len : max;
    for (int i = 0; i < n; i++) out[i] = (int)a->kids[i]->num;
    return n;
}

static double num(jval *o, const char *k, double dflt)
{
    jval *v = json_get(o, k);
    return (v && v->t == J_NUM) ? v->num : dflt;
}

/* Comparison against the reference, reporting the worst element.
 *
 * THE ERROR IS SCALED BY THE VECTOR, NOT BY THE ELEMENT, and that choice is
 * load-bearing. Judging each element against its own magnitude lets an element
 * that is near zero through cancellation dominate the metric, even when its
 * absolute error is negligible and the C is the MORE accurate of the two.
 *
 * That is not hypothetical. rope_dense element 25 comes out at 9.12e-06 against
 * a vector RMS of 0.722 -- 79,000x smaller than typical -- because
 * re*cos - im*sin cancels there. The reference computes its cos/sin in f32;
 * this engine accumulates them in double. Measured against an f64 recomputation,
 * torch's own answer is 1.65e-3 relative from the truth and the C is ~250x
 * closer. A per-element relative metric fails the C for being right.
 *
 * Scaling by max(|ref_i|, rms(ref)) keeps full sensitivity on elements that
 * carry signal and stops a cancelled one from dominating. It is NOT a loosened
 * tolerance: a wrong constant or a wrong reduction still moves the elements that
 * matter, and those are still judged against themselves.
 */
static void agree(const char *what, const float *got, const float *ref,
                  int n, float tol)
{
    double sq = 0.0;
    for (int i = 0; i < n; i++) sq += (double)ref[i] * ref[i];
    const float rms = (float)sqrt(sq / (double)n);

    float worst = 0.0f; int at = -1;
    for (int i = 0; i < n; i++) {
        const float d = fabsf(got[i] - ref[i]);
        const float scale = fmaxf(1e-30f,
                            fmaxf(rms, fmaxf(fabsf(got[i]), fabsf(ref[i]))));
        const float rel = d / scale;
        if (rel > worst) { worst = rel; at = i; }
    }
    if (worst > tol)
        CHECK(0, "%s: worst relative error %.3g at element %d "
                 "(C %.9g vs torch %.9g), tolerance %.1g",
              what, (double)worst, at, (double)got[at], (double)ref[at], (double)tol);
    else
        printf("  ok    %-22s worst rel err %.2g over %d values\n", what,
               (double)worst, n);
}

#define BUF 4096
static float g_a[BUF], g_b[BUF], g_c[BUF], g_r[BUF], g_o[BUF];

int main(void)
{
    char *txt; jval *f;
    printf("DeepSeek-V4 ORACLE gate -- C against the PyTorch reference\n");

    printf("\n-- RMSNorm --\n");
    for (int k = 0; k < 2; k++) {
        const char *nm = k ? "rmsnorm_tiny" : "rmsnorm_normal";
        if (!(f = load(nm, &txt))) continue;
        const int n = (int)num(f, "n", 0);
        arr(f, "x", g_a, BUF); arr(f, "w", g_b, BUF); arr(f, "y", g_r, BUF);
        dsv4_rmsnorm(g_o, g_a, g_b, n, (float)num(f, "eps", 1e-6));
        agree(nm, g_o, g_r, n, 2e-6f);
        free(txt);
    }

    printf("\n-- SwiGLU (the asymmetric clamp) --\n");
    for (int k = 0; k < 2; k++) {
        const char *nm = k ? "swiglu_unclamped" : "swiglu_clamped";
        if (!(f = load(nm, &txt))) continue;
        const int n = (int)num(f, "n", 0);
        arr(f, "gate", g_a, BUF); arr(f, "up", g_b, BUF); arr(f, "y", g_r, BUF);
        dsv4_swiglu(g_o, g_a, g_b, n, (float)num(f, "limit", 0));
        agree(nm, g_o, g_r, n, 5e-6f);
        free(txt);
    }

    printf("\n-- router (bias steers selection only) --\n");
    for (int k = 0; k < 2; k++) {
        const char *nm = k ? "router_large_logits" : "router_scored";
        if (!(f = load(nm, &txt))) continue;
        const int ne = (int)num(f, "n_experts", 0), tk = (int)num(f, "topk", 0);
        const int have_bias = json_get(f, "bias") &&
                              json_get(f, "bias")->t == J_ARR;
        arr(f, "logits", g_a, BUF);
        if (have_bias) arr(f, "bias", g_b, BUF);
        arr(f, "weights", g_r, BUF);
        int ridx[DSV4_MAX_TOPK]; iarr(f, "indices", ridx, DSV4_MAX_TOPK);

        int idx[DSV4_MAX_TOPK]; float wts[DSV4_MAX_TOPK];
        static float orig[BUF];
        memcpy(g_c, g_a, (size_t)ne * sizeof(float));
        dsv4_route(idx, wts, g_c, orig, have_bias ? g_b : NULL, NULL, 0,
                   ne, tk, (float)num(f, "route_scale", 1.0));
        /* selection first: a different expert set makes the weights meaningless */
        int same = 1;
        for (int i = 0; i < tk; i++) if (idx[i] != ridx[i]) same = 0;
        CHECK(same, "%s selected a different expert set than torch", nm);
        if (same) agree(nm, wts, g_r, tk, 1e-5f);
        free(txt);
    }

    printf("\n-- RoPE (interleaved pairs, YaRN on and off) --\n");
    for (int k = 0; k < 2; k++) {
        const char *nm = k ? "rope_yarn" : "rope_dense";
        if (!(f = load(nm, &txt))) continue;
        const int rd = (int)num(f, "rd", 0), pos = (int)num(f, "pos", 0);
        arr(f, "x", g_a, BUF); arr(f, "y", g_r, BUF);
        static float cs[128 * 64], sn[128 * 64];
        if (k) dsv4_rope_table(cs, sn, rd, 128, 65536, 160000.0, 16.0, 32.0, 1.0);
        else   dsv4_rope_table(cs, sn, rd, 128, 0,     10000.0, 16.0, 32.0, 1.0);
        memcpy(g_o, g_a, (size_t)rd * sizeof(float));
        dsv4_rope_apply(g_o, rd, rd, cs, sn, pos, rd / 2, 0);
        agree(nm, g_o, g_r, rd, 5e-6f);
        free(txt);
    }

    printf("\n-- mHC Sinkhorn --\n");
    if ((f = load("sinkhorn", &txt))) {
        const int hc = (int)num(f, "hc", 4);
        arr(f, "mixes", g_a, BUF); arr(f, "hc_scale", g_b, BUF);
        arr(f, "hc_base", g_c, BUF);
        float pre[8], post[8], comb[64];
        dsv4_hc_split_sinkhorn(pre, post, comb, g_a, g_b, g_c, hc,
                               (int)num(f, "iters", 20), (float)num(f, "eps", 1e-6));
        arr(f, "pre", g_r, BUF);  agree("sinkhorn pre",  pre,  g_r, hc, 1e-5f);
        arr(f, "post", g_r, BUF); agree("sinkhorn post", post, g_r, hc, 1e-5f);
        arr(f, "comb", g_r, BUF); agree("sinkhorn comb", comb, g_r, hc * hc, 5e-5f);
        free(txt);
    }

    printf("\n-- mHC expand (comb contracted on its FIRST index) --\n");
    if ((f = load("hc_post", &txt))) {
        const int hc = (int)num(f, "hc", 4), d = (int)num(f, "d", 8);
        arr(f, "x", g_a, BUF); arr(f, "residual", g_b, BUF);
        arr(f, "post", g_c, BUF);
        static float comb[64]; arr(f, "comb", comb, 64);
        arr(f, "y", g_r, BUF);
        dsv4_hc_post(g_o, g_a, g_b, g_c, comb, hc, d);
        agree("hc_post", g_o, g_r, hc * d, 1e-5f);
        free(txt);
    }

    printf("\n-- sparse attention (sink in the denominator only) --\n");
    if ((f = load("sparse_attn", &txt))) {
        const int h = (int)num(f, "h", 0), d = (int)num(f, "d", 0);
        const int tk = (int)num(f, "topk", 0);
        arr(f, "q", g_a, BUF); arr(f, "kv", g_b, BUF); arr(f, "sink", g_c, BUF);
        static int idxs[64]; iarr(f, "idxs", idxs, 64);
        arr(f, "o", g_r, BUF);
        static float scratch[256];
        dsv4_sparse_attn(g_o, g_a, g_b, g_c, idxs, h, d, tk,
                         (float)num(f, "scale", 1.0), scratch);
        agree("sparse_attn", g_o, g_r, h * d, 2e-5f);
        free(txt);
    }

    printf("\n-- CSA indexer (ReLU before the head weighting) --\n");
    if ((f = load("indexer", &txt))) {
        const int nh = (int)num(f, "n_heads", 0), hd = (int)num(f, "head_dim", 0);
        const int np = (int)num(f, "n_pos", 0);
        arr(f, "q", g_a, BUF); arr(f, "kv", g_b, BUF); arr(f, "weights", g_c, BUF);
        arr(f, "score", g_r, BUF);
        dsv4_indexer_score(g_o, g_a, g_b, g_c, nh, hd, np);
        agree("indexer_score", g_o, g_r, np, 2e-5f);
        free(txt);
    }

    printf("\n");
    if (fails) { printf("ORACLE GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("ORACLE GATE PASSED\n");
    return 0;
}
