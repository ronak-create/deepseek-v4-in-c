/* SPDX-License-Identifier: Apache-2.0 */
/* test_hc.c - mHC: Sinkhorn splitting, the reduce, and the expand.
 *
 * The four traps documented at the top of dsv4_hc.c each get a gate that a
 * wrong-but-plausible implementation fails. "Output is finite and roughly the
 * right magnitude" passes on all four, so none of these gates settle for that.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "dsv4.h"

void dsv4_hc_split_sinkhorn(float *, float *, float *, const float *,
                            const float *, const float *, int, int, float);
void dsv4_hc_pre(float *, float *, float *, float *, const float *,
                 const DSV4HcW *, int, int, int, float, float);
void dsv4_hc_post(float *, const float *, const float *, const float *,
                  const float *, int, int);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

#define HC 4
#define MIX ((2 + HC) * HC)      /* 24 */

int main(void)
{
    printf("DeepSeek-V4 mHC gate\n");

    printf("\n-- GATE 1  Sinkhorn produces a near doubly-stochastic comb --\n");
    {
        float mixes[MIX], base[MIX], scale[3] = { 1.0f, 1.0f, 1.0f };
        float pre[HC], post[HC], comb[HC * HC];
        unsigned st = 7u;
        for (int i = 0; i < MIX; i++) {
            st = st * 1103515245u + 12345u;
            mixes[i] = (float)((int)((st >> 16) & 0xFF) - 128) * 0.02f;
            base[i]  = 0.0f;
        }
        dsv4_hc_split_sinkhorn(pre, post, comb, mixes, scale, base, HC, 20, 1e-6f);

        for (int j = 0; j < HC; j++) {
            double rs = 0.0, cs = 0.0;
            for (int k = 0; k < HC; k++) { rs += comb[j * HC + k]; cs += comb[k * HC + j]; }
            CHECK(fabs(rs - 1.0) < 1e-3, "row %d sums to %.6f, expected ~1", j, rs);
            CHECK(fabs(cs - 1.0) < 1e-3, "col %d sums to %.6f, expected ~1", j, cs);
        }
        for (int i = 0; i < HC * HC; i++)
            CHECK(comb[i] > 0.0f && isfinite(comb[i]), "comb[%d] = %g", i, (double)comb[i]);
        printf("  ok    rows and columns both sum to 1 within 1e-3\n");
    }

    printf("\n-- GATE 2  pre carries +eps, post carries a factor of 2 --\n");
    {
        /* With mixes = 0, base = 0 the sigmoid is exactly 0.5, so the two
         * asymmetries are directly readable:
         *     pre  = 0.5 + eps
         *     post = 2 * 0.5 = 1.0
         * A symmetric implementation gives 0.5 and 0.5, or 0.5+eps twice. */
        float mixes[MIX] = {0}, base[MIX] = {0}, scale[3] = { 1.0f, 1.0f, 1.0f };
        float pre[HC], post[HC], comb[HC * HC];
        const float eps = 1e-3f;             /* large, so it is unmistakable */
        dsv4_hc_split_sinkhorn(pre, post, comb, mixes, scale, base, HC, 20, eps);

        for (int j = 0; j < HC; j++) {
            CHECK(fabsf(pre[j]  - (0.5f + eps)) < 1e-6f,
                  "pre[%d] = %.6f, expected 0.5 + eps", j, (double)pre[j]);
            CHECK(fabsf(post[j] - 1.0f) < 1e-6f,
                  "post[%d] = %.6f, expected 2*sigmoid(0) = 1.0", j, (double)post[j]);
            CHECK(post[j] != pre[j], "pre and post are identical; the asymmetry is lost");
        }
        printf("  ok    pre = %.4f (0.5+eps), post = %.4f (2*sigmoid)\n",
               (double)pre[0], (double)post[0]);
    }

    printf("\n-- GATE 3  the first pass is a SOFTMAX, not a row normalise --\n");
    {
        /* Feed comb logits that are spread out. A row softmax exponentiates
         * them, so the largest entry dominates; a plain row normalise (dividing
         * by the sum of the raw values) does not. Both end doubly stochastic
         * after Sinkhorn, so only the RATIO between entries separates them. */
        float mixes[MIX] = {0}, base[MIX] = {0}, scale[3] = { 1.0f, 1.0f, 1.0f };
        float pre[HC], post[HC], comb[HC * HC];
        /* row 0 of comb: logits 3, 0, 0, 0 */
        mixes[2 * HC + 0] = 3.0f;
        dsv4_hc_split_sinkhorn(pre, post, comb, mixes, scale, base, HC, 20, 1e-6f);

        const float big = comb[0], other = comb[1];
        CHECK(big > other * 2.0f,
              "comb[0][0]=%.5f is not dominant over comb[0][1]=%.5f; "
              "the softmax may have been replaced by a plain normalise",
              (double)big, (double)other);
        printf("  ok    comb[0][0] = %.4f dominates comb[0][1] = %.4f\n",
               (double)big, (double)other);
    }

    printf("\n-- GATE 4  hc_post contracts comb's FIRST index --\n");
    {
        /* Asymmetric comb, so the transposed contraction gives a different
         * answer. residual copy j is the constant j+1; x is zero and post is
         * zero, isolating the comb term.
         *
         *   correct:    y[k] = SUM_j comb[j][k] * residual[j]
         *   transposed: y[k] = SUM_k' comb[k][k'] * residual[k']
         */
        const int hidden = 2;
        float residual[HC * 2], x[2] = { 0.0f, 0.0f }, y[HC * 2];
        float post[HC] = { 0.0f, 0.0f, 0.0f, 0.0f };
        float comb[HC * HC];
        memset(comb, 0, sizeof comb);
        /* comb[0][1] = 1, everything else 0. */
        comb[0 * HC + 1] = 1.0f;
        for (int j = 0; j < HC; j++) { residual[j * 2] = (float)(j + 1);
                                       residual[j * 2 + 1] = (float)(j + 1); }

        dsv4_hc_post(y, x, residual, post, comb, HC, hidden);

        /* comb[0][1] = 1 contracts j=0, so it lands in OUTPUT copy k=1 and
         * carries residual copy 0, whose value is 1. */
        CHECK(y[1 * 2] == 1.0f, "y[k=1] = %.4f, expected 1 (residual copy 0)",
              (double)y[1 * 2]);
        CHECK(y[0 * 2] == 0.0f, "y[k=0] = %.4f, expected 0", (double)y[0 * 2]);
        /* The transposed reading would have put residual copy 1 (value 2) into
         * output copy 0. Assert we are not in that world. */
        CHECK(y[0 * 2] != 2.0f, "y[k=0] = 2: comb is being used TRANSPOSED");
        printf("  ok    comb[0][1] moved residual copy 0 into output copy 1\n");
    }

    printf("\n-- GATE 5  hc_pre reduces hc copies with the pre weights --\n");
    {
        /* hc_fn = 0 makes mixes zero, so pre[j] = sigmoid(0)+eps = 0.5+eps for
         * every j, and the reduce is 0.5 * (sum of copies). */
        const int hidden = 8;
        const int hcdim = HC * hidden;
        static float fn[MIX * (HC * 8)];
        float base[MIX] = {0}, scale[3] = { 1.0f, 1.0f, 1.0f };
        float x[HC * 8], y[8], post[HC], comb[HC * HC], mixes[MIX];
        memset(fn, 0, sizeof fn);
        for (int j = 0; j < HC; j++)
            for (int d = 0; d < hidden; d++) x[j * hidden + d] = (float)(j + 1);

        DSV4HcW w; w.fn = fn; w.base = base; w.scale = scale;
        dsv4_hc_pre(y, post, comb, mixes, x, &w, HC, hidden, 20, 1e-6f, 0.0f);

        /* pre = 0.5 exactly (eps 0), copies are 1,2,3,4 -> 0.5*10 = 5. */
        for (int d = 0; d < hidden; d++)
            CHECK(fabsf(y[d] - 5.0f) < 1e-5f, "y[%d] = %.6f, expected 5", d,
                  (double)y[d]);
        (void)hcdim;
        printf("  ok    reduce gives 0.5*(1+2+3+4) = %.4f\n", (double)y[0]);
    }

    printf("\n-- GATE 6  hc_pre's rsqrt spans the WHOLE flattened vector --\n");
    {
        /* If the rsqrt were computed per copy instead of over hc*hidden, the
         * scale would differ whenever the copies have different magnitudes.
         * Use hc_fn = identity-ish so mixes is a plain read of x, then check
         * the resulting mixes against a hand-computed whole-vector rsqrt. */
        const int hidden = 4;
        const int hcdim = HC * hidden;              /* 16 */
        static float fn[MIX * 16];
        float base[MIX] = {0}, scale[3] = { 1.0f, 1.0f, 1.0f };
        float x[16], y[4], post[HC], comb[HC * HC], mixes[MIX];
        memset(fn, 0, sizeof fn);
        fn[0] = 1.0f;                               /* mixes[0] = x[0] * rsqrt */
        /* copies of very different magnitude: 1,1,1,1, 10,10,10,10, ... */
        for (int j = 0; j < HC; j++)
            for (int d = 0; d < hidden; d++)
                x[j * hidden + d] = (float)((j + 1) * (j + 1));

        DSV4HcW w; w.fn = fn; w.base = base; w.scale = scale;
        dsv4_hc_pre(y, post, comb, mixes, x, &w, HC, hidden, 20, 0.0f, 0.0f);

        double sq = 0.0;
        for (int i = 0; i < hcdim; i++) sq += (double)x[i] * x[i];
        const float whole = (float)(1.0 / sqrt(sq / hcdim));
        const float percopy = (float)(1.0 / sqrt((double)x[0] * x[0]));
        CHECK(fabsf(mixes[0] - x[0] * whole) < 1e-4f,
              "mixes[0] = %.6f, whole-vector rsqrt gives %.6f", (double)mixes[0],
              (double)(x[0] * whole));
        CHECK(fabsf(whole - percopy) > 0.1f,
              "the two rsqrt scopes are indistinguishable here; test is too weak");
        printf("  ok    whole-vector rsqrt %.4f used, not per-copy %.4f\n",
               (double)whole, (double)percopy);
    }

    printf("\n-- GATE 7  iteration count matters --\n");
    {
        /* One iteration is softmax + column normalise only, so rows are NOT
         * normalised and cannot sum to 1. If 1 and 20 iterations agreed, the
         * loop would not be running. */
        float mixes[MIX], base[MIX] = {0}, scale[3] = { 1.0f, 1.0f, 1.0f };
        float pre[HC], post[HC], c1[HC * HC], c20[HC * HC];
        unsigned st = 99u;
        for (int i = 0; i < MIX; i++) {
            st = st * 1103515245u + 12345u;
            mixes[i] = (float)((int)((st >> 16) & 0xFF) - 128) * 0.03f;
        }
        dsv4_hc_split_sinkhorn(pre, post, c1,  mixes, scale, base, HC, 1,  1e-6f);
        dsv4_hc_split_sinkhorn(pre, post, c20, mixes, scale, base, HC, 20, 1e-6f);
        CHECK(memcmp(c1, c20, sizeof c1) != 0,
              "1 and 20 iterations give identical comb; the loop is not running");
        double rs1 = 0.0, rs20 = 0.0;
        for (int k = 0; k < HC; k++) { rs1 += c1[k]; rs20 += c20[k]; }
        CHECK(fabs(rs20 - 1.0) < 1e-3, "20-iter row sum %.6f, expected ~1", rs20);
        printf("  ok    1 iter row sum %.4f, 20 iters %.4f\n", rs1, rs20);
    }

    printf("\n");
    if (fails) { printf("mHC GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("mHC GATE PASSED\n");
    return 0;
}
