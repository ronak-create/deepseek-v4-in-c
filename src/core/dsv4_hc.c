/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_hc.c - Manifold-Constrained Hyper-Connections (mHC).
 *
 * DeepSeek-V4 does not have one residual stream. It has hc_mult of them, and
 * each block reduces them to one, does its work, and expands back out through a
 * doubly-stochastic mixing matrix produced by Sinkhorn normalisation.
 *
 * Sources: kernel.py hc_split_sinkhorn_kernel, model.py Block.hc_pre/hc_post.
 *
 * FOUR PLACES THIS GOES WRONG QUIETLY
 *
 * 1. THE SINKHORN LOOP IS NOT N IDENTICAL PASSES. The first pass is
 *      softmax over rows, +eps, then a COLUMN normalise.
 *    The remaining sinkhorn_iters - 1 passes are
 *      row normalise, then column normalise.
 *    Writing 20 uniform row/col passes drops the softmax and changes every
 *    mixing weight, while still producing a near-doubly-stochastic matrix.
 *
 * 2. comb IS USED TRANSPOSED. hc_post computes, in torch,
 *      sum over dim 2 of  comb[..., j, k, None] * residual[..., j, None, :]
 *    which contracts the FIRST index of comb against the residual copy index:
 *      y[k] = post[k] * x + SUM_j comb[j][k] * residual[j]
 *    The natural reading, SUM_k comb[j][k] * residual[k], has the same shape
 *    and the same cost and is a different model.
 *
 * 3. pre GETS AN EPS, post DOES NOT, and post carries a factor of 2:
 *      pre[j]  = sigmoid(...) + eps
 *      post[j] = 2 * sigmoid(...)
 *    Making them symmetric is the obvious tidy-up and is wrong.
 *
 * 4. THE RSQRT IN hc_pre IS OVER THE WHOLE FLATTENED hc_mult*hidden VECTOR,
 *    not per copy, and it uses norm_eps (rms_norm_eps), not hc_eps. Two
 *    different epsilons live in this file and swapping them is invisible.
 */
#include <math.h>
#include <string.h>

#include "dsv4.h"

static inline float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

/* Split `mixes` into the three mHC quantities.
 *
 * mixes is mix_hc = (2 + hc) * hc long, laid out as
 *   [0 .. hc)          -> pre
 *   [hc .. 2*hc)       -> post
 *   [2*hc .. mix_hc)   -> comb, row-major hc x hc
 *
 * comb must hold hc*hc floats; pre and post hc each. */
void dsv4_hc_split_sinkhorn(float *pre, float *post, float *comb,
                            const float *mixes, const float *hc_scale,
                            const float *hc_base, int hc, int iters, float eps)
{
    for (int j = 0; j < hc; j++)
        pre[j] = sigmoidf_(mixes[j] * hc_scale[0] + hc_base[j]) + eps;

    for (int j = 0; j < hc; j++)
        post[j] = 2.0f * sigmoidf_(mixes[j + hc] * hc_scale[1] + hc_base[j + hc]);

    for (int j = 0; j < hc; j++)
        for (int k = 0; k < hc; k++) {
            const int o = j * hc + k + hc * 2;
            comb[j * hc + k] = mixes[o] * hc_scale[2] + hc_base[o];
        }

    /* ---- first pass: row softmax, +eps, then a COLUMN normalise ---- */
    for (int j = 0; j < hc; j++) {
        float mx = comb[j * hc];
        for (int k = 1; k < hc; k++) if (comb[j * hc + k] > mx) mx = comb[j * hc + k];
        double sum = 0.0;
        for (int k = 0; k < hc; k++) {
            const float e = expf(comb[j * hc + k] - mx);
            comb[j * hc + k] = e;
            sum += (double)e;
        }
        const float inv = (float)(1.0 / sum);
        for (int k = 0; k < hc; k++) comb[j * hc + k] = comb[j * hc + k] * inv + eps;
    }
    for (int k = 0; k < hc; k++) {
        double cs = 0.0;
        for (int j = 0; j < hc; j++) cs += (double)comb[j * hc + k];
        const float d = (float)cs + eps;
        for (int j = 0; j < hc; j++) comb[j * hc + k] /= d;
    }

    /* ---- the remaining iters-1 passes: row normalise, then column ----
     * Note iters-1, not iters: the softmax above counts as the first. */
    for (int it = 0; it < iters - 1; it++) {
        for (int j = 0; j < hc; j++) {
            double rs = 0.0;
            for (int k = 0; k < hc; k++) rs += (double)comb[j * hc + k];
            const float d = (float)rs + eps;
            for (int k = 0; k < hc; k++) comb[j * hc + k] /= d;
        }
        for (int k = 0; k < hc; k++) {
            double cs = 0.0;
            for (int j = 0; j < hc; j++) cs += (double)comb[j * hc + k];
            const float d = (float)cs + eps;
            for (int j = 0; j < hc; j++) comb[j * hc + k] /= d;
        }
    }
}

/* Reduce hc copies to one.
 *
 * x is [hc][hidden]. mixes = (hc_fn @ flat(x)) * rsqrt(mean(flat(x)^2)+norm_eps),
 * where the mean is over ALL hc*hidden elements. Then y = sum_j pre[j] * x[j].
 *
 * hc_fn is [mix_hc][hc*hidden] and is stored f32 in the checkpoint.
 * `mixes` and `comb` are caller-owned scratch. */
void dsv4_hc_pre(float *y, float *post, float *comb, float *mixes,
                 const float *x, const DSV4HcW *w, int hc, int hidden,
                 int iters, float norm_eps, float hc_eps)
{
    const int hcdim  = hc * hidden;
    const int mix_hc = (2 + hc) * hc;

    /* RMS over the WHOLE flattened vector, with norm_eps -- not hc_eps, and not
     * per copy. */
    double sq = 0.0;
    for (int i = 0; i < hcdim; i++) sq += (double)x[i] * (double)x[i];
    const float rsqrt = (float)(1.0 / sqrt(sq / (double)hcdim + (double)norm_eps));

    for (int m = 0; m < mix_hc; m++) {
        const float *row = w->fn + (size_t)m * hcdim;
        double acc = 0.0;
        for (int i = 0; i < hcdim; i++) acc = fma((double)row[i], (double)x[i], acc);
        mixes[m] = (float)acc * rsqrt;
    }

    float pre[DSV4_MAX_HC_MULT];
    dsv4_hc_split_sinkhorn(pre, post, comb, mixes, w->scale, w->base,
                           hc, iters, hc_eps);

    for (int d = 0; d < hidden; d++) {
        double acc = 0.0;
        for (int j = 0; j < hc; j++)
            acc += (double)pre[j] * (double)x[(size_t)j * hidden + d];
        y[d] = (float)acc;
    }
}

/* Expand one stream back to hc.
 *
 *   y[k] = post[k] * x + SUM_j comb[j][k] * residual[j]
 *
 * The contraction is over comb's FIRST index. See note 2 at the top: the
 * transposed form has identical shape and cost. */
void dsv4_hc_post(float *y, const float *x, const float *residual,
                  const float *post, const float *comb, int hc, int hidden)
{
    for (int k = 0; k < hc; k++) {
        float *out = y + (size_t)k * hidden;
        for (int d = 0; d < hidden; d++) {
            double acc = (double)post[k] * (double)x[d];
            for (int j = 0; j < hc; j++)
                acc += (double)comb[j * hc + k]
                     * (double)residual[(size_t)j * hidden + d];
            out[d] = (float)acc;
        }
    }
}
