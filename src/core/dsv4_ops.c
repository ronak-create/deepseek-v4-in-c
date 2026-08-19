/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_ops.c - the elementwise kernels, and the router.
 *
 * Each of these is short, and each has a way of being subtly wrong that leaves
 * the model running and fluent. The comments record which way.
 */
#include <math.h>
#include <string.h>

#include "dsv4.h"

/* ---------------------------------------------------------------- RMSNorm ---
 * model.py RMSNorm.forward:
 *     var = x.square().mean(-1)
 *     x   = x * rsqrt(var + eps)
 *     out = weight * x
 *
 * The eps is added to the MEAN of squares, not to the sum and not inside the
 * square root of the sum. With eps = 1e-6 and hidden = 4096 the difference is
 * small enough to look like noise and large enough to move logits.
 *
 * Accumulated in double: the sum of 4096 squares in f32 loses low bits in a way
 * that varies with the compiler's vectorisation, which would break the
 * bit-identity contract the matmuls maintain. */
void dsv4_rmsnorm(float *y, const float *x, const float *w, int n, float eps)
{
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += (double)x[i] * (double)x[i];
    const double mean = sum / (double)n;
    const float inv = (float)(1.0 / sqrt(mean + (double)eps));
    for (int i = 0; i < n; i++) y[i] = w[i] * (x[i] * inv);
}

/* --------------------------------------------------------------- softplus ---
 * torch.nn.functional.softplus has a THRESHOLD, default 20: above it the
 * function returns x unchanged rather than evaluating log1p(exp(x)).
 *
 * That is not only an optimisation. exp(x) overflows f32 near x = 88, so a
 * naive log1p(exp(x)) returns +inf for large scores, and sqrt(+inf) is +inf,
 * and a routing weight of +inf normalises to NaN. Matching the threshold keeps
 * this finite AND keeps it bit-identical to the reference. */
static inline float dsv4_softplus(float x)
{
    return (x > 20.0f) ? x : log1pf(expf(x));
}

/* ------------------------------------------------------------ SwiGLU/clamp ---
 * model.py Expert.forward, and note the ASYMMETRY:
 *
 *     up   = clamp(up,   min=-limit, max=+limit)     <- both sides
 *     gate = clamp(gate,            max=+limit)     <- UPPER ONLY
 *     x    = silu(gate) * up
 *
 * Clamping gate from below as well is the obvious-looking implementation and is
 * wrong. It would only bite on strongly negative gate values, where silu is
 * already near zero, so the output stays plausible and the error is invisible
 * without a reference.
 *
 * limit <= 0 disables clamping entirely, matching `if self.swiglu_limit > 0`. */
void dsv4_swiglu(float *y, const float *gate, const float *up, int n, float limit)
{
    if (limit > 0.0f) {
        for (int i = 0; i < n; i++) {
            float u = up[i];
            if (u >  limit) u =  limit;
            if (u < -limit) u = -limit;
            float g = gate[i];
            if (g >  limit) g =  limit;       /* deliberately no lower bound */
            const float s = g / (1.0f + expf(-g));            /* silu */
            y[i] = s * u;
        }
    } else {
        for (int i = 0; i < n; i++) {
            const float g = gate[i];
            y[i] = (g / (1.0f + expf(-g))) * up[i];
        }
    }
}

/* ----------------------------------------------------------------- router ---
 * model.py Gate.forward. The order of operations is the whole point:
 *
 *     scores          = sqrt(softplus(x @ W^T))
 *     original_scores = scores                 <- SAVED BEFORE THE BIAS
 *     scores          = scores + bias          <- scored layers only
 *     indices         = tid2eid[token]  or  topk(scores)
 *     weights         = original_scores[indices]
 *     weights        /= weights.sum()
 *     weights        *= route_scale
 *
 * THE BIAS STEERS SELECTION ONLY. Gathering the weights from the BIASED scores
 * is the natural mistake -- they are the array in hand at that point -- and it
 * produces a model that routes correctly and combines wrongly. Every expert
 * still fires, every number stays finite, and the output is merely worse.
 *
 * `scores` is modified in place (it becomes the biased copy); `orig` must hold
 * a separate n_experts floats. idx and wts receive topk entries.
 *
 * tid2eid is int64 in the checkpoint even though the values are small; it is
 * read as int64_t here rather than cast at bind time so the checkpoint's own
 * bytes stay untouched. */
void dsv4_route(int *idx, float *wts, float *scores, float *orig,
                const float *bias, const int64_t *tid2eid, int token_id,
                int n_experts, int topk, float route_scale)
{
    for (int e = 0; e < n_experts; e++)
        orig[e] = sqrtf(dsv4_softplus(scores[e]));

    if (bias) for (int e = 0; e < n_experts; e++) scores[e] = orig[e] + bias[e];
    else      for (int e = 0; e < n_experts; e++) scores[e] = orig[e];

    if (tid2eid) {
        /* Hash routing: the experts are a property of the TOKEN, known before
         * any of this was computed. The scores are still needed for the
         * weights, which is why they are not skipped. */
        const int64_t *row = tid2eid + (int64_t)token_id * topk;
        for (int k = 0; k < topk; k++) idx[k] = (int)row[k];
    } else {
        /* Selection top-k over the BIASED scores. Descending by value, ties to
         * the lower index, matching torch.topk so a tie cannot silently pick a
         * different expert. */
        for (int k = 0; k < topk; k++) {
            int best = -1;
            for (int e = 0; e < n_experts; e++) {
                int taken = 0;
                for (int j = 0; j < k; j++) if (idx[j] == e) { taken = 1; break; }
                if (taken) continue;
                if (best < 0 || scores[e] > scores[best]) best = e;
            }
            idx[k] = best;
        }
    }

    /* Weights from the UNBIASED scores. */
    double sum = 0.0;
    for (int k = 0; k < topk; k++) { wts[k] = orig[idx[k]]; sum += (double)wts[k]; }
    /* norm_topk_prob is true for both released models, and score_func is not
     * softmax, so the normalisation always runs. */
    const float inv = (sum != 0.0) ? (float)(1.0 / sum) : 0.0f;
    for (int k = 0; k < topk; k++) wts[k] = wts[k] * inv * route_scale;
}
