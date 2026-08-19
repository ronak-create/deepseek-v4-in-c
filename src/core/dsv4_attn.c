/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_attn.c - sparse attention over gathered KV positions, with sinks.
 *
 * Source: kernel.py sparse_attn_kernel. That kernel is FlashAttention-shaped
 * (blocked, online softmax with a running max) because it runs on a GPU. The
 * blocking is a scheduling detail, not semantics: for one query position the
 * result is an ordinary softmax over the gathered positions. This is written in
 * the plain form, which is what a CPU wants and what a reference should look
 * like.
 *
 * ONE KV HEAD, MANY QUERY HEADS. num_key_value_heads is 1, so kv is [n][d] and
 * every one of the h query heads attends over the same kv rows. There is no
 * per-head kv indexing anywhere.
 *
 * THREE THINGS THE SINK DOES THAT ARE EASY TO GET WRONG
 *
 * 1. IT ONLY EVER ENTERS THE DENOMINATOR:
 *        sum_exp[i] += exp(attn_sink[i] - scores_max[i])
 *    and no matching term is added to acc_o. It is a learned logit whose value
 *    vector is zero, so it absorbs probability mass and lets a head attend to
 *    nothing. Adding it to the numerator too is the obvious symmetry and it
 *    silently changes every output.
 *
 * 2. IT IS NOT PART OF THE MAX. scores_max is reduced over the score blocks
 *    only, and the sink is folded in afterwards against that already-final max.
 *    Including the sink in the max is numerically tidier and is not what the
 *    reference does; the two differ whenever the sink exceeds every score.
 *
 * 3. IT IS PER HEAD, one float per query head, and it is FP32 in the checkpoint
 *    while everything around it is BF16.
 *
 * MASKED POSITIONS. topk_idxs carries -1 for slots that should not participate.
 * The kernel sets those scores to -inf before the max, so they contribute
 * exactly zero. A slot list that is entirely -1 leaves scores_max at -inf, and
 * exp(sink - -inf) is +inf, so the output is zero -- which is the sensible
 * answer for a query attending to nothing, and is reproduced here rather than
 * special-cased.
 */
#include <math.h>
#include <string.h>

#include "dsv4.h"

/* o and q are [h][d]; kv is [n][d] and is addressed through idxs.
 * idxs holds `topk` entries, each an index into kv or -1 for "masked".
 * sink holds one float per query head. */
void dsv4_sparse_attn(float *o, const float *q, const float *kv,
                      const float *sink, const int *idxs,
                      int h, int d, int topk, float scale, float *scratch)
{
    float *s = scratch;                    /* topk scores for the current head */

    for (int i = 0; i < h; i++) {
        const float *qi = q + (size_t)i * d;

        /* Scores, and the max over the VALID ones only. */
        float mx = -INFINITY;
        for (int j = 0; j < topk; j++) {
            const int idx = idxs[j];
            if (idx < 0) { s[j] = -INFINITY; continue; }
            const float *kvj = kv + (size_t)idx * d;
            double acc = 0.0;
            for (int c = 0; c < d; c++)
                acc = fma((double)qi[c], (double)kvj[c], acc);
            s[j] = (float)acc * scale;
            if (s[j] > mx) mx = s[j];
        }

        /* Denominator: the valid scores, PLUS the sink measured against the
         * same max. The sink took no part in choosing that max. */
        double denom = 0.0;
        for (int j = 0; j < topk; j++) {
            if (idxs[j] < 0) { s[j] = 0.0f; continue; }
            s[j] = expf(s[j] - mx);
            denom += (double)s[j];
        }
        denom += (double)expf(sink[i] - mx);

        /* Numerator: valid positions only. The sink contributes no value. */
        float *oi = o + (size_t)i * d;
        for (int c = 0; c < d; c++) oi[c] = 0.0f;
        for (int j = 0; j < topk; j++) {
            const int idx = idxs[j];
            if (idx < 0) continue;
            const float w = s[j];
            if (w == 0.0f) continue;
            const float *kvj = kv + (size_t)idx * d;
            for (int c = 0; c < d; c++) oi[c] += w * kvj[c];
        }
        const float inv = (float)(1.0 / denom);
        for (int c = 0; c < d; c++) oi[c] *= inv;
    }
}

/* Sliding-window index list for one query at absolute position `pos`.
 *
 * Fills `idxs` with `win` entries: the ring-buffer slots holding the most recent
 * min(pos+1, win) positions, and -1 for slots not yet written. The cache is a
 * ring of `win` entries indexed by (position % win), which is why this returns
 * slot numbers rather than absolute positions.
 *
 * Returns the number of valid entries. */
int dsv4_window_idxs(int *idxs, int pos, int win)
{
    int n = pos + 1;
    if (n > win) n = win;
    for (int k = 0; k < win; k++) idxs[k] = -1;
    for (int k = 0; k < n; k++) {
        const int abs_pos = pos - k;
        idxs[k] = abs_pos % win;
    }
    return n;
}
