/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_compress.c - HCA: the KV compressor.
 *
 * Source: model.py Compressor. Present only where compress_ratio != 0.
 *
 * WHAT IT DOES
 *   Instead of caching every KV row, a compressed layer pools `ratio`
 *   consecutive rows into one, weighted by a learned gate. The pooling is a
 *   SOFTMAX OVER THE SLOT AXIS, computed independently for every channel:
 *
 *       out[c] = SUM_s  kv_state[s][c] * softmax_over_s(score_state[.][c])[s]
 *
 *   Not a softmax over channels, and not one shared weight per slot. Getting
 *   the axis wrong gives a correctly-shaped, plausibly-scaled, different result.
 *
 * TWO MODES, AND coff IS WHY THE TENSORS ARE TWO SIZES
 *   overlap = (ratio == 4), coff = 1 + overlap.
 *   wkv and wgate produce coff*head_dim, so on a ratio-4 layer they are twice
 *   as wide (checkpoint: layer 2 wkv is [1024, 4096], layer 3 is [512, 4096]).
 *   In overlap mode the FIRST half of those channels belongs to the overlapping
 *   window and the SECOND half to the current one, and the pool is assembled as
 *
 *       cat([state[:ratio, :d], state[ratio:, d:]])
 *
 *   giving 2*ratio slots of width d. Treating the wide tensor as one flat
 *   2d-wide vector is the natural reading and is wrong.
 *
 * THE POSITION USED FOR RoPE IS NOT THE CURRENT ONE
 *       freqs_cis = self.freqs_cis[start_pos + 1 - compress_ratio]
 *   The compressed row is stamped with the position of the FIRST token in the
 *   window it summarises, not the last. Using start_pos shifts every compressed
 *   key by ratio-1 positions and still decodes.
 *
 * SCOPE: this file implements the DECODE path (start_pos > 0), which is what a
 * streaming engine runs for every token after the prompt. The prefill path
 * (start_pos == 0) has its own remainder/cutoff handling and is not here yet.
 */
#include <math.h>
#include <string.h>

#include "dsv4.h"

/* Softmax over the SLOT axis for one channel, then the weighted sum.
 * -INFINITY in a slot marks it unfilled; those contribute exactly zero. */
static void pool_channel(float *out, const float *kv, const float *sc,
                         int slots, int kv_stride, int sc_stride, int c)
{
    float mx = -INFINITY;
    for (int s = 0; s < slots; s++) {
        const float v = sc[(size_t)s * sc_stride + c];
        if (v > mx) mx = v;
    }
    if (mx == -INFINITY) { *out = 0.0f; return; }

    double denom = 0.0, acc = 0.0;
    for (int s = 0; s < slots; s++) {
        const float v = sc[(size_t)s * sc_stride + c];
        if (v == -INFINITY) continue;
        const double e = (double)expf(v - mx);
        denom += e;
        acc += e * (double)kv[(size_t)s * kv_stride + c];
    }
    *out = (float)(acc / denom);
}

/* One decode step.
 *
 *   kv_in, score_in   coff*head_dim, already projected by wkv / wgate
 *   ape               [ratio][coff*head_dim]
 *   kv_state,
 *   score_state       [coff*ratio][coff*head_dim], caller-owned, persistent
 *   out               head_dim, written only when a window closes
 *
 * Returns 1 when a compressed row was produced, 0 otherwise. The caller applies
 * norm, RoPE and quantisation to `out` and stores it at kv_cache[pos / ratio].
 *
 * `score_state` must be initialised to -INFINITY before the first call. */
int dsv4_compress_step(float *out, const float *kv_in, const float *score_in,
                       const float *ape, float *kv_state, float *score_state,
                       int pos, int ratio, int head_dim)
{
    const int overlap = (ratio == 4);
    const int coff    = 1 + overlap;
    const int width   = coff * head_dim;          /* channels per slot */
    const int nslot   = coff * ratio;
    const int phase   = pos % ratio;

    /* score += ape[pos % ratio], elementwise over the full width. */
    const float *apr = ape + (size_t)phase * width;

    /* Write this token into its slot. In overlap mode the live half of the ring
     * starts at `ratio`; the first `ratio` slots hold the previous window. */
    const int slot = overlap ? (ratio + phase) : phase;
    float *kvs = kv_state    + (size_t)slot * width;
    float *scs = score_state + (size_t)slot * width;
    for (int c = 0; c < width; c++) {
        kvs[c] = kv_in[c];
        scs[c] = score_in[c] + apr[c];
    }

    if ((pos + 1) % ratio != 0) return 0;         /* window still open */

    if (!overlap) {
        for (int c = 0; c < head_dim; c++)
            pool_channel(out + c, kv_state, score_state, nslot, width, width, c);
    } else {
        /* Assemble 2*ratio slots of width d:
         *   slots 0..ratio-1      from state[0..ratio-1][0 .. d)      (previous)
         *   slots ratio..2ratio-1 from state[ratio..][d .. 2d)        (current)
         * Done by pooling the two halves against a gathered view. */
        static float gk[2 * 4 * 4096];            /* ratio is 4 in overlap mode */
        static float gs[2 * 4 * 4096];
        const int d = head_dim;
        for (int s = 0; s < ratio; s++) {
            memcpy(gk + (size_t)s * d, kv_state    + (size_t)s * width,
                   (size_t)d * sizeof(float));
            memcpy(gs + (size_t)s * d, score_state + (size_t)s * width,
                   (size_t)d * sizeof(float));
        }
        for (int s = 0; s < ratio; s++) {
            memcpy(gk + (size_t)(ratio + s) * d,
                   kv_state    + (size_t)(ratio + s) * width + d,
                   (size_t)d * sizeof(float));
            memcpy(gs + (size_t)(ratio + s) * d,
                   score_state + (size_t)(ratio + s) * width + d,
                   (size_t)d * sizeof(float));
        }
        for (int c = 0; c < d; c++)
            pool_channel(out + c, gk, gs, 2 * ratio, d, d, c);

        /* Slide: the current window becomes the previous one. */
        memcpy(kv_state,    kv_state    + (size_t)ratio * width,
               (size_t)ratio * width * sizeof(float));
        memcpy(score_state, score_state + (size_t)ratio * width,
               (size_t)ratio * width * sizeof(float));
    }
    return 1;
}

/* The position a compressed row is stamped with: the FIRST token of the window
 * it summarises, not the last. See the note at the top. */
int dsv4_compress_rope_pos(int pos, int ratio) { return pos + 1 - ratio; }
