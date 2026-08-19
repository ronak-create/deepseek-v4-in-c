/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_rope.c - rotary embeddings with YaRN frequency interpolation.
 *
 * Source: model.py precompute_freqs_cis and apply_rotary_emb.
 *
 * FOUR THINGS THAT ARE NOT WHAT A READER EXPECTS
 *
 * 1. THE PAIRS ARE INTERLEAVED, NOT SPLIT-HALF. apply_rotary_emb does
 *      view_as_complex(x.unflatten(-1, (-1, 2)))
 *    so the complex pairs are (x[0],x[1]), (x[2],x[3]), ... This is the GPT-J
 *    convention. The far more common GPT-NeoX / "rotate_half" convention pairs
 *    x[i] with x[i + d/2] instead. Both rotate, both preserve norm, both give
 *    fluent text, and they are different models.
 *
 * 2. ROPE COVERS ONLY THE LAST rope_head_dim DIMENSIONS. The call sites are
 *      apply_rotary_emb(q[..., -rd:], ...)
 *    with rd = 64 out of head_dim = 512. The leading 448 dims are NoPE and must
 *    be left exactly alone.
 *
 * 3. YARN IS DISABLED ON DENSE LAYERS. Attention.__init__:
 *      if compress_ratio:  original_seq_len, theta = args.original_seq_len,
 *                                                   args.compress_rope_theta
 *      else:               original_seq_len, theta = 0, args.rope_theta
 *    So a compress_ratio 0 layer uses theta = 10000 with NO interpolation,
 *    while every compressed layer uses theta = 160000 WITH it. Building one
 *    table for the whole model is wrong on one set of layers or the other.
 *
 * 4. THE OUTPUT IS DE-ROTATED. After attention,
 *      apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
 *    which conjugates the rotation. Omitting it leaves the output in the
 *    rotated frame, which is a plausible-looking and wrong residual stream.
 */
#include <math.h>
#include <string.h>
#include <stdlib.h>

#include "dsv4.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

/* YaRN correction range, verbatim from find_correction_dim/find_correction_range.
 * NOTE the argument order at the call site: beta_fast is passed as low_rot and
 * beta_slow as high_rot. find_correction_dim decreases in num_rotations, so with
 * beta_fast = 32 > beta_slow = 1 this does give low < high. Swapping them
 * inverts the ramp and still produces a monotone frequency table. */
static void correction_range(double low_rot, double high_rot, int dim,
                             double base, int max_seq_len, int *lo, int *hi)
{
    const double l = (double)dim * log((double)max_seq_len / (low_rot  * 2.0 * M_PI))
                     / (2.0 * log(base));
    const double h = (double)dim * log((double)max_seq_len / (high_rot * 2.0 * M_PI))
                     / (2.0 * log(base));
    int li = (int)floor(l), hi_ = (int)ceil(h);
    if (li < 0) li = 0;
    if (hi_ > dim - 1) hi_ = dim - 1;
    *lo = li; *hi = hi_;
}

/* Build cos/sin tables for `seqlen` positions and dim/2 frequency pairs.
 *
 * cos and sin must each hold seqlen * (dim/2) floats.
 *
 * orig_seq_len == 0 disables YaRN entirely, which is the dense-layer path. */
void dsv4_rope_table(float *cs, float *sn, int dim, int seqlen,
                     int orig_seq_len, double base, double factor,
                     double beta_fast, double beta_slow)
{
    const int half = dim / 2;
    double *freqs = (double *)malloc((size_t)half * sizeof(double));
    if (!freqs) return;

    for (int i = 0; i < half; i++)
        freqs[i] = 1.0 / pow(base, (double)(2 * i) / (double)dim);

    if (orig_seq_len > 0) {
        int lo, hi;
        correction_range(beta_fast, beta_slow, dim, base, orig_seq_len, &lo, &hi);
        double mn = (double)lo, mx = (double)hi;
        if (mn == mx) mx += 0.001;            /* matches linear_ramp_factor */
        for (int i = 0; i < half; i++) {
            double ramp = ((double)i - mn) / (mx - mn);
            if (ramp < 0.0) ramp = 0.0;
            if (ramp > 1.0) ramp = 1.0;
            const double smooth = 1.0 - ramp;
            /* freqs/factor * (1-smooth) + freqs * smooth */
            freqs[i] = freqs[i] / factor * (1.0 - smooth) + freqs[i] * smooth;
        }
    }

    for (int t = 0; t < seqlen; t++)
        for (int i = 0; i < half; i++) {
            const double th = (double)t * freqs[i];
            cs[(size_t)t * half + i] = (float)cos(th);
            sn[(size_t)t * half + i] = (float)sin(th);
        }
    free(freqs);
}

/* Rotate the LAST `rd` elements of a `head_dim`-long vector in place.
 *
 * Pairs are INTERLEAVED: (v[0],v[1]), (v[2],v[3]), ... within the rd-long tail.
 * `inverse` conjugates, i.e. negates the sine, which is what the output
 * de-rotation needs. */
void dsv4_rope_apply(float *v, int head_dim, int rd, const float *cs,
                     const float *sn, int pos, int table_half, int inverse)
{
    float *tail = v + (head_dim - rd);
    const float *c = cs + (size_t)pos * table_half;
    const float *s = sn + (size_t)pos * table_half;
    const float sgn = inverse ? -1.0f : 1.0f;

    for (int i = 0; i < rd / 2; i++) {
        const float re = tail[2 * i], im = tail[2 * i + 1];
        const float ci = c[i], si = sgn * s[i];
        tail[2 * i]     = re * ci - im * si;
        tail[2 * i + 1] = re * si + im * ci;
    }
}
