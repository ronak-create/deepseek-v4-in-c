/* SPDX-License-Identifier: Apache-2.0 */
/* dsv4_quant.h - the three numeric formats DeepSeek-V4 ships weights in.
 *
 * Header-only and branch-free per element, because every one of these runs
 * inside a matmul inner loop. Nothing here allocates and nothing here widens a
 * whole tensor: the packed bytes stay packed and are decoded where they are
 * consumed. Widening Flash's routed experts to f32 would cost 8x their 137 GB.
 *
 * WHAT WAS VERIFIED, AND HOW
 *   Every claim below comes from the checkpoint's own inference/kernel.py or
 *   from the released tensor headers, not from the model card.
 *
 *   E8M0 scale          kernel.py fast_pow2 is `(x + 127) << 23` reinterpreted
 *                       as f32, so a stored byte b denotes exactly 2^(b-127).
 *                       Exponent only: no sign, no mantissa. 0xFF is NaN.
 *   E4M3 weights        torch.float8_e4m3fn: 1 sign, 4 exponent (bias 7),
 *                       3 mantissa. FN = "finite": there are no infinities, and
 *                       0x7F / 0xFF are the only NaNs. Max finite is 448.
 *   E2M1 weights        torch.float4_e2m1fn: 1 sign, 2 exponent (bias 1),
 *                       1 mantissa, giving exactly eight magnitudes
 *                       {0, .5, 1, 1.5, 2, 3, 4, 6}. kernel.py sets
 *                       fp4_max = 6.0, which agrees.
 *   Dequantisation      kernel.py fp4_quant_kernel stores x/scale clamped to
 *                       +/-6, so recovering x is value * scale. fp4_gemm applies
 *                       the scale to the ACCUMULATOR per 32-wide K block, which
 *                       is algebraically the same thing.
 *   Block grids         weights 1x32 along K (E8M0), activations 1x128.
 *                       The 128x128 in config.json's weight_block_size describes
 *                       the FP8 tensors only.
 *
 * THE NIBBLE ORDER, NOW SETTLED
 *   kernel.py delegates packing to the float4_e2m1fn_x2 dtype, so nothing in
 *   the checkpoint states which nibble holds the even-indexed element. But the
 *   checkpoint was WRITTEN by PyTorch into that dtype, which makes PyTorch's
 *   definition of it the specification rather than a convention to guess at.
 *
 *   From torch/headeronly/util/Float4_e2m1fn_x2.h (torch 2.11.0+cu128), which
 *   cites OCP Microscaling Formats MX v1.0 section 5.3.3:
 *
 *       original value             | val1 : val0
 *       ========================================
 *       bit index (MSB==7, LSB==0) | 7654 : 3210
 *       sign/exponent/mantissa     | seem : seem
 *
 *   val0 -- the FIRST, even-indexed element -- is bits 3..0, the LOW nibble.
 *   So DSV4_FP4_LOW_NIBBLE_FIRST = 1 is correct.
 *
 *   This was worth pinning down rather than assuming. Getting it wrong swaps
 *   adjacent weights within every pair, and the value HISTOGRAM is unchanged,
 *   so no statistical check could ever have caught it -- only a normative
 *   statement or a bit-exact reference comparison.
 *
 *   Two routes were tried first and both dead-ended, which is why the citation
 *   above is the evidence rather than a measurement: torch implements NO
 *   element-wise conversion for this dtype on CPU (copy_kernel unimplemented),
 *   and on CUDA sm_120 the generic cast asserts inside fetch_and_cast. FP4 is
 *   reachable only through specific ops such as _scaled_mm, never through .to().
 */
#ifndef DSV4_QUANT_H
#define DSV4_QUANT_H

#include <stdint.h>

/* val0 (even element) is the LOW nibble. Established from PyTorch's own
 * Float4_e2m1fn_x2.h, which is normative here because the checkpoint was
 * written by PyTorch into that dtype. See the header note above. */
#define DSV4_FP4_LOW_NIBBLE_FIRST 1

/* ------------------------------------------------------------ E8M0 scale ---
 * 2^(b-127), exponent only. Built by bit assembly rather than by ldexp so the
 * result is exact and the function stays branch-free in the inner loop.
 * b == 0xFF is NaN in the format; it does not occur in a well-formed
 * checkpoint, and producing NaN here is the correct, loud behaviour if it does. */
static inline float dsv4_e8m0_to_f32(uint8_t b)
{
    union { uint32_t u; float f; } v;
    /* Both ends need spelling out, and a first attempt at this function got both
     * wrong while looking obviously correct:
     *   b == 0xFF is the format's NaN. Shifting alone gives exponent 255 with a
     *     zero mantissa, which is +INFINITY -- a different, quieter failure.
     *   b == 0 denotes 2^-127. That is BELOW f32's smallest normal (2^-126), so
     *     it must be written as the subnormal 0x00400000; shifting alone gives
     *     exactly +0.0, which would silently zero a whole block of weights. */
    if (b == 0xFFu) { v.u = 0x7FC00000u; return v.f; }   /* NaN                */
    if (b == 0x00u) { v.u = 0x00400000u; return v.f; }   /* 2^-127, subnormal  */
    v.u = (uint32_t)b << 23;      /* exponent field = b, mantissa 0, sign 0 */
    return v.f;
}

/* ------------------------------------------------------ E4M3 (float8_e4m3fn)
 * Widening is exact: every e4m3 value is representable in f32. Subnormals must
 * be renormalised, which is the only part that is not a shift.
 *
 *   sign  1 bit   [7]
 *   exp   4 bits  [6:3], bias 7
 *   man   3 bits  [2:0]
 *
 * FN semantics: exp==15 && man==7 is NaN; every other exp==15 encoding is a
 * NORMAL number (up to 448), unlike IEEE where it would be infinity. Treating
 * 0x7E as infinity instead of 416 is a silent, plausible error. */
static inline float dsv4_e4m3_to_f32(uint8_t b)
{
    const uint32_t sign = (uint32_t)(b & 0x80u) << 24;
    const uint32_t exp  = (uint32_t)(b >> 3) & 0x0Fu;
    const uint32_t man  = (uint32_t)b & 0x07u;
    union { uint32_t u; float f; } v;

    if (exp == 0) {
        if (man == 0) { v.u = sign; return v.f; }   /* +/- 0 */
        /* Subnormal: value = man * 2^-9, man in 1..7.
         *
         * Writing man = 1.f * 2^k gives value = 1.f * 2^(k-9), so the f32
         * exponent field is 127 + k - 9 = 118 + k. Normalising shifts left
         * until bit 2 is set; s shifts means k = 2 - s, so the field is
         * 120 - s. Starting from 121 (i.e. 127-6) is the natural-looking
         * mistake and makes every subnormal exactly twice too large. */
        uint32_t m = man, e = 120u;
        while (!(m & 0x04u)) { m <<= 1; e--; }
        m &= 0x03u;                                 /* 2 fraction bits remain */
        v.u = sign | (e << 23) | (m << 21);
        return v.f;
    }
    if (exp == 0x0Fu && man == 0x07u) {             /* the only NaN encodings */
        v.u = sign | 0x7FC00000u;
        return v.f;
    }
    /* Normal: rebias 7 -> 127 and left-align the 3 mantissa bits. */
    v.u = sign | ((exp + 120u) << 23) | (man << 20);
    return v.f;
}

/* --------------------------------------------------- E2M1 (float4_e2m1fn) ---
 * Only sixteen encodings exist, so a table is both exact and faster than any
 * bit manipulation. Index is the 4-bit code; the sign bit is bit 3.
 *
 *   code 0..7  = +{0, 0.5, 1, 1.5, 2, 3, 4, 6}
 *   code 8..15 = the same, negated
 *
 * There is no NaN and no infinity in e2m1fn. fp4_max = 6.0 in kernel.py is this
 * table's largest entry, which is the cross-check that it is the right table. */
static const float dsv4_e2m1_lut[16] = {
     0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f
};

static inline float dsv4_e2m1_to_f32(uint8_t nibble)
{
    return dsv4_e2m1_lut[nibble & 0x0Fu];
}

/* Element i of a packed FP4 row. `row` is the stored bytes; i is the LOGICAL
 * element index, so the byte is i/2 and the nibble is chosen by i&1.
 *
 * Order per PyTorch's Float4_e2m1fn_x2.h: val0 is bits 3..0. */
static inline float dsv4_fp4_at(const uint8_t *row, int64_t i)
{
    const uint8_t byte = row[i >> 1];
#if DSV4_FP4_LOW_NIBBLE_FIRST
    const uint8_t nib = (i & 1) ? (uint8_t)(byte >> 4) : (uint8_t)(byte & 0x0Fu);
#else
    const uint8_t nib = (i & 1) ? (uint8_t)(byte & 0x0Fu) : (uint8_t)(byte >> 4);
#endif
    return dsv4_e2m1_lut[nib];
}

/* ------------------------------------------------------------- dequant -----
 * Scale lookup for one element of a block-quantised matrix.
 *
 * `scales` is the E8M0 grid, `sr` x `sc` in shape, laid out row-major. For an
 * FP8 tensor the block is 128x128; for an FP4 tensor it is 1x32. Passing the
 * FP8 grid for an FP4 tensor reads a valid byte at a wrong index and yields a
 * plausible number, which is why DSV4QMat carries its own blk_r/blk_c and
 * nothing here takes them as defaults. */
static inline float dsv4_block_scale(const uint8_t *scales, int sc,
                                     int64_t r, int64_t c, int blk_r, int blk_c)
{
    return dsv4_e8m0_to_f32(scales[(r / blk_r) * (int64_t)sc + (c / blk_c)]);
}

#endif /* DSV4_QUANT_H */
