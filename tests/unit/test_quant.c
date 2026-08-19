/* SPDX-License-Identifier: Apache-2.0 */
/* test_quant.c - the numeric formats, checked against values derived from the
 * format definitions rather than from this implementation.
 *
 * Expectations here are hand-computed from the bit layouts (and, where kernel.py
 * states a number, from kernel.py). Deriving them by running the code under test
 * would prove only that it is self-consistent.
 *
 * NOT COVERED: the FP4 nibble ORDER. kernel.py delegates packing to the
 * float4_e2m1fn_x2 dtype, so it cannot be established from anything on disk.
 * GATE 5 pins the property that IS knowable -- that the two nibbles decode to
 * the two distinct values and that the mapping is a bijection -- and states
 * plainly what remains unverified.
 */
#include <stdio.h>
#include <math.h>
#include <string.h>
#include "dsv4_quant.h"

static int fails = 0;

#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

static void eq(const char *what, float got, float want)
{
    /* Every one of these conversions is EXACT in f32, so bit equality is the
     * right test. A tolerance here would hide a mis-rebiased exponent. */
    CHECK(memcmp(&got, &want, sizeof got) == 0,
          "%s = %.9g, expected %.9g", what, (double)got, (double)want);
}

int main(void)
{
    printf("DeepSeek-V4 numeric format gate\n");

    printf("\n-- GATE 1  E8M0 scale is exactly 2^(b-127) --\n");
    eq("e8m0[127]", dsv4_e8m0_to_f32(127), 1.0f);
    eq("e8m0[128]", dsv4_e8m0_to_f32(128), 2.0f);
    eq("e8m0[126]", dsv4_e8m0_to_f32(126), 0.5f);
    eq("e8m0[137]", dsv4_e8m0_to_f32(137), 1024.0f);
    eq("e8m0[117]", dsv4_e8m0_to_f32(117), 1.0f / 1024.0f);
    eq("e8m0[1]",   dsv4_e8m0_to_f32(1),   ldexpf(1.0f, -126));
    /* Both ends are special and both were wrong in the first implementation.
     * 0 denotes 2^-127, which is BELOW f32's smallest normal and must come back
     * as a subnormal rather than as +0.0 -- a zero here silently blanks a whole
     * block of weights. 255 is the format's NaN, not +infinity. */
    eq("e8m0[0] = 2^-127", dsv4_e8m0_to_f32(0), ldexpf(1.0f, -127));
    CHECK(dsv4_e8m0_to_f32(0) != 0.0f, "e8m0[0] must NOT be zero");
    CHECK(isnan(dsv4_e8m0_to_f32(255)), "e8m0[255] must be NaN, not +inf");
    CHECK(!isinf(dsv4_e8m0_to_f32(255)), "e8m0[255] must not be infinity");
    /* Cross-check against the definition itself, over the whole normal range. */
    for (int b = 1; b < 255; b++)
        CHECK(dsv4_e8m0_to_f32((uint8_t)b) == ldexpf(1.0f, b - 127),
              "e8m0[%d] != 2^%d", b, b - 127);
    printf("  ok    all 254 normal exponents match ldexp exactly\n");

    printf("\n-- GATE 2  E4M3 normals, subnormals and the FN edge --\n");
    eq("e4m3 0x00 (+0)",   dsv4_e4m3_to_f32(0x00), 0.0f);
    eq("e4m3 0x38 (1.0)",  dsv4_e4m3_to_f32(0x38), 1.0f);   /* exp 7, man 0 */
    eq("e4m3 0xB8 (-1.0)", dsv4_e4m3_to_f32(0xB8), -1.0f);
    eq("e4m3 0x3C (1.5)",  dsv4_e4m3_to_f32(0x3C), 1.5f);   /* man 4 -> 1.5   */
    eq("e4m3 0x3A (1.25)", dsv4_e4m3_to_f32(0x3A), 1.25f);  /* man 2 -> 1.25  */
    eq("e4m3 0x40 (2.0)",  dsv4_e4m3_to_f32(0x40), 2.0f);
    eq("e4m3 0x30 (0.5)",  dsv4_e4m3_to_f32(0x30), 0.5f);
    /* Subnormals: value = man * 2^-9. */
    eq("e4m3 0x01 (min sub)", dsv4_e4m3_to_f32(0x01), ldexpf(1.0f, -9));
    eq("e4m3 0x02",           dsv4_e4m3_to_f32(0x02), ldexpf(2.0f, -9));
    eq("e4m3 0x07 (max sub)", dsv4_e4m3_to_f32(0x07), ldexpf(7.0f, -9));
    /* THE FN EDGE. exp==15 is NOT infinity here: it is an ordinary normal
     * range, and only 0x7F/0xFF are NaN. An IEEE-shaped implementation returns
     * inf for 0x7E and still looks reasonable downstream.
     *   0x7E: exp 15, man 6 -> 2^8 * (1 + 6/8) = 448   <- the max finite
     *   0x7B: exp 15, man 3 -> 2^8 * (1 + 3/8) = 352
     * These two lines first carried 416 and 240, which were my arithmetic
     * errors rather than the implementation's. */
    eq("e4m3 0x7E (448)",  dsv4_e4m3_to_f32(0x7E), 448.0f);
    eq("e4m3 0x7B (352)",  dsv4_e4m3_to_f32(0x7B), 352.0f);
    CHECK(isnan(dsv4_e4m3_to_f32(0x7F)), "0x7F must be NaN");
    CHECK(isnan(dsv4_e4m3_to_f32(0xFF)), "0xFF must be NaN");
    CHECK(!isinf(dsv4_e4m3_to_f32(0x7E)), "0x7E must be FINITE (FN has no inf)");
    /* Max finite is 448, at 0x7E (exp 15, man 6); 0x7F is the NaN. */
    {
        float mx = 0.0f;
        for (int b = 0; b < 256; b++) {
            float v = dsv4_e4m3_to_f32((uint8_t)b);
            if (isfinite(v) && v > mx) mx = v;
        }
        CHECK(mx == 448.0f, "max finite e4m3 is %.1f, expected 448", (double)mx);
        printf("  ok    max finite = %.0f, 0x7E = %.0f (finite, not inf)\n",
               (double)mx, (double)dsv4_e4m3_to_f32(0x7E));
    }
    /* Sign symmetry over every encoding. */
    for (int b = 0; b < 128; b++) {
        float p = dsv4_e4m3_to_f32((uint8_t)b);
        float n = dsv4_e4m3_to_f32((uint8_t)(b | 0x80));
        if (isnan(p) || isnan(n)) continue;
        CHECK(n == -p, "e4m3 0x%02X and 0x%02X are not negatives", b, b | 0x80);
    }
    printf("  ok    sign symmetry holds across all encodings\n");

    printf("\n-- GATE 3  E2M1 has exactly eight magnitudes, max 6 --\n");
    {
        const float want[8] = { 0.0f, 0.5f, 1.0f, 1.5f, 2.0f, 3.0f, 4.0f, 6.0f };
        for (int i = 0; i < 8; i++) {
            eq("e2m1 +", dsv4_e2m1_to_f32((uint8_t)i), want[i]);
            CHECK(dsv4_e2m1_to_f32((uint8_t)(i | 8)) == -want[i],
                  "e2m1 code %d is not the negative of code %d", i | 8, i);
        }
        /* kernel.py sets fp4_max = 6.0; the table's largest entry must agree,
         * which is the cross-check that this is the right table at all. */
        float mx = 0.0f;
        for (int i = 0; i < 16; i++) {
            float v = dsv4_e2m1_to_f32((uint8_t)i);
            if (v > mx) mx = v;
        }
        CHECK(mx == 6.0f, "max e2m1 is %.1f, kernel.py says fp4_max = 6.0", (double)mx);
        printf("  ok    {0,.5,1,1.5,2,3,4,6} and signs, max matches fp4_max\n");
    }

    printf("\n-- GATE 4  block scale indexing distinguishes the two grids --\n");
    {
        /* An FP8 grid over a 256x256 matrix is 2x2; an FP4 grid over the same
         * is 256x8. Reading one with the other's geometry lands on a valid byte
         * and yields a plausible number, which is the whole hazard. */
        uint8_t g_fp8[4] = { 127, 128, 129, 130 };      /* 1, 2, 4, 8 */
        eq("fp8 blk (0,0)",     dsv4_block_scale(g_fp8, 2, 0,   0,   128, 128), 1.0f);
        eq("fp8 blk (0,200)",   dsv4_block_scale(g_fp8, 2, 0,   200, 128, 128), 2.0f);
        eq("fp8 blk (200,0)",   dsv4_block_scale(g_fp8, 2, 200, 0,   128, 128), 4.0f);
        eq("fp8 blk (200,200)", dsv4_block_scale(g_fp8, 2, 200, 200, 128, 128), 8.0f);

        uint8_t g_fp4[16];
        for (int i = 0; i < 16; i++) g_fp4[i] = (uint8_t)(127 + i);
        /* row 1, col 64 -> block (1, 2) -> index 1*8 + 2 = 10 -> 2^(137-127) */
        eq("fp4 blk (1,64)", dsv4_block_scale(g_fp4, 8, 1, 64, 1, 32), 1024.0f);
        /* Same coordinates read with the FP8 geometry give a DIFFERENT, finite
         * answer -- the failure mode this struct exists to prevent. */
        CHECK(dsv4_block_scale(g_fp4, 8, 1, 64, 128, 128) !=
              dsv4_block_scale(g_fp4, 8, 1, 64, 1, 32),
              "the two grid geometries must not agree, or this test proves nothing");
        printf("  ok    128x128 and 1x32 indexing are distinct and correct\n");
    }

    printf("\n-- GATE 5  FP4 packing: the LOW nibble is the EVEN element --\n");
    {
        /* What is knowable without a reference: the two nibbles of a byte
         * decode to the two distinct codes, and every element index maps to
         * exactly one nibble. */
        uint8_t row[8];
        for (int i = 0; i < 8; i++) row[i] = (uint8_t)((i * 2 + 1) << 4 | (i * 2));
        for (int i = 0; i < 16; i++) {
            float v = dsv4_fp4_at(row, i);
            int found = 0;
            for (int k = 0; k < 16; k++)
                if (memcmp(&v, &dsv4_e2m1_lut[k], sizeof v) == 0) found = 1;
            CHECK(found, "element %d decoded to a value outside the e2m1 table", i);
        }
        /* Adjacent elements come from the same byte and must differ. */
        for (int i = 0; i < 16; i += 2)
            CHECK(dsv4_fp4_at(row, i) != dsv4_fp4_at(row, i + 1),
                  "elements %d and %d decoded identically; the nibble split is wrong",
                  i, i + 1);
        printf("  ok    every element decodes to a table value; pairs are distinct\n");

        /* THE ORDER ITSELF, from torch/headeronly/util/Float4_e2m1fn_x2.h:
         *
         *     original value             | val1 : val0
         *     bit index (MSB==7, LSB==0) | 7654 : 3210
         *
         * val0 -- the first, even-indexed element -- is the LOW nibble. That
         * header is normative here, not merely conventional, because the
         * checkpoint's expert weights were written BY PyTorch into this dtype.
         *
         * Byte 0x21: low nibble 1 -> 0.5 at index 0, high nibble 2 -> 1.0 at
         * index 1. Swap the convention and this reads {1.0, 0.5}. */
        {
            uint8_t one[1] = { 0x21 };
            CHECK(dsv4_fp4_at(one, 0) == 0.5f,
                  "element 0 = %.3f, expected 0.5 (low nibble of 0x21)",
                  (double)dsv4_fp4_at(one, 0));
            CHECK(dsv4_fp4_at(one, 1) == 1.0f,
                  "element 1 = %.3f, expected 1.0 (high nibble of 0x21)",
                  (double)dsv4_fp4_at(one, 1));
            printf("  ok    0x21 -> {%.1f, %.1f}, matching PyTorch's val1:val0\n",
                   (double)dsv4_fp4_at(one, 0), (double)dsv4_fp4_at(one, 1));
        }
    }

    printf("\n");
    if (fails) { printf("QUANT GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("QUANT GATE PASSED\n");
    return 0;
}
