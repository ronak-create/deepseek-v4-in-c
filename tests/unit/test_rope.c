/* SPDX-License-Identifier: Apache-2.0 */
/* test_rope.c - YaRN table construction and the interleaved rotation. */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdlib.h>

void dsv4_rope_table(float *, float *, int, int, int, double, double, double, double);
void dsv4_rope_apply(float *, int, int, const float *, const float *, int, int, int);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

int main(void)
{
    const int rd = 64, half = rd / 2, seqlen = 256;
    printf("DeepSeek-V4 RoPE gate\n");

    printf("\n-- GATE 1  position 0 is the identity --\n");
    {
        static float cs[256 * 32], sn[256 * 32];
        dsv4_rope_table(cs, sn, rd, seqlen, 0, 10000.0, 16.0, 32.0, 1.0);
        for (int i = 0; i < half; i++) {
            CHECK(cs[i] == 1.0f, "cos[0][%d] = %g, expected 1", i, (double)cs[i]);
            CHECK(sn[i] == 0.0f, "sin[0][%d] = %g, expected 0", i, (double)sn[i]);
        }
        float v[512];
        for (int i = 0; i < 512; i++) v[i] = (float)(i + 1);
        float ref[512]; memcpy(ref, v, sizeof v);
        dsv4_rope_apply(v, 512, rd, cs, sn, 0, half, 0);
        CHECK(memcmp(v, ref, sizeof v) == 0, "position 0 changed the vector");
        printf("  ok    cos=1 sin=0, vector unchanged\n");
    }

    printf("\n-- GATE 2  only the LAST rd dims are touched --\n");
    {
        static float cs[256 * 32], sn[256 * 32];
        dsv4_rope_table(cs, sn, rd, seqlen, 0, 10000.0, 16.0, 32.0, 1.0);
        float v[512], ref[512];
        for (int i = 0; i < 512; i++) v[i] = (float)(i + 1);
        memcpy(ref, v, sizeof v);
        dsv4_rope_apply(v, 512, rd, cs, sn, 7, half, 0);
        /* The leading 448 NoPE dims must be byte-identical. */
        CHECK(memcmp(v, ref, (size_t)(512 - rd) * sizeof(float)) == 0,
              "the NoPE prefix was modified");
        int changed = 0;
        for (int i = 512 - rd; i < 512; i++) if (v[i] != ref[i]) changed = 1;
        CHECK(changed, "the rope tail was NOT modified");
        printf("  ok    448 NoPE dims untouched, 64 rope dims rotated\n");
    }

    printf("\n-- GATE 3  pairs are INTERLEAVED, not split-half --\n");
    {
        /* Put a single 1.0 at tail index 0 and zero elsewhere. Interleaved
         * pairing couples index 0 with index 1; split-half would couple it with
         * index rd/2 = 32. Only one of those positions becomes non-zero. */
        static float cs[256 * 32], sn[256 * 32];
        dsv4_rope_table(cs, sn, rd, seqlen, 0, 10000.0, 16.0, 32.0, 1.0);
        float v[512];
        memset(v, 0, sizeof v);
        const int base = 512 - rd;
        v[base + 0] = 1.0f;
        dsv4_rope_apply(v, 512, rd, cs, sn, 5, half, 0);

        CHECK(fabsf(v[base + 1]) > 1e-6f,
              "tail[1] is zero: the partner of tail[0] is not tail[1] (not interleaved)");
        CHECK(fabsf(v[base + 32]) < 1e-12f,
              "tail[32] is non-zero: pairing looks SPLIT-HALF, not interleaved");
        /* Rotation preserves the norm of each pair. */
        const float n2 = v[base] * v[base] + v[base + 1] * v[base + 1];
        CHECK(fabsf(n2 - 1.0f) < 1e-5f, "pair norm %.6f, expected 1", (double)n2);
        printf("  ok    tail[0] rotated into tail[1] (%.4f, %.4f), tail[32] = 0\n",
               (double)v[base], (double)v[base + 1]);
    }

    printf("\n-- GATE 4  inverse undoes the rotation exactly --\n");
    {
        /* The output de-rotation. If inverse were implemented as anything other
         * than a conjugate, round-tripping would drift. */
        static float cs[256 * 32], sn[256 * 32];
        dsv4_rope_table(cs, sn, rd, seqlen, 0, 10000.0, 16.0, 32.0, 1.0);
        float v[512], ref[512];
        unsigned st = 5u;
        for (int i = 0; i < 512; i++) {
            st = st * 1103515245u + 12345u;
            v[i] = (float)((int)((st >> 16) & 0xFF) - 128) * 0.01f;
        }
        memcpy(ref, v, sizeof v);
        dsv4_rope_apply(v, 512, rd, cs, sn, 101, half, 0);
        int moved = 0;
        for (int i = 512 - rd; i < 512; i++) if (v[i] != ref[i]) moved = 1;
        CHECK(moved, "forward rotation had no effect at position 101");
        dsv4_rope_apply(v, 512, rd, cs, sn, 101, half, 1);
        float worst = 0.0f;
        for (int i = 0; i < 512; i++) {
            const float d = fabsf(v[i] - ref[i]);
            if (d > worst) worst = d;
        }
        CHECK(worst < 1e-6f, "round trip drifted by %.3g", (double)worst);
        printf("  ok    forward then inverse returns to within %.2g\n", (double)worst);
    }

    printf("\n-- GATE 5  YaRN changes the table, and only when enabled --\n");
    {
        static float c0[256 * 32], s0[256 * 32];
        static float c1[256 * 32], s1[256 * 32];
        /* Dense layer: orig_seq_len 0, theta 10000, NO interpolation. */
        dsv4_rope_table(c0, s0, rd, seqlen, 0, 10000.0, 16.0, 32.0, 1.0);
        /* Compressed layer: orig_seq_len 65536, theta 160000, WITH it. */
        dsv4_rope_table(c1, s1, rd, seqlen, 65536, 160000.0, 16.0, 32.0, 1.0);
        CHECK(memcmp(c0, c1, sizeof c0) != 0,
              "dense and compressed layers produced identical tables");

        /* With YaRN off, the highest-frequency pair must be exactly
         * cos(t * 1.0) since freq[0] = base^0 = 1. */
        CHECK(fabsf(c0[1 * 32 + 0] - cosf(1.0f)) < 1e-6f,
              "freq[0] is not 1.0 with YaRN disabled: cos = %.6f, expected %.6f",
              (double)c0[32], (double)cosf(1.0f));
        printf("  ok    dense and compressed tables differ; freq[0] = 1 when off\n");
    }

    printf("\n-- GATE 6  YaRN divides the low-frequency tail by `factor` --\n");
    {
        /* smooth = 1 - ramp. At i below `low` the ramp is 0, so smooth is 1 and
         * the frequency is UNCHANGED. At i above `high` the ramp is 1, smooth is
         * 0, and the frequency is divided by factor. Compare the two tables at
         * the last pair, where interpolation is fully applied. */
        static float cA[2 * 32], sA[2 * 32], cB[2 * 32], sB[2 * 32];
        const double th = 160000.0, fac = 16.0;
        dsv4_rope_table(cA, sA, rd, 2, 0,     th, fac, 32.0, 1.0);   /* no yarn */
        dsv4_rope_table(cB, sB, rd, 2, 65536, th, fac, 32.0, 1.0);   /* yarn    */
        /* Recover the phase at t = 1 with atan2, NOT acos. The lowest frequency
         * is ~9e-6, so cos(freq) rounds to exactly 1.0f and acos returns 0 --
         * an earlier draft did that and measured nothing. atan2 reads the angle
         * off the sine, which keeps full relative precision near zero. */
        const double fA = atan2((double)sA[32 + (half - 1)], (double)cA[32 + (half - 1)]);
        const double fB = atan2((double)sB[32 + (half - 1)], (double)cB[32 + (half - 1)]);
        CHECK(fB < fA, "YaRN did not reduce the lowest frequency (%.3g -> %.3g)", fA, fB);
        const double ratio = fA / fB;
        CHECK(fabs(ratio - fac) < fac * 0.05,
              "lowest frequency scaled by %.3f, expected ~%.1f", ratio, fac);
        printf("  ok    lowest frequency divided by %.2f (factor is %.0f)\n",
               ratio, fac);
    }

    printf("\n");
    if (fails) { printf("ROPE GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("ROPE GATE PASSED\n");
    return 0;
}
