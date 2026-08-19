/* SPDX-License-Identifier: Apache-2.0 */
/* test_compress.c - the HCA compressor's decode path.
 *
 * The failures worth catching here all produce a correctly-shaped, finite,
 * plausibly-scaled compressed row:
 *   - softmax over the wrong axis (channels instead of slots)
 *   - one shared gate weight per slot instead of one per channel
 *   - the overlap halves spliced the wrong way round
 *   - emitting on the wrong step of the window
 *   - stamping the compressed row with the last position instead of the first
 */
#include <stdio.h>
#include <string.h>
#include <math.h>

int dsv4_compress_step(float *, const float *, const float *, const float *,
                       float *, float *, int, int, int);
int dsv4_compress_rope_pos(int, int);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

/* ratio 128 is the non-overlap mode; use a small stand-in ratio so the test is
 * readable. The code branches on ratio == 4 for overlap, so any other value
 * exercises the non-overlap path identically. */
#define R    3          /* non-overlap ratio */
#define D    4          /* head_dim */

static void reset(float *ks, float *ss, int nslot, int width)
{
    memset(ks, 0, (size_t)nslot * width * sizeof(float));
    for (int i = 0; i < nslot * width; i++) ss[i] = -INFINITY;
}

int main(void)
{
    printf("DeepSeek-V4 HCA compressor gate\n");

    printf("\n-- GATE 1  emits only when the window closes --\n");
    {
        float ks[R * D], ss[R * D], out[D];
        float kv[D] = { 1, 2, 3, 4 }, sc[D] = { 0, 0, 0, 0 }, ape[R * D] = {0};
        reset(ks, ss, R, D);
        int emitted = 0;
        for (int pos = 0; pos < R * 2; pos++)
            emitted += dsv4_compress_step(out, kv, sc, ape, ks, ss, pos, R, D);
        CHECK(emitted == 2, "%d emissions over %d steps, expected 2", emitted, R * 2);
        /* And specifically on the last step of each window. */
        reset(ks, ss, R, D);
        CHECK(dsv4_compress_step(out, kv, sc, ape, ks, ss, 0, R, D) == 0, "pos 0 emitted");
        CHECK(dsv4_compress_step(out, kv, sc, ape, ks, ss, 1, R, D) == 0, "pos 1 emitted");
        CHECK(dsv4_compress_step(out, kv, sc, ape, ks, ss, 2, R, D) == 1, "pos 2 did not emit");
        printf("  ok    emits on pos 2, 5, ... only\n");
    }

    printf("\n-- GATE 2  equal gates give the mean of the window --\n");
    {
        float ks[R * D], ss[R * D], out[D];
        float ape[R * D] = {0}, sc[D] = { 0, 0, 0, 0 };
        reset(ks, ss, R, D);
        for (int pos = 0; pos < R; pos++) {
            float kv[D];
            for (int c = 0; c < D; c++) kv[c] = (float)(pos + 1);
            dsv4_compress_step(out, kv, sc, ape, ks, ss, pos, R, D);
        }
        for (int c = 0; c < D; c++)
            CHECK(fabsf(out[c] - 2.0f) < 1e-5f, "out[%d] = %.6f, expected 2 "
                  "(mean of 1,2,3)", c, (double)out[c]);
        printf("  ok    mean of 1,2,3 = %.4f\n", (double)out[0]);
    }

    printf("\n-- GATE 3  the softmax is over SLOTS, PER CHANNEL --\n");
    {
        /* Give each channel a different winning slot. A softmax over channels,
         * or a single gate shared across channels, cannot reproduce this. */
        float ks[R * D], ss[R * D], out[D];
        float ape[R * D] = {0};
        reset(ks, ss, R, D);
        for (int pos = 0; pos < R; pos++) {
            float kv[D], sc[D];
            for (int c = 0; c < D; c++) {
                kv[c] = (float)(pos + 1) * 10.0f;
                /* channel c is dominated by slot (c % R) */
                sc[c] = (pos == (c % R)) ? 20.0f : -20.0f;
            }
            dsv4_compress_step(out, kv, sc, ape, ks, ss, pos, R, D);
        }
        for (int c = 0; c < D; c++) {
            const float want = (float)((c % R) + 1) * 10.0f;
            CHECK(fabsf(out[c] - want) < 0.1f,
                  "channel %d = %.4f, expected ~%.1f (slot %d should win)",
                  c, (double)out[c], (double)want, c % R);
        }
        CHECK(out[0] != out[1], "all channels agree; the gate is not per channel");
        printf("  ok    channels select different slots: %.1f %.1f %.1f %.1f\n",
               (double)out[0], (double)out[1], (double)out[2], (double)out[3]);
    }

    printf("\n-- GATE 4  ape is added per phase, not once --\n");
    {
        /* ape[phase] biases the gate. Make ape favour the LAST slot strongly and
         * check the output follows it, then flip it to the first. */
        float ks[R * D], ss[R * D], out[D];
        float sc[D] = {0};
        float ape_last[R * D], ape_first[R * D];
        for (int p = 0; p < R; p++)
            for (int c = 0; c < D; c++) {
                ape_last[p * D + c]  = (p == R - 1) ? 20.0f : -20.0f;
                ape_first[p * D + c] = (p == 0)     ? 20.0f : -20.0f;
            }
        reset(ks, ss, R, D);
        for (int pos = 0; pos < R; pos++) {
            float kv[D];
            for (int c = 0; c < D; c++) kv[c] = (float)(pos + 1);
            dsv4_compress_step(out, kv, sc, ape_last, ks, ss, pos, R, D);
        }
        CHECK(fabsf(out[0] - 3.0f) < 0.01f,
              "with ape favouring the last slot, got %.4f, expected ~3", (double)out[0]);
        const float favour_last = out[0];

        reset(ks, ss, R, D);
        for (int pos = 0; pos < R; pos++) {
            float kv[D];
            for (int c = 0; c < D; c++) kv[c] = (float)(pos + 1);
            dsv4_compress_step(out, kv, sc, ape_first, ks, ss, pos, R, D);
        }
        CHECK(fabsf(out[0] - 1.0f) < 0.01f,
              "with ape favouring the first slot, got %.4f, expected ~1", (double)out[0]);
        CHECK(favour_last != out[0], "ape had no effect");
        printf("  ok    ape steers the pool: last -> %.3f, first -> %.3f\n",
               (double)favour_last, (double)out[0]);
    }

    printf("\n-- GATE 5  overlap mode pools 2*ratio slots of width d --\n");
    {
        /* ratio 4 selects the overlap path. wkv is 2*d wide there: the first d
         * channels feed the overlapping window, the second d the current one. */
        const int r = 4, d = 4, width = 2 * 4, nslot = 2 * 4;
        static float ks[8 * 8], ss[8 * 8], out[4];
        static float ape[4 * 8];
        memset(ape, 0, sizeof ape);
        memset(ks, 0, sizeof ks);
        for (int i = 0; i < nslot * width; i++) ss[i] = -INFINITY;

        /* First window: values 1..4, uniform gates. */
        int emitted = 0;
        for (int pos = 0; pos < r; pos++) {
            float kv[8], sc[8];
            for (int c = 0; c < width; c++) { kv[c] = (float)(pos + 1); sc[c] = 0.0f; }
            emitted += dsv4_compress_step(out, kv, sc, ape, ks, ss, pos, r, d);
        }
        CHECK(emitted == 1, "overlap emitted %d times in the first window", emitted);
        /* On the very first window the previous-window slots were never written,
         * so their scores are still -inf and only the 4 current slots pool:
         * mean of 1,2,3,4 = 2.5. */
        CHECK(fabsf(out[0] - 2.5f) < 1e-4f,
              "first overlap window gave %.4f, expected 2.5", (double)out[0]);

        /* Second window: values 10..13. Now the previous window's slots are
         * populated, so the pool spans 8 slots: mean of 1,2,3,4,10,11,12,13. */
        for (int pos = r; pos < 2 * r; pos++) {
            float kv[8], sc[8];
            for (int c = 0; c < width; c++) { kv[c] = (float)(pos + 7); sc[c] = 0.0f; }
            dsv4_compress_step(out, kv, sc, ape, ks, ss, pos, r, d);
        }
        const float want = (1 + 2 + 3 + 4 + 11 + 12 + 13 + 14) / 8.0f;
        CHECK(fabsf(out[0] - want) < 1e-3f,
              "second overlap window gave %.4f, expected %.4f (8 slots)",
              (double)out[0], (double)want);
        printf("  ok    window 1 pooled 4 slots (%.2f), window 2 pooled 8 (%.2f)\n",
               2.5, (double)out[0]);
    }

    printf("\n-- GATE 6  the compressed row is stamped with the FIRST position --\n");
    {
        /* start_pos + 1 - ratio, not start_pos. */
        CHECK(dsv4_compress_rope_pos(3, 4) == 0, "pos 3 ratio 4 -> %d, expected 0",
              dsv4_compress_rope_pos(3, 4));
        CHECK(dsv4_compress_rope_pos(7, 4) == 4, "pos 7 ratio 4 -> %d, expected 4",
              dsv4_compress_rope_pos(7, 4));
        CHECK(dsv4_compress_rope_pos(127, 128) == 0, "pos 127 ratio 128 -> %d, expected 0",
              dsv4_compress_rope_pos(127, 128));
        CHECK(dsv4_compress_rope_pos(255, 128) == 128,
              "pos 255 ratio 128 -> %d, expected 128", dsv4_compress_rope_pos(255, 128));
        printf("  ok    pos 7 ratio 4 stamps position 4, not 7\n");
    }

    printf("\n");
    if (fails) { printf("COMPRESSOR GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("COMPRESSOR GATE PASSED\n");
    return 0;
}
