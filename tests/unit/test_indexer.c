/* SPDX-License-Identifier: Apache-2.0 */
/* test_indexer.c - CSA scoring, top-k, and the offset into the attention cache. */
#include <stdio.h>
#include <string.h>
#include <math.h>

void dsv4_indexer_score(float *, const float *, const float *, const float *,
                        int, int, int);
int  dsv4_topk(int *, const float *, int, int);
void dsv4_indexer_offset(int *, int, int);
int  dsv4_indexer_navail(int, int);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

int main(void)
{
    printf("DeepSeek-V4 CSA indexer gate\n");

    printf("\n-- GATE 1  the ReLU discards negative head votes --\n");
    {
        /* Two heads, two positions. Head 0 likes position 0 and dislikes
         * position 1 by the same magnitude; head 1 is neutral.
         *
         * With the relu:    pos 1 scores 0, because the only opinion of it is
         *                   negative and is discarded.
         * Without the relu: pos 1 scores strictly NEGATIVE, and the ranking is
         *                   the same, so ranking alone cannot detect the bug.
         *                   The VALUE can. */
        enum { H = 2, D = 2, N = 2 };
        float q[H * D]  = { 1.0f, 0.0f,   0.0f, 0.0f };
        float kv[N * D] = { 1.0f, 0.0f,  -1.0f, 0.0f };
        float w[H] = { 1.0f, 1.0f };
        float out[N];
        dsv4_indexer_score(out, q, kv, w, H, D, N);

        CHECK(out[0] > 0.0f, "pos 0 = %.6f, expected positive", (double)out[0]);
        CHECK(out[1] == 0.0f,
              "pos 1 = %.6f, expected exactly 0; a negative value means the "
              "ReLU is missing", (double)out[1]);
        printf("  ok    pos 0 = %.4f, pos 1 = %.4f (clamped, not negative)\n",
               (double)out[0], (double)out[1]);
    }

    printf("\n-- GATE 2  the scale is d^-0.5 * h^-0.5, not d^-0.5 alone --\n");
    {
        /* One head, dot product exactly 1, weight exactly 1. The output is then
         * the scale itself, so it can be read off directly. */
        enum { H = 4, D = 16, N = 1 };
        float q[H * D], kv[N * D], w[H], out[N];
        memset(q, 0, sizeof q); memset(kv, 0, sizeof kv);
        for (int h = 0; h < H; h++) w[h] = 0.0f;
        w[0] = 1.0f;
        q[0] = 1.0f;                     /* head 0 only */
        kv[0] = 1.0f;                    /* dot = 1 */
        dsv4_indexer_score(out, q, kv, w, H, D, N);

        const float want  = (1.0f / sqrtf((float)D)) * (1.0f / sqrtf((float)H));
        const float wrong = 1.0f / sqrtf((float)D);
        CHECK(fabsf(out[0] - want) < 1e-6f,
              "score = %.6f, expected %.6f (d^-0.5 * h^-0.5)",
              (double)out[0], (double)want);
        CHECK(fabsf(want - wrong) > 1e-3f,
              "the two scales are indistinguishable here; test is too weak");
        printf("  ok    scale %.5f used, not %.5f\n", (double)want, (double)wrong);
    }

    printf("\n-- GATE 3  head weights are applied per head --\n");
    {
        /* Two heads pointing at different positions, with different weights.
         * A single shared weight cannot produce an asymmetric result. */
        enum { H = 2, D = 2, N = 2 };
        float q[H * D]  = { 1.0f, 0.0f,   0.0f, 1.0f };
        float kv[N * D] = { 1.0f, 0.0f,   0.0f, 1.0f };
        float w[H] = { 1.0f, 4.0f };
        float out[N];
        dsv4_indexer_score(out, q, kv, w, H, D, N);
        CHECK(fabsf(out[1] / out[0] - 4.0f) < 1e-4f,
              "ratio %.4f, expected 4 (head 1 has 4x the weight)",
              (double)(out[1] / out[0]));
        printf("  ok    weights 1 and 4 give scores %.4f and %.4f\n",
               (double)out[0], (double)out[1]);
    }

    printf("\n-- GATE 4  top-k is descending with ties to the lower index --\n");
    {
        float s[6] = { 1.0f, 5.0f, 5.0f, 2.0f, 0.0f, 5.0f };
        int idx[6];
        int n = dsv4_topk(idx, s, 6, 4);
        CHECK(n == 4, "returned %d, expected 4", n);
        /* The three 5.0s must come out in index order, then 2.0. */
        CHECK(idx[0] == 1 && idx[1] == 2 && idx[2] == 5,
              "ties gave {%d,%d,%d}, expected {1,2,5} in index order",
              idx[0], idx[1], idx[2]);
        CHECK(idx[3] == 3, "fourth is %d, expected 3 (value 2.0)", idx[3]);
        /* k > n must clamp rather than read past the array. */
        int idx2[8];
        CHECK(dsv4_topk(idx2, s, 6, 10) == 6, "k > n did not clamp to n");
        printf("  ok    {1,2,5,3}; ties resolved toward the lower index\n");
    }

    printf("\n-- GATE 5  offset shifts real indices and preserves masks --\n");
    {
        int idx[5] = { 0, 3, -1, 7, -1 };
        dsv4_indexer_offset(idx, 5, 128);
        CHECK(idx[0] == 128 && idx[1] == 131 && idx[3] == 135,
              "shifted to {%d,%d,%d}, expected {128,131,135}",
              idx[0], idx[1], idx[3]);
        CHECK(idx[2] == -1 && idx[4] == -1,
              "masked slots became %d/%d; they must stay -1", idx[2], idx[4]);
        printf("  ok    real indices +128, masked slots still -1\n");
    }

    printf("\n-- GATE 6  availability grows one row per `ratio` positions --\n");
    {
        CHECK(dsv4_indexer_navail(0, 4) == 0, "pos 0 -> %d, expected 0",
              dsv4_indexer_navail(0, 4));
        CHECK(dsv4_indexer_navail(3, 4) == 1, "pos 3 -> %d, expected 1",
              dsv4_indexer_navail(3, 4));
        CHECK(dsv4_indexer_navail(7, 4) == 2, "pos 7 -> %d, expected 2",
              dsv4_indexer_navail(7, 4));
        CHECK(dsv4_indexer_navail(2047, 4) == 512, "pos 2047 -> %d, expected 512",
              dsv4_indexer_navail(2047, 4));
        /* Below index_topk every row is selected, so the indexer is a no-op and
         * cannot change the answer. That is the regime a short prompt sits in. */
        printf("  ok    pos 2047 ratio 4 -> 512 rows, exactly Flash's index_topk\n");
    }

    printf("\n-- GATE 7  a Hadamard on both sides leaves the score alone --\n");
    {
        /* rotate_activation is orthogonal, so applying it to q and kv together
         * preserves the inner product. This is why the f32 path may omit it.
         * Uses the order-4 Hadamard, scaled by 1/sqrt(4) to be orthonormal. */
        enum { H = 1, D = 4, N = 1 };
        /* The dot product MUST be positive. An earlier draft used vectors whose
         * dot was -3, so the ReLU zeroed both sides and the gate compared 0 to
         * 0 -- it passed while proving nothing. */
        float q[D]  = { 1.0f, 2.0f, -1.0f, 0.5f };
        float kv[D] = { 2.0f, 1.0f, -0.5f, 1.0f };   /* dot = 2+2+0.5+0.5 = 5 */
        float w[H] = { 1.0f };
        float a[N], b[N];
        dsv4_indexer_score(a, q, kv, w, H, D, N);
        CHECK(a[0] > 0.1f, "baseline score %.6f is not positive; the ReLU would "
              "make this gate vacuous", (double)a[0]);

        const float Hm[4][4] = {{ 1, 1, 1, 1}, { 1,-1, 1,-1},
                                { 1, 1,-1,-1}, { 1,-1,-1, 1}};
        float qr[D], kr[D];
        const float s = 0.5f;                /* 1/sqrt(4) */
        for (int i = 0; i < D; i++) {
            float aq = 0.0f, ak = 0.0f;
            for (int j = 0; j < D; j++) { aq += Hm[i][j] * q[j]; ak += Hm[i][j] * kv[j]; }
            qr[i] = aq * s; kr[i] = ak * s;
        }
        dsv4_indexer_score(b, qr, kr, w, H, D, N);
        CHECK(fabsf(a[0] - b[0]) < 1e-5f,
              "score changed under an orthogonal rotation: %.6f -> %.6f",
              (double)a[0], (double)b[0]);
        printf("  ok    score %.5f unchanged by the Hadamard on both sides\n",
               (double)a[0]);
    }

    printf("\n");
    if (fails) { printf("INDEXER GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("INDEXER GATE PASSED\n");
    return 0;
}
