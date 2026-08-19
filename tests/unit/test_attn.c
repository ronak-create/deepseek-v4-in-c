/* SPDX-License-Identifier: Apache-2.0 */
/* test_attn.c - sparse attention with sinks, and the sliding-window index list. */
#include <stdio.h>
#include <string.h>
#include <math.h>

void dsv4_sparse_attn(float *, const float *, const float *, const float *,
                      const int *, int, int, int, float, float *);
int  dsv4_window_idxs(int *, int, int);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

int main(void)
{
    printf("DeepSeek-V4 sparse attention gate\n");

    printf("\n-- GATE 1  uniform scores give the mean of the values --\n");
    {
        /* q = 0 makes every score 0, so with a very negative sink the softmax is
         * uniform over the gathered positions and the output is their mean. */
        enum { H = 2, D = 4, N = 4, TOPK = 4 };
        float q[H * D] = {0}, kv[N * D], o[H * D], scratch[TOPK];
        float sink[H] = { -1000.0f, -1000.0f };
        int idxs[TOPK] = { 0, 1, 2, 3 };
        for (int n = 0; n < N; n++)
            for (int c = 0; c < D; c++) kv[n * D + c] = (float)(n + 1);

        dsv4_sparse_attn(o, q, kv, sink, idxs, H, D, TOPK, 1.0f, scratch);
        for (int i = 0; i < H; i++)
            for (int c = 0; c < D; c++)
                CHECK(fabsf(o[i * D + c] - 2.5f) < 1e-5f,
                      "o[%d][%d] = %.6f, expected 2.5", i, c, (double)o[i * D + c]);
        printf("  ok    mean of 1,2,3,4 = %.4f\n", (double)o[0]);
    }

    printf("\n-- GATE 2  the sink enters ONLY the denominator --\n");
    {
        /* One position, score 0, sink 0. The denominator is exp(0)+exp(0) = 2
         * while the numerator is exp(0)*kv = kv, so the output is HALF the
         * value. If the sink also entered the numerator the output would be the
         * full value, and nothing would look wrong. */
        enum { H = 1, D = 2, TOPK = 1 };
        float q[H * D] = { 0.0f, 0.0f };
        float kv[D] = { 8.0f, 8.0f };
        float sink[H] = { 0.0f };
        float o[H * D], scratch[TOPK];
        int idxs[TOPK] = { 0 };

        dsv4_sparse_attn(o, q, kv, sink, idxs, H, D, TOPK, 1.0f, scratch);
        CHECK(fabsf(o[0] - 4.0f) < 1e-5f,
              "o = %.6f, expected 4 (half of 8: the sink halves the weight)",
              (double)o[0]);
        CHECK(fabsf(o[0] - 8.0f) > 1e-3f,
              "o equals the full value; the sink is not in the denominator");
        printf("  ok    value 8 attenuated to %.4f by an equal-weight sink\n",
               (double)o[0]);
    }

    printf("\n-- GATE 3  a large sink drives the output toward zero --\n");
    {
        enum { H = 1, D = 2, TOPK = 1 };
        float q[H * D] = { 0.0f, 0.0f };
        float kv[D] = { 8.0f, 8.0f };
        float o[H * D], scratch[TOPK];
        int idxs[TOPK] = { 0 };
        float small[H] = { -20.0f }, large[H] = { 20.0f };

        dsv4_sparse_attn(o, q, kv, small, idxs, H, D, TOPK, 1.0f, scratch);
        const float with_small = o[0];
        dsv4_sparse_attn(o, q, kv, large, idxs, H, D, TOPK, 1.0f, scratch);
        const float with_large = o[0];
        CHECK(fabsf(with_small - 8.0f) < 1e-3f,
              "a very negative sink should leave the value intact, got %.6f",
              (double)with_small);
        CHECK(with_large < 0.001f,
              "a large sink should absorb nearly all the mass, got %.6f",
              (double)with_large);
        printf("  ok    sink -20 -> %.4f, sink +20 -> %.2e\n",
               (double)with_small, (double)with_large);
    }

    printf("\n-- GATE 4  the sink is PER HEAD --\n");
    {
        enum { H = 2, D = 2, TOPK = 1 };
        float q[H * D] = {0};
        float kv[D] = { 8.0f, 8.0f };
        float sink[H] = { -20.0f, 20.0f };
        float o[H * D], scratch[TOPK];
        int idxs[TOPK] = { 0 };
        dsv4_sparse_attn(o, q, kv, sink, idxs, H, D, TOPK, 1.0f, scratch);
        CHECK(fabsf(o[0] - 8.0f) < 1e-3f, "head 0 = %.6f, expected ~8", (double)o[0]);
        CHECK(o[D] < 0.001f, "head 1 = %.6f, expected ~0", (double)o[D]);
        printf("  ok    head 0 -> %.4f, head 1 -> %.2e from the same kv\n",
               (double)o[0], (double)o[D]);
    }

    printf("\n-- GATE 5  masked (-1) slots contribute nothing --\n");
    {
        /* Four slots, two masked. The answer must equal the mean of the two
         * live values, not of all four, and must not read kv[-1]. */
        enum { H = 1, D = 2, N = 4, TOPK = 4 };
        float q[H * D] = {0}, kv[N * D], o[H * D], scratch[TOPK];
        float sink[H] = { -1000.0f };
        int idxs[TOPK] = { 0, -1, 2, -1 };
        for (int n = 0; n < N; n++)
            for (int c = 0; c < D; c++) kv[n * D + c] = (float)(n + 1);

        dsv4_sparse_attn(o, q, kv, sink, idxs, H, D, TOPK, 1.0f, scratch);
        /* live values are kv[0] = 1 and kv[2] = 3 -> mean 2 */
        CHECK(fabsf(o[0] - 2.0f) < 1e-5f, "o = %.6f, expected 2 (mean of 1 and 3)",
              (double)o[0]);
        printf("  ok    masked slots ignored: mean of 1 and 3 = %.4f\n", (double)o[0]);
    }

    printf("\n-- GATE 6  scores actually select --\n");
    {
        /* q aligned with kv[1]: that position should dominate. */
        enum { H = 1, D = 2, N = 3, TOPK = 3 };
        float q[H * D] = { 0.0f, 10.0f };
        float kv[N * D] = { 1.0f, 0.0f,   0.0f, 1.0f,   -1.0f, 0.0f };
        float o[H * D], scratch[TOPK];
        float sink[H] = { -1000.0f };
        int idxs[TOPK] = { 0, 1, 2 };
        dsv4_sparse_attn(o, q, kv, sink, idxs, H, D, TOPK, 1.0f, scratch);
        CHECK(o[1] > 0.99f, "o[1] = %.6f, expected ~1 (kv[1] should dominate)",
              (double)o[1]);
        CHECK(fabsf(o[0]) < 0.01f, "o[0] = %.6f, expected ~0", (double)o[0]);
        printf("  ok    aligned position dominates: o = (%.4f, %.4f)\n",
               (double)o[0], (double)o[1]);
    }

    printf("\n-- GATE 7  sliding-window index list --\n");
    {
        const int win = 4;
        int idxs[8];
        /* Early: fewer valid entries than the window. */
        int n = dsv4_window_idxs(idxs, 0, win);
        CHECK(n == 1, "pos 0 gave %d valid entries, expected 1", n);
        CHECK(idxs[0] == 0, "pos 0 slot %d, expected 0", idxs[0]);
        for (int k = 1; k < win; k++)
            CHECK(idxs[k] == -1, "slot %d should be masked at pos 0", k);

        n = dsv4_window_idxs(idxs, 2, win);
        CHECK(n == 3, "pos 2 gave %d, expected 3", n);
        CHECK(idxs[0] == 2 && idxs[1] == 1 && idxs[2] == 0,
              "pos 2 slots {%d,%d,%d}, expected {2,1,0}", idxs[0], idxs[1], idxs[2]);
        CHECK(idxs[3] == -1, "pos 2 slot 3 should still be masked");

        /* Saturated, and wrapping: the ring holds positions 5,4,3,2. */
        n = dsv4_window_idxs(idxs, 5, win);
        CHECK(n == win, "pos 5 gave %d, expected %d", n, win);
        CHECK(idxs[0] == 1 && idxs[1] == 0 && idxs[2] == 3 && idxs[3] == 2,
              "pos 5 slots {%d,%d,%d,%d}, expected {1,0,3,2}",
              idxs[0], idxs[1], idxs[2], idxs[3]);
        /* Every slot must be distinct once saturated, or a position is counted
         * twice and another is lost. */
        for (int a = 0; a < win; a++)
            for (int b2 = a + 1; b2 < win; b2++)
                CHECK(idxs[a] != idxs[b2], "slots %d and %d collide (%d)",
                      a, b2, idxs[a]);
        printf("  ok    pos 5 -> ring slots {1,0,3,2}, all distinct\n");
    }

    printf("\n");
    if (fails) { printf("ATTENTION GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("ATTENTION GATE PASSED\n");
    return 0;
}
