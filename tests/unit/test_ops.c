/* SPDX-License-Identifier: Apache-2.0 */
/* test_ops.c - RMSNorm, clamped SwiGLU, and the router.
 *
 * Each gate is built around the way its kernel can be wrong while still looking
 * right. A test that only checks "output is finite and roughly the right size"
 * would pass on every one of these mistakes.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <stdint.h>

void  dsv4_rmsnorm(float *, const float *, const float *, int, float);
void  dsv4_swiglu(float *, const float *, const float *, int, float);
void  dsv4_route(int *, float *, float *, float *, const float *,
                 const int64_t *, int, int, int, float);

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

static void close_to(const char *what, float got, float want, float tol)
{
    CHECK(fabsf(got - want) <= tol, "%s = %.9g, expected %.9g (tol %g)",
          what, (double)got, (double)want, (double)tol);
}

int main(void)
{
    printf("DeepSeek-V4 elementwise + router gate\n");

    printf("\n-- GATE 1  RMSNorm: eps goes on the MEAN of squares --\n");
    {
        /* x = [3,4,0,0]: mean square = 25/4 = 6.25, rsqrt = 0.4.
         * Exactly representable, so this is a bit-equality check. */
        float x[4] = { 3.0f, 4.0f, 0.0f, 0.0f };
        float w[4] = { 1.0f, 1.0f, 1.0f, 1.0f };
        float y[4];
        dsv4_rmsnorm(y, x, w, 4, 0.0f);
        close_to("y[0]", y[0], 1.2f, 0.0f);
        close_to("y[1]", y[1], 1.6f, 0.0f);
        /* Weight is applied AFTER the normalisation, elementwise. */
        float w2[4] = { 2.0f, 0.5f, 1.0f, 1.0f };
        dsv4_rmsnorm(y, x, w2, 4, 0.0f);
        close_to("weighted y[0]", y[0], 2.4f, 0.0f);
        close_to("weighted y[1]", y[1], 0.8f, 0.0f);
        printf("  ok    1.2 / 1.6, and the weight applies after normalising\n");

        /* eps INSIDE the mean vs added to the SUM. At n = 4 the two answers sit
         * within 0.4% of each other, which is not a test -- an earlier draft
         * asserted exactly that and its own guard caught it.
         *
         * The gap opens up at the real hidden size with a variance comparable to
         * eps. n = 4096, one nonzero of 0.064: mean square is exactly 1e-6, so
         * eps doubles it and the answer is ~45. Adding eps to the SUM instead
         * leaves 4.096e-3 dominated by the sum, and the answer is ~1. */
        enum { N = 4096 };
        static float t[N], ones[N], y2[N];
        for (int i = 0; i < N; i++) { t[i] = 0.0f; ones[i] = 1.0f; }
        t[0] = 0.064f;
        dsv4_rmsnorm(y2, t, ones, N, 1e-6f);

        const float mean_sq = 0.064f * 0.064f / (float)N;      /* = 1e-6 */
        const float want    = 0.064f / sqrtf(mean_sq + 1e-6f);  /* ~45.25 */
        const float wrong   = 0.064f / sqrtf(0.064f * 0.064f + 1e-6f); /* ~1.0 */
        close_to("eps-on-mean", y2[0], want, 1e-2f);
        CHECK(fabsf(want - wrong) > 10.0f,
              "cannot distinguish eps-on-mean from eps-on-sum; test is too weak");
        printf("  ok    eps-on-mean (%.2f) vs eps-on-sum (%.2f): a 45x gap\n",
               (double)want, (double)wrong);
    }

    printf("\n-- GATE 2  SwiGLU clamp is ASYMMETRIC --\n");
    {
        /* model.py:
         *     up   = clamp(up, min=-limit, max=+limit)
         *     gate = clamp(gate,           max=+limit)
         * gate has NO lower bound. Clamping it symmetrically is the obvious
         * implementation and is wrong. */
        const float limit = 10.0f;

        /* up below -limit MUST be clamped to -limit. */
        {
            float g[1] = { 1.0f }, u[1] = { -50.0f }, y[1];
            dsv4_swiglu(y, g, u, 1, limit);
            const float silu1 = 1.0f / (1.0f + expf(-1.0f));
            close_to("up clamped below", y[0], silu1 * -limit, 1e-6f);
        }
        /* gate below -limit MUST NOT be clamped. silu(-50) is ~ -9.6e-21;
         * silu(-10) is ~ -4.5e-4. Those differ by 17 orders of magnitude, so a
         * symmetric clamp is unmissable here -- and invisible in any test that
         * only checks the output is small. */
        {
            float g[1] = { -50.0f }, u[1] = { 1.0f }, y[1];
            dsv4_swiglu(y, g, u, 1, limit);
            const float unclamped = -50.0f / (1.0f + expf(50.0f));
            const float if_clamped = -10.0f / (1.0f + expf(10.0f));
            close_to("gate NOT clamped below", y[0], unclamped, 1e-12f);
            CHECK(fabsf(y[0]) < fabsf(if_clamped) * 0.5f,
                  "gate appears clamped from below: got %.3g, symmetric would give %.3g",
                  (double)y[0], (double)if_clamped);
            printf("  ok    silu(-50)=%.3g used, not silu(-10)=%.3g\n",
                   (double)unclamped, (double)if_clamped);
        }
        /* gate above +limit IS clamped. */
        {
            float g[2] = { 50.0f, 10.0f }, u[2] = { 1.0f, 1.0f }, y[2];
            dsv4_swiglu(y, g, u, 2, limit);
            CHECK(y[0] == y[1], "gate above the limit was not clamped: %.9g vs %.9g",
                  (double)y[0], (double)y[1]);
            printf("  ok    gate is clamped from above\n");
        }
        /* limit <= 0 disables clamping entirely. */
        {
            float g[1] = { 50.0f }, u[1] = { 50.0f }, y[1];
            dsv4_swiglu(y, g, u, 1, 0.0f);
            close_to("unclamped", y[0], (50.0f / (1.0f + expf(-50.0f))) * 50.0f, 1e-2f);
            printf("  ok    limit <= 0 disables the clamp\n");
        }
    }

    printf("\n-- GATE 3  router: the bias steers SELECTION ONLY --\n");
    {
        /* The invariant this whole engine inherits from K3. Weights must come
         * from the scores BEFORE the bias was added. */
        enum { NE = 6, TOPK = 2 };
        float scores[NE], orig[NE], wts[TOPK];
        int   idx[TOPK];
        /* Pre-softplus logits. Expert 0 is highest unbiased; the bias promotes
         * expert 5 so that selection changes but the weights must not. */
        float logits[NE]  = { 3.0f, 0.1f, 0.2f, 0.3f, 0.4f, 2.0f };
        float bias[NE]    = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 5.0f };

        memcpy(scores, logits, sizeof logits);
        dsv4_route(idx, wts, scores, orig, bias, NULL, 0, NE, TOPK, 1.0f);

        /* Selection: bias must push 5 to the front. */
        CHECK(idx[0] == 5, "top-1 is expert %d, expected 5 (the bias must select it)",
              idx[0]);
        CHECK(idx[1] == 0, "top-2 is expert %d, expected 0", idx[1]);

        /* Weights: computed from UNBIASED sqrt(softplus(logit)), normalised. */
        const float s0 = sqrtf(log1pf(expf(3.0f)));
        const float s5 = sqrtf(log1pf(expf(2.0f)));
        const float tot = s0 + s5;
        close_to("w[expert 5]", wts[0], s5 / tot, 1e-6f);
        close_to("w[expert 0]", wts[1], s0 / tot, 1e-6f);

        /* The trap: if the weights had been gathered from the BIASED scores,
         * expert 5 would dominate. Assert we are not in that world. */
        const float b5 = s5 + 5.0f, b0 = s0;
        const float biased_w5 = b5 / (b5 + b0);
        CHECK(fabsf(wts[0] - biased_w5) > 0.05f,
              "weight for expert 5 (%.4f) matches the BIASED gather (%.4f); "
              "the bias is leaking into the weights", (double)wts[0], (double)biased_w5);
        printf("  ok    bias selected expert 5, weights stayed unbiased "
               "(%.4f, not %.4f)\n", (double)wts[0], (double)biased_w5);

        double sum = 0.0;
        for (int k = 0; k < TOPK; k++) sum += wts[k];
        close_to("weights sum", (float)sum, 1.0f, 1e-6f);
    }

    printf("\n-- GATE 4  router: hash routing ignores the scores for SELECTION --\n");
    {
        enum { NE = 6, TOPK = 2, VOCAB = 4 };
        float scores[NE], orig[NE], wts[TOPK];
        int   idx[TOPK];
        float logits[NE] = { 9.0f, 0.1f, 0.2f, 0.3f, 0.4f, 0.5f };
        int64_t tid2eid[VOCAB * TOPK] = { 3,4,  1,2,  5,0,  2,3 };

        memcpy(scores, logits, sizeof logits);
        dsv4_route(idx, wts, scores, orig, NULL, tid2eid, 2, NE, TOPK, 1.0f);
        /* Token 2 -> experts {5, 0}, regardless of expert 0 having the top score
         * and regardless of order. */
        CHECK(idx[0] == 5 && idx[1] == 0, "hash route gave {%d,%d}, expected {5,0}",
              idx[0], idx[1]);
        /* Weights still come from the scores. */
        const float s5 = sqrtf(log1pf(expf(0.5f)));
        const float s0 = sqrtf(log1pf(expf(9.0f)));
        close_to("hash w0", wts[0], s5 / (s5 + s0), 1e-6f);
        printf("  ok    token 2 -> {5,0} from the table, weights still from scores\n");
    }

    printf("\n-- GATE 5  softplus threshold keeps large logits finite --\n");
    {
        /* torch's softplus returns x unchanged above 20. A naive log1p(exp(x))
         * overflows f32 near 88, giving +inf, then sqrt(+inf), then a NaN weight
         * after normalisation. */
        enum { NE = 4, TOPK = 2 };
        float scores[NE], orig[NE], wts[TOPK];
        int   idx[TOPK];
        float logits[NE] = { 200.0f, 100.0f, 1.0f, 0.5f };
        memcpy(scores, logits, sizeof logits);
        dsv4_route(idx, wts, scores, orig, NULL, NULL, 0, NE, TOPK, 1.0f);
        for (int k = 0; k < TOPK; k++)
            CHECK(isfinite(wts[k]), "weight %d is not finite (%g)", k, (double)wts[k]);
        /* softplus(200) == 200 exactly above the threshold. */
        close_to("sqrt(softplus(200))", orig[0], sqrtf(200.0f), 1e-4f);
        printf("  ok    logits of 200 stay finite; sqrt(softplus(200)) = %.4f\n",
               (double)orig[0]);
    }

    printf("\n-- GATE 6  route_scale multiplies the normalised weights --\n");
    {
        enum { NE = 4, TOPK = 2 };
        float scores[NE], orig[NE], wts[TOPK];
        int idx[TOPK];
        float logits[NE] = { 2.0f, 1.0f, 0.5f, 0.2f };
        memcpy(scores, logits, sizeof logits);
        dsv4_route(idx, wts, scores, orig, NULL, NULL, 0, NE, TOPK, 2.5f);
        double sum = 0.0;
        for (int k = 0; k < TOPK; k++) sum += wts[k];
        close_to("sum with route_scale 2.5", (float)sum, 2.5f, 1e-5f);
        printf("  ok    weights sum to the route_scale, not to 1\n");
    }

    printf("\n");
    if (fails) { printf("OPS GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("OPS GATE PASSED\n");
    return 0;
}
