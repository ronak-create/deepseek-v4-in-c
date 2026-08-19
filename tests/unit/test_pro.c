/* SPDX-License-Identifier: Apache-2.0 */
/* test_pro.c - can this binary run DeepSeek-V4-Pro from --model alone?
 *
 * Pro's weights are ~865 GB and are not on this machine. That does not stop us
 * checking the claim that "Pro is a config change", because everything the
 * engine decides before the first byte of weight is read comes from config.json:
 * every dimension, every layer's kind, every buffer size, the tensor names the
 * binder will ask for, and the memory the budget planner will hand out.
 *
 * So this gate plans Pro and refuses to accept a plan that is secretly Flash's.
 *
 * WHY THIS IS NOT PARANOIA. Until today the binder validated wo_a's shape as
 * [o_lora*o_groups, hidden]. The correct column count is n_heads*head_dim /
 * o_groups. For Flash those are the same number -- 64*512/8 = 4096 = hidden --
 * so the wrong rule passed on both released checkpoints. For Pro they are
 * 128*512/16 = 4096 against a hidden of 7168, and the load would have failed on
 * the first layer. A Flash-only test suite could not see it. This one can.
 */
#include <stdio.h>
#include <stdlib.h>

#include "dsv4_cfg.h"
#include "dsv4_layer.h"

static int fails = 0;
#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

/* Bytes one routed expert occupies on disk, from config alone. Mirrors
 * expert_geometry() in dsv4_cache.c, which the cache gate checks against the
 * real fixture -- so if these two ever disagree, one of them is wrong. */
static int64_t expert_bytes(const DSV4Cfg *c)
{
    const int64_t inter = c->moe_inter, d = c->hidden;
    /* FP4 packs two values per byte; the scale grid is 1x32 over UNPACKED
     * columns, which is the distinction that makes w2 differ from w1/w3. */
    const int64_t w13 = inter * d / 2, s13 = inter * ((d + 31) / 32);
    const int64_t w2  = d * inter / 2, s2  = d * ((inter + 31) / 32);
    return 2 * (w13 + s13) + (w2 + s2);
}

static void plan(const char *label, const char *path, int want_layers,
                 int want_dense, int want_idx, int want_cmp)
{
    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];

    printf("\n-- %s --\n", label);
    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS, path)) {
        CHECK(0, "%s did not load", path);
        return;
    }

    CHECK(c.n_layers == want_layers, "%d layers, expected %d",
          c.n_layers, want_layers);

    int nd = 0, ni = 0, nc = 0;
    for (int L = 0; L < c.n_layers; L++) {
        const int r = dsv4_compress_ratio(&c, L);
        if      (r == 0) nd++;
        else if (r == 4) ni++;
        else             nc++;
    }
    CHECK(nd == want_dense && ni == want_idx && nc == want_cmp,
          "layer kinds %d dense / %d indexed / %d compressed, expected %d/%d/%d",
          nd, ni, nc, want_dense, want_idx, want_cmp);

    /* The grouped output projection. This is the shape that was wrong. */
    const int64_t gw = (int64_t)c.n_heads * c.head_dim / c.o_groups;
    CHECK((int64_t)c.n_heads * c.head_dim % c.o_groups == 0,
          "n_heads*head_dim (%lld) is not divisible by o_groups (%d)",
          (long long)c.n_heads * c.head_dim, c.o_groups);
    printf("  wo_a is [%lld, %lld]; hidden is %d %s\n",
           (long long)c.o_lora * c.o_groups, (long long)gw, c.hidden,
           gw == c.hidden ? "(equal -- a coincidence, not a rule)"
                          : "(DIFFERENT: the old hidden-based rule would fail here)");

    /* Every per-token buffer must allocate at this size. maxpos matches what
     * the CLI uses, so this is the real allocation, not a token one. */
    const int maxpos = c.sliding_window + 4096;
    DSV4Scratch s;
    CHECK(dsv4_scratch_init(&s, &c, maxpos) == 0,
          "scratch would not allocate for %s", label);
    CHECK(s.idx_cap >= maxpos, "idx_scores holds %d, needs >= %d",
          s.idx_cap, maxpos);

    /* One state per layer kind present. A ratio-128 layer needs 32x the
     * compressor ape of a ratio-4 one, which is the sizing that used to be
     * planned from whichever layer happened to be first. */
    for (int L = 0; L < c.n_layers; L++) {
        static int seen4 = 0, seen128 = 0, seen0 = 0;
        const int r = dsv4_compress_ratio(&c, L);
        if ((r == 4 && seen4) || (r == 0 && seen0)
            || (r != 0 && r != 4 && seen128)) continue;
        if (r == 4) seen4 = 1; else if (r == 0) seen0 = 1; else seen128 = 1;
        DSV4LayerState st;
        CHECK(dsv4_state_init(&st, &c, L, maxpos) == 0,
              "state would not allocate for layer %d (ratio %d)", L, r);
        dsv4_state_free(&st);
    }
    dsv4_scratch_free(&s);

    /* What the memory planner will be asked for. The cache cannot hit at all
     * below one pass's working set -- measured, not argued -- so this number
     * decides whether the model is usable on a given machine. */
    const int64_t eb = expert_bytes(&c);
    const int64_t ws = (int64_t)c.n_layers * c.topk * eb;
    printf("  one expert %lld B (%.2f MiB); one pass touches %lld experts "
           "= %.2f GB\n",
           (long long)eb, (double)eb / 1048576.0,
           (long long)c.n_layers * c.topk, (double)ws / 1073741824.0);
    printf("  routing: %d hash + %d scored | experts %d top%d | mHC x%d\n",
           c.num_hash_layers, c.n_layers - c.num_hash_layers,
           c.n_experts, c.topk, c.hc_mult);
    CHECK(eb > 0 && ws > 0, "expert sizing overflowed");
}

int main(void)
{
    printf("DeepSeek-V4 Pro-readiness gate\n");
    printf("Plans both released models from config alone. No weights needed:\n"
           "every size the engine commits to is decided before the first byte\n"
           "of a tensor is read.\n");

    /* Counts are of the DECODER only. compress_ratios carries one extra entry
     * per MTP layer, which dsv4_compress_ratio deliberately cannot reach. */
    plan("V4-Flash  (43 layers, the model on this disk)",
         "tests/fixtures/cfg/dsv4_flash_config.json", 43, 2, 21, 20);
    plan("V4-Pro    (61 layers, not downloaded)",
         "tests/fixtures/cfg/dsv4_pro_config.json",   61, 0, 30, 31);

    printf("\n");
    if (fails) { printf("PRO GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("PRO GATE PASSED\n");
    return 0;
}
