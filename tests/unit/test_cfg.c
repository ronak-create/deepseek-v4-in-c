/* SPDX-License-Identifier: Apache-2.0 */
/* test_cfg.c - the config reader must accept both released models exactly, and
 * refuse every mutation of them.
 *
 * The refusals are the point. Accepting a good config proves very little: a
 * reader that defaulted every field would also "accept" it. What distinguishes a
 * safe reader is that it STOPS on a config it cannot fully account for, because
 * the alternative is a model that loads, streams, decodes, and emits fluent text
 * from the wrong architecture.
 *
 * Each bad fixture is one realistic mistake:
 *   no_compress_ratios     the key a flat-schema reader would miss entirely
 *   short_compress_ratios  a list that parses but does not cover every layer
 *   no_mtp_compress_tail   a list exactly n_layers long -- the mistake this
 *                          test caught in its own first draft
 *   bad_compress_value     a ratio this engine has no path for
 *   bad_scoring_func       sqrtsoftplus -> sigmoid; both route, one is wrong
 *   bad_expert_dtype       fp4 -> fp8; same tensor names, half the bytes
 *   no_hc_mult             mHC silently collapsing to a plain residual stream
 *   bad_model_type         a sibling architecture with colliding field names
 *   no_swiglu_limit        the dangerous one: the default would be CORRECT
 */
#include <stdio.h>
#include <string.h>
#include "dsv4_cfg.h"

static int fails = 0;

#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

/* Expect a load to SUCCEED, then verify the values it produced. */
static void expect_good(const char *path, const char *label,
                        int layers, int hidden, int experts, int index_topk,
                        int n_dense_attn, int n_indexed)
{
    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    printf("\n-- %s --\n", label);
    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS, path)) {
        printf("  FAIL  %s was REFUSED but should load\n", path);
        fails++;
        return;
    }
    CHECK(c.n_layers == layers,      "n_layers %d, expected %d", c.n_layers, layers);
    CHECK(c.hidden   == hidden,      "hidden %d, expected %d", c.hidden, hidden);
    CHECK(c.n_experts == experts,    "n_experts %d, expected %d", c.n_experts, experts);
    CHECK(c.index_topk == index_topk,"index_topk %d, expected %d", c.index_topk, index_topk);
    CHECK(c.topk == 6,               "topk %d, expected 6", c.topk);
    CHECK(c.n_shared == 1,           "n_shared %d, expected 1", c.n_shared);
    CHECK(c.vocab == 129280,         "vocab %d, expected 129280", c.vocab);
    CHECK(c.hc_mult == 4,            "hc_mult %d, expected 4", c.hc_mult);
    CHECK(c.hc_sinkhorn_iters == 20, "sinkhorn iters %d, expected 20", c.hc_sinkhorn_iters);
    CHECK(c.head_dim == 512,         "head_dim %d, expected 512", c.head_dim);
    CHECK(c.n_kv_heads == 1,         "n_kv_heads %d, expected 1", c.n_kv_heads);
    CHECK(c.qk_rope == 64,           "qk_rope %d, expected 64", c.qk_rope);
    CHECK(c.sliding_window == 128,   "sliding_window %d, expected 128", c.sliding_window);
    CHECK(c.index_n_heads == 64,     "index_n_heads %d, expected 64", c.index_n_heads);
    CHECK(c.index_head_dim == 128,   "index_head_dim %d, expected 128", c.index_head_dim);
    CHECK(c.scoring_func == DSV4_SCORE_SQRTSOFTPLUS, "scoring_func not sqrtsoftplus");
    CHECK(c.topk_method  == DSV4_TOPK_NOAUX_TC,      "topk_method not noaux_tc");
    CHECK(c.norm_topk == 1,          "norm_topk %d, expected 1", c.norm_topk);
    CHECK(c.swiglu_limit == 10.0f,   "swiglu_limit %g, expected 10", (double)c.swiglu_limit);
    CHECK(c.wblock_m == 128 && c.wblock_n == 128, "weight_block_size not 128x128");
    CHECK(c.yarn_factor == 16.0f,    "yarn factor %g, expected 16", (double)c.yarn_factor);
    CHECK(c.yarn_orig_ctx == 65536,  "yarn orig ctx %d, expected 65536", c.yarn_orig_ctx);

    /* Invariant 1: after the reader drops the MTP tail, the layer map covers
     * exactly the decoder, and the ZEROS ARE LOAD-BEARING. */
    CHECK(c.n_compress == c.n_layers, "n_compress %d != n_layers %d",
          c.n_compress, c.n_layers);
    /* Nothing may address the MTP block as though it were a decoder layer. */
    CHECK(dsv4_compress_ratio(&c, c.n_layers) == -1,
          "layer n_layers is addressable; the MTP entry leaked into the map");

    int dense = 0, idx = 0;
    for (int i = 0; i < c.n_compress; i++) {
        if (dsv4_is_dense_attn(&c, i)) dense++;
        if (dsv4_has_indexer(&c, i))   idx++;
    }
    CHECK(dense == n_dense_attn, "%d dense-attention layers, expected %d",
          dense, n_dense_attn);
    CHECK(idx == n_indexed, "%d indexed layers, expected %d", idx, n_indexed);

    /* Invariant 4: routing mode is a prefix split at num_hash_layers. Verified
     * in the released checkpoint -- layer 2 carries ffn.gate.tid2eid and no
     * bias, layer 3 carries ffn.gate.bias and no tid2eid. */
    CHECK(c.num_hash_layers == 3, "num_hash_layers %d, expected 3", c.num_hash_layers);
    CHECK(dsv4_is_hash_routed(&c, 0) && dsv4_is_hash_routed(&c, 2),
          "layers 0 and 2 must be hash-routed");
    CHECK(!dsv4_is_hash_routed(&c, 3), "layer 3 must be scored, not hash-routed");
    CHECK(!dsv4_is_hash_routed(&c, c.n_layers - 1), "last layer must be scored");
    CHECK(!dsv4_is_hash_routed(&c, -1) && !dsv4_is_hash_routed(&c, c.n_layers),
          "out-of-range layers must not report hash routing");
    /* Invariant 2: the indexer runs where and only where the ratio is 4. */
    CHECK(idx > 0, "no layer has an indexer; CSA would never run");
    CHECK(dense + idx < c.n_layers, "no compressed-128 layers; HCA would never run");

    printf("  ok    %d layers: %d dense, %d indexed, %d compressed-128\n",
           c.n_layers, dense, idx, c.n_layers - dense - idx);
}

/* Expect a load to be REFUSED. Anything else is the failure this file exists
 * to catch. */
static void expect_refused(const char *path, const char *why)
{
    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    printf("\n-- refusing: %s --\n", why);
    if (dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS, path)) {
        printf("  FAIL  %s was ACCEPTED; it must be refused (%s)\n", path, why);
        fails++;
    } else {
        printf("  ok    correctly refused %s\n", path);
    }
}

int main(void)
{
    printf("DeepSeek-V4 config reader gate\n");

    /* Counts measured from the released files, AFTER dropping the trailing MTP
     * entry. Flash zeroes layers 0 and 1 only. Pro has NO dense layer at all --
     * its trailing 0 belongs to the MTP block, not to layer 60. */
    expect_good("tests/fixtures/cfg/dsv4_flash_config.json", "DeepSeek-V4-Flash",
                43, 4096, 256, 512, /*dense*/ 2, /*indexed*/ 21);
    expect_good("tests/fixtures/cfg/dsv4_pro_config.json",   "DeepSeek-V4-Pro",
                61, 7168, 384, 1024, /*dense*/ 0, /*indexed*/ 30);

    expect_refused("tests/fixtures/cfg/no_compress_ratios.json",
                   "compress_ratios absent (every layer would run dense)");
    expect_refused("tests/fixtures/cfg/short_compress_ratios.json",
                   "compress_ratios shorter than n_layers");
    expect_refused("tests/fixtures/cfg/no_mtp_compress_tail.json",
                   "compress_ratios exactly n_layers long, missing the MTP entry "
                   "the released files carry");
    expect_refused("tests/fixtures/cfg/bad_compress_value.json",
                   "a compress ratio this engine has no path for");
    expect_refused("tests/fixtures/cfg/bad_scoring_func.json",
                   "scoring_func sigmoid instead of sqrtsoftplus");
    expect_refused("tests/fixtures/cfg/bad_expert_dtype.json",
                   "expert_dtype fp8 instead of fp4");
    expect_refused("tests/fixtures/cfg/no_hc_mult.json",
                   "hc_mult absent (mHC would collapse to one stream)");
    expect_refused("tests/fixtures/cfg/hash_layers_all.json",
                   "num_hash_layers == n_layers, leaving no scored layer and so "
                   "no ffn.gate.bias anywhere");
    expect_refused("tests/fixtures/cfg/bad_model_type.json",
                   "a different architecture with colliding field names");
    expect_refused("tests/fixtures/cfg/no_swiglu_limit.json",
                   "swiglu_limit absent (10.0 is the CORRECT default, which is "
                   "exactly why defaulting it is dangerous)");

    printf("\n");
    if (fails) { printf("CONFIG GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("CONFIG GATE PASSED\n");
    return 0;
}
