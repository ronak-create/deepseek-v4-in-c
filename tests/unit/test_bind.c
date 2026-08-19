/* SPDX-License-Identifier: Apache-2.0 */
/* test_bind.c - the binder must request exactly the tensors a layer has, and
 * refuse anything it cannot account for.
 *
 * The interesting failures are not "missing tensor". They are:
 *   - requesting a tensor that should be ABSENT (an indexer on a ratio-128
 *     layer, gate.bias on a hash-routed layer). That fails loudly here and
 *     would otherwise be a whole attention path bound to the wrong weights.
 *   - binding an FP8 weight without its scale, or with the wrong block grid.
 *     Both produce finite numbers and a different model.
 *   - planning a quantised tensor WIDE, which would hand a kernel a buffer of
 *     uninitialised floats (see the read_f32 bug pinned in test_st.c).
 */
#include <stdio.h>
#include <string.h>
#include "dsv4_bind.h"
#include "dsv4_cfg.h"

static int fails = 0;

#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

static void check_qmat(const char *what, const DSV4QMat *m, int wdt,
                       int rows, int cols, int blk_r, int blk_c)
{
    CHECK(m->w != NULL, "%s: weight not bound", what);
    CHECK(m->wdt == wdt, "%s: wdt %d, expected %d", what, m->wdt, wdt);
    CHECK(m->rows == rows, "%s: rows %d, expected %d", what, m->rows, rows);
    CHECK(m->cols == cols, "%s: cols %d, expected %d", what, m->cols, cols);
    CHECK(m->blk_r == blk_r && m->blk_c == blk_c,
          "%s: block %dx%d, expected %dx%d", what, m->blk_r, m->blk_c, blk_r, blk_c);
    /* A quantised matrix without its scale is the silent-wrong-output case. */
    if (blk_c) CHECK(m->s != NULL, "%s: QUANTISED BUT SCALE NOT BOUND", what);
    else       CHECK(m->s == NULL, "%s: unquantised but a scale was bound", what);
}

int main(void)
{
    DSV4Cfg c;
    int cr[DSV4_MAX_LAYERS];
    DSV4St s;

    printf("DeepSeek-V4 binder gate\n");

    if (!dsv4_cfg_load_file(&c, cr, DSV4_MAX_LAYERS,
                            "tests/fixtures/cfg/dsv4_flash_config.json")) {
        printf("  FAIL  could not load Flash config\n"); return 1;
    }
    if (dsv4_st_open(&s, "tests/fixtures/st") != 0) {
        printf("  FAIL  could not open the fixture\n"); return 1;
    }
    /* The fixture carries a cut-down vocab so tid2eid stays small; the binder
     * derives its expectation from config, so tell it the truth. */
    const DSV4Tensor *tid = dsv4_st_find(&s, "layers.2.ffn.gate.tid2eid");
    if (tid) c.vocab = (int)tid->shape[0];

    printf("\n-- GATE 1  bind a ratio-4 layer (indexed, hash-routed) --\n");
    {
        DSV4LayerBind b;
        if (dsv4_bind_layer(&s, &c, 2, &b) != 0) {
            printf("  FAIL  layer 2 refused\n"); fails++;
        } else {
            printf("  ok    bound %zu bytes\n", b.nbytes);
            CHECK(b.w.compress_ratio == 4, "ratio %d, expected 4", b.w.compress_ratio);
            CHECK(b.w.has_idx,  "layer 2 must have an indexer");
            CHECK(b.w.has_comp, "layer 2 must have a compressor");
            CHECK(b.w.hash_routed, "layer 2 must be hash-routed");

            /* FP8 attention, 128x128 blocks, shapes derived from config. */
            check_qmat("wq_a", &b.w.attn.wq_a, DSV4_WFP8, c.q_lora, c.hidden, 128, 128);
            check_qmat("wq_b", &b.w.attn.wq_b, DSV4_WFP8,
                       c.n_heads * c.head_dim, c.q_lora, 128, 128);
            check_qmat("wo_a", &b.w.attn.wo_a, DSV4_WFP8,
                       c.o_lora * c.o_groups, c.hidden, 128, 128);
            /* BF16 gate: no scale at all. */
            check_qmat("gate", &b.w.moe.gate, DSV4_WBF16, c.n_experts, c.hidden, 0, 0);
            /* The indexer's own narrower compressor. */
            CHECK(b.w.attn.idx.comp.coff == 2,
                  "ratio-4 compressor coff %d, expected 2", b.w.attn.idx.comp.coff);

            /* Routing mode: exactly one of the two, never both. */
            CHECK(b.w.moe.tid2eid != NULL, "hash layer must bind tid2eid");
            CHECK(b.w.moe.bias == NULL, "hash layer must NOT bind gate.bias");

            /* Routed experts are streamed, never resident. */
            CHECK(b.w.moe.shared.w1.w != NULL, "the shared expert must be bound");
            dsv4_bind_free(&b);
        }
    }

    printf("\n-- GATE 2  bind a ratio-128 layer (no indexer, scored) --\n");
    {
        DSV4LayerBind b;
        if (dsv4_bind_layer(&s, &c, 3, &b) != 0) {
            printf("  FAIL  layer 3 refused\n"); fails++;
        } else {
            printf("  ok    bound %zu bytes\n", b.nbytes);
            CHECK(b.w.compress_ratio == 128, "ratio %d, expected 128", b.w.compress_ratio);
            CHECK(!b.w.has_idx, "layer 3 must NOT have an indexer");
            CHECK(b.w.has_comp, "layer 3 must have a compressor");
            CHECK(!b.w.hash_routed, "layer 3 must be scored, not hash-routed");
            CHECK(b.w.attn.idx.wq_b.w == NULL, "no indexer weights may be bound");
            CHECK(b.w.moe.bias != NULL, "scored layer must bind gate.bias");
            CHECK(b.w.moe.tid2eid == NULL, "scored layer must NOT bind tid2eid");
            CHECK(b.w.attn.comp.coff == 1,
                  "ratio-128 compressor coff %d, expected 1", b.w.attn.comp.coff);
            dsv4_bind_free(&b);
        }
    }

    printf("\n-- GATE 3  mHC geometry comes from config, not literals --\n");
    {
        DSV4LayerBind b;
        if (dsv4_bind_layer(&s, &c, 2, &b) == 0) {
            CHECK(b.w.hc_attn.fn != NULL && b.w.hc_ffn.fn != NULL,
                  "both mHC parameter sets must bind");
            CHECK(dsv4_mix_hc(&c) == 24, "mix_hc %d, expected 24 for hc_mult=4",
                  dsv4_mix_hc(&c));
            CHECK(dsv4_hc_dim(&c) == 16384, "hc_dim %d, expected 16384",
                  dsv4_hc_dim(&c));
            printf("  ok    mix_hc=%d hc_dim=%d, both derived\n",
                   dsv4_mix_hc(&c), dsv4_hc_dim(&c));
            dsv4_bind_free(&b);
        } else { printf("  FAIL  layer 2 refused on the second bind\n"); fails++; }
    }

    printf("\n-- GATE 4  a config that disagrees with the checkpoint is REFUSED --\n");
    {
        /* The check that earns its keep. Perturb one config field and the
         * derived element count no longer matches the file. */
        DSV4Cfg bad = c;
        bad.q_lora = c.q_lora * 2;
        DSV4LayerBind b;
        printf("  (expect a diagnostic naming the mismatched tensor)\n");
        if (dsv4_bind_layer(&s, &bad, 2, &b) == 0) {
            printf("  FAIL  accepted a config whose q_lora is wrong\n");
            fails++;
            dsv4_bind_free(&b);
        } else {
            printf("  ok    refused a config that disagrees with the checkpoint\n");
        }
    }

    dsv4_st_close(&s);
    printf("\n");
    if (fails) { printf("BINDER GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("BINDER GATE PASSED\n");
    return 0;
}
