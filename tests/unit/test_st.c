/* SPDX-License-Identifier: Apache-2.0 */
/* test_st.c - the safetensors reader must index DeepSeek-V4's real dtypes and
 * geometry, and must REFUSE anything it cannot account for.
 *
 * The fixture is synthetic but its dtypes, ranks and shapes were read out of the
 * released Flash checkpoint (shards 4 and 5), not invented. It covers the two
 * layer kinds that differ structurally:
 *
 *   layers.2   compress_ratio 4   -> has an indexer, hash-routed  (gate.tid2eid)
 *   layers.3   compress_ratio 128 -> no indexer,    scored        (gate.bias)
 *
 * The refusals matter more than the acceptances. This file exists because the
 * port shipped, briefly, with a dsv4_st_read_f32 that returned "n floats read"
 * for an F8_E4M3 tensor while writing none of them -- success reported over
 * uninitialised memory. GATE 4 is that bug, pinned.
 */
#include <stdio.h>
#include <string.h>
#include <math.h>
#include "dsv4_st.h"

static int fails = 0;

#define CHECK(cond, ...) do {                                                  \
    if (!(cond)) { printf("  FAIL  "); printf(__VA_ARGS__); printf("\n");      \
                   fails++; }                                                  \
} while (0)

static const DSV4Tensor *want(const DSV4St *s, const char *name)
{
    const DSV4Tensor *t = dsv4_st_find(s, name);
    if (!t) { printf("  FAIL  %s not indexed\n", name); fails++; }
    return t;
}

static void check_tensor(const DSV4St *s, const char *name, DSV4Dtype dt,
                         int ndim, int64_t d0, int64_t d1)
{
    const DSV4Tensor *t = want(s, name);
    if (!t) return;
    CHECK(t->dtype == dt, "%s dtype %d, expected %d", name, t->dtype, dt);
    CHECK(t->ndim == ndim, "%s ndim %d, expected %d", name, t->ndim, ndim);
    CHECK(t->shape[0] == d0, "%s shape[0] %lld, expected %lld",
          name, (long long)t->shape[0], (long long)d0);
    if (ndim > 1)
        CHECK(t->shape[1] == d1, "%s shape[1] %lld, expected %lld",
              name, (long long)t->shape[1], (long long)d1);
    /* Declared bytes must equal shape product times stored element size, or the
     * reader and the file disagree about what is on disk. */
    CHECK(t->nbytes == dsv4_st_numel(t) * dsv4_st_elemsize(t->dtype),
          "%s nbytes %lld != numel*esz %lld", name, (long long)t->nbytes,
          (long long)(dsv4_st_numel(t) * dsv4_st_elemsize(t->dtype)));
}

int main(void)
{
    DSV4St s;
    printf("DeepSeek-V4 safetensors reader gate\n");

    printf("\n-- GATE 1  open and index --\n");
    if (dsv4_st_open(&s, "tests/fixtures/st") != 0) {
        printf("  FAIL  reader refused the fixture directory\n");
        return 1;
    }
    printf("  ok    %d shard(s), %d tensors\n", s.nshard, s.nt);
    CHECK(s.nt == 38, "indexed %d tensors, expected 38", s.nt);

    printf("\n-- GATE 2  the three storage formats --\n");
    /* BF16, no scale partner. */
    check_tensor(&s, "layers.2.ffn.gate.weight",  DSV4_DT_BF16, 2, 256, 4096);
    /* F8_E4M3 with a 128x128-blocked F8_E8M0 scale: 1024/128=8, 4096/128=32. */
    check_tensor(&s, "layers.2.attn.wq_a.weight", DSV4_DT_F8_E4M3, 2, 1024, 4096);
    check_tensor(&s, "layers.2.attn.wq_a.scale",  DSV4_DT_F8_E8M0, 2, 8, 32);
    /* I8 = packed FP4. The SHAPE IS IN BYTES: [2048,2048] holds a [2048,4096]
     * matrix. Its scale is [2048,128], i.e. one scale per 32 values -- 1x32,
     * NOT the 128x128 that config.json's weight_block_size would suggest. */
    check_tensor(&s, "layers.2.ffn.experts.0.w1.weight", DSV4_DT_I8,      2, 2048, 2048);
    check_tensor(&s, "layers.2.ffn.experts.0.w1.scale",  DSV4_DT_F8_E8M0, 2, 2048, 128);
    /* I64 index table, a dtype K3 never had. */
    check_tensor(&s, "layers.2.ffn.gate.tid2eid", DSV4_DT_I64, 2, 512, 6);

    const DSV4Tensor *w1 = dsv4_st_find(&s, "layers.2.ffn.experts.0.w1.weight");
    const DSV4Tensor *sc = dsv4_st_find(&s, "layers.2.ffn.experts.0.w1.scale");
    if (w1 && sc) {
        CHECK(dsv4_dt_is_packed4(w1->dtype), "w1 must report as packed FP4");
        /* two values per stored byte */
        int64_t values = dsv4_st_numel(w1) * 2;
        int64_t scales = dsv4_st_numel(sc);
        CHECK(values == 2048LL * 4096LL, "w1 holds %lld values, expected %lld",
              (long long)values, 2048LL * 4096LL);
        CHECK(values / scales == 32, "%lld values per scale, expected 32 (1x32 block)",
              (long long)(values / scales));
        printf("  ok    w1: %lld bytes -> %lld FP4 values, %lld scales, %lld per block\n",
               (long long)dsv4_st_numel(w1), (long long)values,
               (long long)scales, (long long)(values / scales));
    }

    printf("\n-- GATE 3  structural layer differences --\n");
    /* The indexer exists on the ratio-4 layer and must NOT exist on ratio-128. */
    CHECK(dsv4_st_find(&s, "layers.2.attn.indexer.wq_b.weight") != NULL,
          "ratio-4 layer must carry an indexer");
    CHECK(dsv4_st_find(&s, "layers.3.attn.indexer.wq_b.weight") == NULL,
          "ratio-128 layer must NOT carry an indexer");
    /* Routing mode is a prefix split: hash layers have tid2eid and no bias. */
    CHECK(dsv4_st_find(&s, "layers.2.ffn.gate.tid2eid") != NULL,
          "hash-routed layer must carry tid2eid");
    CHECK(dsv4_st_find(&s, "layers.2.ffn.gate.bias") == NULL,
          "hash-routed layer must NOT carry gate.bias");
    CHECK(dsv4_st_find(&s, "layers.3.ffn.gate.bias") != NULL,
          "scored layer must carry gate.bias");
    CHECK(dsv4_st_find(&s, "layers.3.ffn.gate.tid2eid") == NULL,
          "scored layer must NOT carry tid2eid");
    printf("  ok    indexer and routing-mode tensors appear exactly where expected\n");

    printf("\n-- GATE 4  read_f32 must REFUSE what it cannot widen --\n");
    /* The pinned bug. A canary that stays untouched proves refusal rather than
     * a silent write of garbage. */
    {
        static float canary[64];
        for (int i = 0; i < 64; i++) canary[i] = -12345.0f;
        const DSV4Tensor *t = dsv4_st_find(&s, "layers.2.attn.wq_a.scale");
        int64_t got = t ? dsv4_st_read_f32(&s, t, canary) : -1;
        CHECK(got == 0, "read_f32 on F8_E8M0 returned %lld, expected 0", (long long)got);
        int touched = 0;
        for (int i = 0; i < 64; i++) if (canary[i] != -12345.0f) touched = 1;
        CHECK(!touched, "read_f32 wrote into the output buffer after refusing");
        printf("  ok    refused F8_E8M0 and left the buffer untouched\n");
    }

    printf("\n-- GATE 5  BF16 widening still works --\n");
    {
        const DSV4Tensor *t = want(&s, "layers.2.attn.q_norm.weight");
        if (t) {
            static float buf[1024];
            int64_t got = dsv4_st_read_f32(&s, t, buf);
            CHECK(got == 1024, "widened %lld floats, expected 1024", (long long)got);
            int finite = 0;
            for (int i = 0; i < 1024; i++) if (isfinite(buf[i])) finite++;
            CHECK(finite == 1024, "%d of 1024 widened floats are finite", finite);
            printf("  ok    widened 1024 BF16 values, all finite\n");
        }
    }
    dsv4_st_close(&s);

    printf("\n-- GATE 6  an unimplemented dtype must be refused outright --\n");
    {
        DSV4St bad;
        if (dsv4_st_open(&bad, "tests/fixtures/st_bad") == 0) {
            printf("  FAIL  accepted a shard whose dtype nothing implements\n");
            fails++;
            dsv4_st_close(&bad);
        } else {
            printf("  ok    correctly refused an unknown dtype\n");
        }
    }

    printf("\n");
    if (fails) { printf("SAFETENSORS GATE FAILED: %d check(s)\n", fails); return 1; }
    printf("SAFETENSORS GATE PASSED\n");
    return 0;
}
