#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Emit a whole-decoder-block fixture: random bf16 weights, one token, and the
reference output.

WHY A WHOLE BLOCK AND NOT MORE KERNELS
    Per-kernel agreement is necessary and not sufficient. Every kernel can match
    PyTorch to 5e-7 while the block still computes the wrong thing, because the
    mistakes that remain live BETWEEN kernels: a residual captured at the wrong
    point, a normalisation applied to the wrong tensor, the q latent handed to
    the indexer instead of to wq_b, heads mixed across output groups. None of
    those are visible from inside a kernel.

WHY bfloat16 WEIGHTS
    So the two implementations see byte-identical numbers. The fixture carries
    raw bf16 bit patterns as integers; the C points at them directly and the
    reference widens the same bits. Emitting float32 and letting the C read bf16
    would compare two different models and call the difference "tolerance".

SCOPE
    compress_ratio 0 (dense attention: no compressor, no indexer) and scored
    routing. That covers mHC twice, the full attention path including the double
    q-normalisation and the grouped output projection, the router, the routed
    experts and the shared expert. The ratio-4 and ratio-128 paths need the
    compressor and are a separate fixture.
"""
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv4_ref as R  # noqa: E402

torch.manual_seed(31337)

CFG = dict(hidden=32, hc_mult=2, n_heads=4, head_dim=8, qk_rope=4,
           q_lora=16, o_lora=4, o_groups=2, window=4,
           n_experts=8, topk=3, moe_inter=12, n_shared=1,
           route_scale=1.5, swiglu_limit=10.0,
           rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20)


def bf16(*shape, scale=0.05):
    """A bf16 tensor plus its raw uint16 bit patterns."""
    t = (torch.randn(*shape) * scale).bfloat16()
    bits = t.view(torch.uint16).flatten().tolist()
    return t, [int(b) for b in bits]


def f32(*shape, scale=1.0):
    t = torch.randn(*shape) * scale
    return t, [float(v) for v in t.flatten().tolist()]


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1
                       else "tests/fixtures/ref")
    out.mkdir(parents=True, exist_ok=True)

    c = CFG
    H, hd, d = c["n_heads"], c["head_dim"], c["hidden"]
    hc, mix = c["hc_mult"], (2 + c["hc_mult"]) * c["hc_mult"]
    W, J = {}, dict(cfg=c)

    def put_bf16(key, *shape, scale=0.05):
        t, bits = bf16(*shape, scale=scale)
        W[key] = t
        J[key] = bits

    def put_f32(key, *shape, scale=1.0):
        t, vals = f32(*shape, scale=scale)
        W[key] = t
        J[key] = vals

    # norms and small vectors are f32 in the engine (widened at bind time)
    put_f32("attn_norm", d, scale=0.2); W["attn_norm"] += 1.0
    J["attn_norm"] = [float(v) for v in W["attn_norm"].tolist()]
    put_f32("ffn_norm", d, scale=0.2); W["ffn_norm"] += 1.0
    J["ffn_norm"] = [float(v) for v in W["ffn_norm"].tolist()]
    put_f32("q_norm", c["q_lora"], scale=0.2); W["q_norm"] += 1.0
    J["q_norm"] = [float(v) for v in W["q_norm"].tolist()]
    put_f32("kv_norm", hd, scale=0.2); W["kv_norm"] += 1.0
    J["kv_norm"] = [float(v) for v in W["kv_norm"].tolist()]
    put_f32("sink", H, scale=1.5)

    # attention projections, bf16
    put_bf16("wq_a", c["q_lora"], d)
    put_bf16("wq_b", H * hd, c["q_lora"])
    put_bf16("wkv", hd, d)
    put_bf16("wo_a", c["o_groups"] * c["o_lora"], H * hd // c["o_groups"])
    put_bf16("wo_b", d, c["o_groups"] * c["o_lora"])

    # mHC
    put_f32("hc_attn_fn", mix, hc * d, scale=0.05)
    put_f32("hc_ffn_fn", mix, hc * d, scale=0.05)
    put_f32("hc_attn_base", mix, scale=0.3)
    put_f32("hc_ffn_base", mix, scale=0.3)
    put_f32("hc_attn_scale", 3, scale=0.2); W["hc_attn_scale"] += 1.0
    J["hc_attn_scale"] = [float(v) for v in W["hc_attn_scale"].tolist()]
    put_f32("hc_ffn_scale", 3, scale=0.2); W["hc_ffn_scale"] += 1.0
    J["hc_ffn_scale"] = [float(v) for v in W["hc_ffn_scale"].tolist()]

    # MoE: a bias big enough to REORDER top-k, so a weights-from-biased-scores
    # bug changes the block output rather than hiding inside the router.
    put_bf16("gate", c["n_experts"], d, scale=0.3)
    b = torch.zeros(c["n_experts"])
    b[torch.randperm(c["n_experts"])[:3]] = 3.0
    W["gate_bias"] = b
    J["gate_bias"] = [float(v) for v in b.tolist()]

    for e in range(c["n_experts"]):
        put_bf16(f"e{e}_w1", c["moe_inter"], d)
        put_bf16(f"e{e}_w3", c["moe_inter"], d)
        put_bf16(f"e{e}_w2", d, c["moe_inter"])
    put_bf16("sh_w1", c["moe_inter"], d)
    put_bf16("sh_w3", c["moe_inter"], d)
    put_bf16("sh_w2", d, c["moe_inter"])

    # Run several positions so the KV ring wraps and the block is exercised as a
    # sequence, not just once: a bug in the ring only shows after `window` steps.
    fc = R.freqs_cis(c["qk_rope"], 64, 0, 10000.0, 16.0, 32.0, 1.0)
    kv_cache = torch.zeros(c["window"], hd)
    h = torch.randn(hc, d) * 0.5
    J["h_in"] = [float(v) for v in h.flatten().tolist()]

    steps, token_id = 6, 3
    # long enough that several windows CLOSE and the overlap slide runs
    steps_c = 20
    J["steps"] = steps
    J["token_id"] = token_id
    outs = []
    for pos in range(steps):
        h = R.block_dense(h, W, c, kv_cache, fc, pos, token_id)
        outs.append([float(v) for v in h.flatten().tolist()])
    J["h_out"] = outs

    p = out / "layer_dense.json"
    p.write_text(json.dumps(J) + "\n")
    print(f"  {p}  ({steps} steps, hidden={d}, hc={hc}, "
          f"{c['n_experts']} experts top{c['topk']})")

    # ---- the compressed variants -------------------------------------------
    # Identical weights plus a compressor, so any difference between these and
    # layer_dense is the compressor and not a different model.
    #
    # ratio 8 stands in for the released 128: the code branches on `ratio == 4`
    # for the overlap path and treats every other value identically, so 8
    # exercises the same branch at a size a fixture can hold. ratio 4 then adds
    # the overlap splice and the window slide.
    for tag, ratio in (("c8", 8), ("c4", 4)):
        cc = dict(c)
        cc["ratio"] = ratio
        coff = 2 if ratio == 4 else 1
        W2, J2 = dict(W), dict(J)
        J2["cfg"] = cc

        for key, shape, isbf, sc in (
                ("c_wkv",   (coff * hd, d),        True,  0.05),
                ("c_wgate", (coff * hd, d),        True,  0.05),
                ("c_ape",   (ratio, coff * hd),    False, 0.5),
                ("c_norm",  (hd,),                 False, 0.2)):
            if isbf:
                t, v = bf16(*shape, scale=sc)
            else:
                t, v = f32(*shape, scale=sc)
            W2[key], J2[key] = t, v
        W2["c_norm"] = W2["c_norm"] + 1.0
        J2["c_norm"] = [float(v) for v in W2["c_norm"].tolist()]

        st = R.CompressorState(ratio, hd)
        cache = torch.zeros(cc["window"] + steps_c // ratio + 2, hd)
        n_comp = [0]
        h2 = torch.tensor(J["h_in"]).reshape(hc, d).clone()
        outs2 = []
        for pos in range(steps_c):
            h2 = R.block_compressed(h2, W2, cc, cache, st, fc, pos,
                                    token_id, n_comp)
            outs2.append([float(v) for v in h2.flatten().tolist()])
        J2["steps"] = steps_c
        J2["h_out"] = outs2
        J2["n_compressed"] = n_comp[0]

        q = out / f"layer_{tag}.json"
        q.write_text(json.dumps(J2) + "\n")
        print(f"  {q}  (ratio {ratio}, {steps_c} steps, "
              f"{n_comp[0]} compressed rows)")


if __name__ == "__main__":
    main()
