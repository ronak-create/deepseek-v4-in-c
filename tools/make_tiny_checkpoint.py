#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build a tiny but STRUCTURALLY REAL deepseek_v4 checkpoint, and the logits the
reference produces from it.

WHY THIS EXISTS
    Per-kernel and whole-block gates both pass, and neither can catch a mistake
    that lives between LAYERS or in the model-level path: the embedding gather,
    the hc expansion to hc_mult copies, the head's own hc reduction, the final
    norm, the lm_head. Those run once per token and have never been compared
    against anything.

    So this writes a complete checkpoint the engine loads exactly as it loads
    the released one -- same tensor names, same dtypes, same config schema, a
    packable trunk -- and records the reference logits for a few positions.

WEIGHTS ARE SAMPLED FROM EXACTLY-REPRESENTABLE VALUES
    An FP8 or FP4 tensor cannot hold an arbitrary float, so a naive generator
    would have to quantise, and then the fixture would encode MY rounding rather
    than the format's. Instead every value is drawn from the set the format
    represents exactly -- all 256 e4m3 codes, all 16 e2m1 codes -- so the byte
    IS the value and the round trip is lossless by construction. Any
    disagreement the gate then reports is a real one.
"""
import json
import math
import os
import pathlib
import struct
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv4_ref as R  # noqa: E402

torch.manual_seed(4242)

CFG = dict(hidden=32, n_layers=4, vocab=64, hc_mult=2,
           n_heads=4, head_dim=8, qk_rope=4, q_lora=16,
           o_lora=4, o_groups=2, window=4,
           # moe_inter MUST be a multiple of 32: routed experts are FP4 with a
           # 1x32 scale block, so w2 [hidden, moe_inter] needs moe_inter/32
           # whole scale columns. At 12 that grid rounds to zero columns and the
           # cache rejects the tensor. The real model uses 2048.
           n_experts=8, topk=3, moe_inter=32, n_shared=1,
           route_scale=1.5, swiglu_limit=10.0,
           rms_eps=1e-6, hc_eps=1e-6, sinkhorn_iters=20,
           num_hash_layers=1,
           compress_ratios=[0, 8, 4, 8, 0])   # 4 layers + 1 MTP tail


# ---- the exact value sets of each format ---------------------------------
def e4m3_table():
    """All 256 e4m3 codes as floats. Mirrors dsv4_quant.h, written from the bit
    layout rather than copied from the C."""
    out = []
    for b in range(256):
        s = -1.0 if (b & 0x80) else 1.0
        e, m = (b >> 3) & 0xF, b & 7
        if e == 0:
            v = m * 2.0 ** -9
        elif e == 15 and m == 7:
            v = float("nan")
        else:
            v = (1.0 + m / 8.0) * 2.0 ** (e - 7)
        out.append(s * v)
    return out


E4M3 = e4m3_table()
E2M1 = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0]

# codes whose magnitude is small enough to keep activations sane
E4M3_SMALL = [i for i, v in enumerate(E4M3)
              if not math.isnan(v) and 0.0 < abs(v) <= 0.5]


def fp8_tensor(rows, cols, blk=128):
    """Returns (bytes, scale_bytes, dequantised f32 tensor)."""
    idx = torch.randint(0, len(E4M3_SMALL), (rows, cols))
    codes = torch.tensor(E4M3_SMALL, dtype=torch.int32)[idx]
    vals = torch.tensor([E4M3[c] for c in codes.flatten().tolist()]).reshape(rows, cols)
    sr, sc = max(1, (rows + blk - 1) // blk), max(1, (cols + blk - 1) // blk)
    sbytes = torch.full((sr, sc), 127, dtype=torch.uint8)          # scale 2^0
    return (codes.to(torch.uint8), sbytes, vals.float())


def fp4_tensor(rows, cols, blk=32):
    """Packed two per byte, LOW nibble first (torch Float4_e2m1fn_x2)."""
    idx = torch.randint(0, 8, (rows, cols))          # positive codes only
    sign = torch.randint(0, 2, (rows, cols)) * 8
    codes = (idx + sign).to(torch.int32)
    vals = torch.tensor([E2M1[c] for c in codes.flatten().tolist()]).reshape(rows, cols)
    lo, hi = codes[:, 0::2], codes[:, 1::2]
    packed = (lo | (hi << 4)).to(torch.uint8)
    sr, sc = rows, max(1, cols // blk)
    sbytes = torch.full((sr, sc), 127, dtype=torch.uint8)
    return (packed, sbytes, vals.float())


def bf16_tensor(*shape, scale=0.1):
    t = (torch.randn(*shape) * scale).bfloat16()
    return t, t.float()


# ---- safetensors writing --------------------------------------------------
DT = {torch.uint8: "U8", torch.bfloat16: "BF16", torch.float32: "F32",
      torch.int64: "I64"}


def write_safetensors(path, tensors):
    """tensors: name -> (torch tensor, dtype string override or None)"""
    hdr, blob, off = {}, bytearray(), 0
    for name, (t, dt) in tensors.items():
        b = t.contiguous().view(torch.uint8).numpy().tobytes() \
            if t.dtype != torch.uint8 else t.contiguous().numpy().tobytes()
        hdr[name] = {"dtype": dt or DT[t.dtype],
                     "shape": list(t.shape),
                     "data_offsets": [off, off + len(b)]}
        blob += b
        off += len(b)
    j = json.dumps(hdr, separators=(",", ":")).encode()
    j += b" " * ((-(8 + len(j))) % 8)
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(j)))
        f.write(j)
        f.write(bytes(blob))
    return 8 + len(j) + len(blob)


# ---- the model ------------------------------------------------------------
def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "~/dsv4-tiny")
    out.mkdir(parents=True, exist_ok=True)
    c = CFG
    H, hd, d = c["n_heads"], c["head_dim"], c["hidden"]
    hc, mix = c["hc_mult"], (2 + c["hc_mult"]) * c["hc_mult"]
    ratios = c["compress_ratios"]

    T = {}          # safetensors payload
    W = {}          # dequantised weights the reference uses
    ref_logits = []

    def put_fp8(name, rows, cols):
        b, s, v = fp8_tensor(rows, cols)
        T[name + ".weight"] = (b, "F8_E4M3")
        T[name + ".scale"] = (s, "F8_E8M0")
        return v

    def put_fp4(name, rows, cols):
        b, s, v = fp4_tensor(rows, cols)
        T[name + ".weight"] = (b, "I8")
        T[name + ".scale"] = (s, "F8_E8M0")
        return v

    def put_bf16(name, *shape, scale=0.1):
        t, v = bf16_tensor(*shape, scale=scale)
        T[name] = (t, None)
        return v

    def put_f32(name, *shape, scale=1.0, add=0.0):
        t = torch.randn(*shape) * scale + add
        T[name] = (t, None)
        return t

    # ---- model level ----
    embed = put_bf16("embed.weight", c["vocab"], d, scale=0.5)
    W["norm"] = put_f32("norm.weight", d, scale=0.1, add=1.0)
    head = put_bf16("head.weight", c["vocab"], d, scale=0.3)
    W["hc_head_fn"] = put_f32("hc_head_fn", hc, hc * d, scale=0.05)
    W["hc_head_base"] = put_f32("hc_head_base", hc, scale=0.3)
    W["hc_head_scale"] = put_f32("hc_head_scale", 1, scale=0.1, add=1.0)

    LW = []
    for L in range(c["n_layers"]):
        ratio = ratios[L]
        coff = 2 if ratio == 4 else 1
        w = {}
        p = "layers.%d" % L
        w["attn_norm"] = put_f32(p + ".attn_norm.weight", d, scale=0.1, add=1.0)
        w["ffn_norm"] = put_f32(p + ".ffn_norm.weight", d, scale=0.1, add=1.0)
        w["sink"] = put_f32(p + ".attn.attn_sink", H, scale=1.0)
        w["q_norm"] = put_f32(p + ".attn.q_norm.weight", c["q_lora"], scale=0.1, add=1.0)
        w["kv_norm"] = put_f32(p + ".attn.kv_norm.weight", hd, scale=0.1, add=1.0)
        w["wq_a"] = put_fp8(p + ".attn.wq_a", c["q_lora"], d)
        w["wq_b"] = put_fp8(p + ".attn.wq_b", H * hd, c["q_lora"])
        w["wkv"] = put_fp8(p + ".attn.wkv", hd, d)
        w["wo_a"] = put_fp8(p + ".attn.wo_a",
                            c["o_groups"] * c["o_lora"], H * hd // c["o_groups"])
        w["wo_b"] = put_fp8(p + ".attn.wo_b", d, c["o_groups"] * c["o_lora"])
        if ratio:
            w["c_ape"] = put_f32(p + ".attn.compressor.ape", ratio, coff * hd, scale=0.5)
            w["c_norm"] = put_f32(p + ".attn.compressor.norm.weight", hd,
                                  scale=0.1, add=1.0)
            w["c_wkv"] = put_bf16(p + ".attn.compressor.wkv.weight", coff * hd, d)
            w["c_wgate"] = put_bf16(p + ".attn.compressor.wgate.weight", coff * hd, d)
        if ratio == 4:
            # Present so the binder finds them; index_topk is set high enough
            # that every compressed row is taken, matching the reference.
            put_fp8(p + ".attn.indexer.wq_b", 64 * 8, c["q_lora"])
            put_bf16(p + ".attn.indexer.weights_proj.weight", 64, d)
            put_f32(p + ".attn.indexer.compressor.ape", ratio, 2 * 8, scale=0.5)
            put_f32(p + ".attn.indexer.compressor.norm.weight", 8, scale=0.1, add=1.0)
            put_bf16(p + ".attn.indexer.compressor.wkv.weight", 2 * 8, d)
            put_bf16(p + ".attn.indexer.compressor.wgate.weight", 2 * 8, d)
        w["hc_attn_fn"] = put_f32(p + ".hc_attn_fn", mix, hc * d, scale=0.05)
        w["hc_ffn_fn"] = put_f32(p + ".hc_ffn_fn", mix, hc * d, scale=0.05)
        w["hc_attn_base"] = put_f32(p + ".hc_attn_base", mix, scale=0.3)
        w["hc_ffn_base"] = put_f32(p + ".hc_ffn_base", mix, scale=0.3)
        w["hc_attn_scale"] = put_f32(p + ".hc_attn_scale", 3, scale=0.1, add=1.0)
        w["hc_ffn_scale"] = put_f32(p + ".hc_ffn_scale", 3, scale=0.1, add=1.0)
        w["gate"] = put_bf16(p + ".ffn.gate.weight", c["n_experts"], d, scale=0.3)
        if L < c["num_hash_layers"]:
            tid = torch.randint(0, c["n_experts"], (c["vocab"], c["topk"]),
                                dtype=torch.int64)
            T[p + ".ffn.gate.tid2eid"] = (tid, "I64")
            w["tid2eid"] = tid
            w["gate_bias"] = None
        else:
            w["gate_bias"] = put_f32(p + ".ffn.gate.bias", c["n_experts"], scale=0.5)
            w["tid2eid"] = None
        for e in range(c["n_experts"]):
            w["e%d_w1" % e] = put_fp4("%s.ffn.experts.%d.w1" % (p, e), c["moe_inter"], d)
            w["e%d_w2" % e] = put_fp4("%s.ffn.experts.%d.w2" % (p, e), d, c["moe_inter"])
            w["e%d_w3" % e] = put_fp4("%s.ffn.experts.%d.w3" % (p, e), c["moe_inter"], d)
        w["sh_w1"] = put_fp8(p + ".ffn.shared_experts.w1", c["moe_inter"], d)
        w["sh_w2"] = put_fp8(p + ".ffn.shared_experts.w2", d, c["moe_inter"])
        w["sh_w3"] = put_fp8(p + ".ffn.shared_experts.w3", c["moe_inter"], d)
        LW.append(w)

    # ---- config.json, in the released schema -----------------------------
    cfg = {
        "architectures": ["DeepseekV4ForCausalLM"], "model_type": "deepseek_v4",
        "hidden_size": d, "num_hidden_layers": c["n_layers"], "vocab_size": c["vocab"],
        "rms_norm_eps": c["rms_eps"], "max_position_embeddings": 4096,
        "num_attention_heads": H, "num_key_value_heads": 1, "head_dim": hd,
        "q_lora_rank": c["q_lora"], "o_lora_rank": c["o_lora"],
        "o_groups": c["o_groups"], "qk_rope_head_dim": c["qk_rope"],
        "sliding_window": c["window"],
        "index_n_heads": 64, "index_head_dim": 8, "index_topk": 1048576,
        "compress_rope_theta": 160000.0, "compress_ratios": ratios,
        "hc_mult": hc, "hc_sinkhorn_iters": c["sinkhorn_iters"], "hc_eps": c["hc_eps"],
        "n_routed_experts": c["n_experts"], "num_experts_per_tok": c["topk"],
        "n_shared_experts": c["n_shared"], "moe_intermediate_size": c["moe_inter"],
        "routed_scaling_factor": c["route_scale"], "norm_topk_prob": True,
        "scoring_func": "sqrtsoftplus", "topk_method": "noaux_tc",
        "swiglu_limit": c["swiglu_limit"], "rope_theta": 10000.0,
        "rope_scaling": {"type": "yarn", "factor": 16, "beta_fast": 32,
                         "beta_slow": 1, "original_max_position_embeddings": 2048},
        "num_hash_layers": c["num_hash_layers"], "num_nextn_predict_layers": 1,
        "expert_dtype": "fp4", "torch_dtype": "bfloat16",
        "quantization_config": {"activation_scheme": "dynamic", "fmt": "e4m3",
                                "quant_method": "fp8", "scale_fmt": "ue8m0",
                                "weight_block_size": [128, 128]},
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2) + "\n")
    n = write_safetensors(out / "model-00001-of-00001.safetensors", T)
    print("  %s/  %d tensors, %d bytes" % (out, len(T), n))

    # ---- run the reference over several positions -------------------------
    fc_d = R.freqs_cis(c["qk_rope"], 256, 0, 10000.0, 16.0, 32.0, 1.0)
    fc_c = R.freqs_cis(c["qk_rope"], 256, 2048, 160000.0, 16.0, 32.0, 1.0)
    caches, states, ncomp = [], [], []
    for L in range(c["n_layers"]):
        r = ratios[L]
        caches.append(torch.zeros(c["window"] + 64, hd))
        states.append(R.CompressorState(r, hd) if r else None)
        ncomp.append([0])

    tokens = [3, 11, 27, 5, 40, 19]
    trace = {}          # stage -> flat list, position 0 only
    for pos, tid in enumerate(tokens):
        h = embed[tid].unsqueeze(0).repeat(hc, 1).clone()
        if pos == 0:
            trace["embed"] = [float(v) for v in h.reshape(-1)]
        for L in range(c["n_layers"]):
            r = ratios[L]
            lc = dict(c)
            lc["ratio"] = r
            fc = fc_c if r else fc_d
            if r:
                h = R.block_compressed(h, LW[L], lc, caches[L], states[L], fc,
                                       pos, tid, ncomp[L])
            else:
                h = R.block_dense(h, LW[L], lc, caches[L], fc, pos, tid)
            if pos == 0:
                trace["layer%d" % L] = [float(v) for v in h.reshape(-1)]
        # THE MODEL HEAD: its own hc reduction, NO Sinkhorn and no comb matrix.
        flat = h.reshape(-1).float()
        rsq = torch.rsqrt(flat.square().mean() + c["rms_eps"])
        m = (W["hc_head_fn"] @ flat) * rsq
        pre = torch.sigmoid(m * W["hc_head_scale"][0] + W["hc_head_base"]) + c["hc_eps"]
        y = (pre.unsqueeze(-1) * h).sum(0)
        if pos == 0:
            trace["hc_head"] = [float(v) for v in y]
        y = R.rmsnorm(y, W["norm"], c["rms_eps"])
        if pos == 0:
            trace["norm"] = [float(v) for v in y]
        ref_logits.append([float(v) for v in (y @ head.T).tolist()])

    (out / "expected.json").write_text(json.dumps(
        {"tokens": tokens, "logits": ref_logits, "vocab": c["vocab"],
         "trace": trace}) + "\n")
    print("  expected.json: %d positions x %d logits" % (len(tokens), c["vocab"]))
    am = [max(range(c["vocab"]), key=lambda i, L=L: L[i]) for L in ref_logits]
    print("  reference argmax per position: %s" % am)


if __name__ == "__main__":
    main()
