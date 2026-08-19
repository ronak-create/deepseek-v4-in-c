#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Write a synthetic safetensors shard carrying DeepSeek-V4's exact dtypes and
tensor geometry, so the reader can be gated before 148 GB has finished arriving.

Every shape here was read from the real checkpoint headers (Flash shards 4 and 5),
not invented:

    layers.2.*   a compress_ratio == 4 layer: has an indexer, hash-routed
                 (ffn.gate.tid2eid, no ffn.gate.bias)
    layers.3.*   a compress_ratio == 128 layer: no indexer, scored
                 (ffn.gate.bias, no tid2eid)

The point is the dtype coverage. A reader that only knows U8/BF16/F16/F32 must
REFUSE this file rather than misread it, and a reader that has been extended must
compute byte counts that agree with the declared data_offsets.

Usage:  python3 tools/make_st_fixture.py tests/fixtures/st
"""
import json
import pathlib
import struct
import sys

# (name, dtype, shape) -- verbatim from the real Flash checkpoint headers.
# Sizes are cut down where a tensor is huge; the DTYPES and the RANK are what
# the reader is being tested on, and those are exact.
TENSORS = [
    # --- layer 2: ratio 4, indexed, hash-routed -------------------------------
    ("layers.2.attn_norm.weight",                   "BF16",    [4096]),
    ("layers.2.attn.q_norm.weight",                 "BF16",    [1024]),
    ("layers.2.attn.kv_norm.weight",                "BF16",    [512]),
    ("layers.2.attn.attn_sink",                     "F32",     [64]),
    ("layers.2.attn.wq_a.weight",                   "F8_E4M3", [1024, 4096]),
    ("layers.2.attn.wq_a.scale",                    "F8_E8M0", [8, 32]),
    ("layers.2.attn.wq_b.weight",                   "F8_E4M3", [32768, 1024]),
    ("layers.2.attn.wq_b.scale",                    "F8_E8M0", [256, 8]),
    ("layers.2.attn.wkv.weight",                    "F8_E4M3", [512, 4096]),
    ("layers.2.attn.wkv.scale",                     "F8_E8M0", [4, 32]),
    ("layers.2.attn.wo_a.weight",                   "F8_E4M3", [8192, 4096]),
    ("layers.2.attn.wo_a.scale",                    "F8_E8M0", [64, 32]),
    ("layers.2.attn.wo_b.weight",                   "F8_E4M3", [4096, 8192]),
    ("layers.2.attn.wo_b.scale",                    "F8_E8M0", [32, 64]),
    # the shared expert: FP8, not FP4, and resident rather than streamed
    ("layers.2.ffn.shared_experts.w1.weight",       "F8_E4M3", [2048, 4096]),
    ("layers.2.ffn.shared_experts.w1.scale",        "F8_E8M0", [16, 32]),
    ("layers.2.ffn.shared_experts.w2.weight",       "F8_E4M3", [4096, 2048]),
    ("layers.2.ffn.shared_experts.w2.scale",        "F8_E8M0", [32, 16]),
    ("layers.2.ffn.shared_experts.w3.weight",       "F8_E4M3", [2048, 4096]),
    ("layers.2.ffn.shared_experts.w3.scale",        "F8_E8M0", [16, 32]),
    # HCA compressor, present only where compress_ratio != 0
    ("layers.2.attn.compressor.ape",                "F32",     [4, 1024]),
    ("layers.2.attn.compressor.norm.weight",        "BF16",    [512]),
    ("layers.2.attn.compressor.wkv.weight",         "BF16",    [1024, 4096]),
    ("layers.2.attn.compressor.wgate.weight",       "BF16",    [1024, 4096]),
    # CSA indexer, present ONLY where compress_ratio == 4
    ("layers.2.attn.indexer.weights_proj.weight",   "BF16",    [64, 4096]),
    ("layers.2.attn.indexer.wq_b.weight",           "F8_E4M3", [8192, 1024]),
    ("layers.2.attn.indexer.wq_b.scale",            "F8_E8M0", [64, 8]),
    ("layers.2.attn.indexer.compressor.ape",        "F32",     [4, 256]),
    ("layers.2.attn.indexer.compressor.norm.weight","BF16",    [128]),
    ("layers.2.attn.indexer.compressor.wkv.weight", "BF16",    [256, 4096]),
    ("layers.2.attn.indexer.compressor.wgate.weight","BF16",   [256, 4096]),
    # mHC
    # mix_hc = (2+hc_mult)*hc_mult = 24, hc_dim = hc_mult*hidden = 16384
    # (model.py:663-667). The fn tensors are the ones a wrong derivation breaks.
    ("layers.2.hc_attn_fn",                         "F32",     [24, 16384]),
    ("layers.2.hc_attn_base",                       "F32",     [24]),
    ("layers.2.hc_attn_scale",                      "F32",     [3]),
    ("layers.2.hc_ffn_fn",                          "F32",     [24, 16384]),
    ("layers.2.hc_ffn_base",                        "F32",     [24]),
    ("layers.2.hc_ffn_scale",                       "F32",     [3]),
    # routing: HASH layer -> tid2eid, and NO bias.  I64 is a dtype K3 never saw.
    ("layers.2.ffn.gate.weight",                    "BF16",    [256, 4096]),
    ("layers.2.ffn.gate.tid2eid",                   "I64",     [512, 6]),
    ("layers.2.ffn_norm.weight",                    "BF16",    [4096]),
    # one routed expert: FP4 packed 2-per-byte, exposed as I8, scales 1x32
    ("layers.2.ffn.experts.0.w1.weight",            "I8",      [2048, 2048]),
    ("layers.2.ffn.experts.0.w1.scale",             "F8_E8M0", [2048, 128]),
    ("layers.2.ffn.experts.0.w2.weight",            "I8",      [4096, 1024]),
    ("layers.2.ffn.experts.0.w2.scale",             "F8_E8M0", [4096, 64]),
    ("layers.2.ffn.experts.0.w3.weight",            "I8",      [2048, 2048]),
    ("layers.2.ffn.experts.0.w3.scale",             "F8_E8M0", [2048, 128]),

    # --- layer 3: ratio 128, NO indexer, scored routing ------------------------
    ("layers.3.attn_norm.weight",                   "BF16",    [4096]),
    ("layers.3.attn.compressor.ape",                "F32",     [128, 512]),
    ("layers.3.attn.compressor.wkv.weight",         "BF16",    [512, 4096]),
    ("layers.3.ffn.gate.weight",                    "BF16",    [256, 4096]),
    ("layers.3.ffn.gate.bias",                      "F32",     [256]),
    ("layers.3.ffn_norm.weight",                    "BF16",    [4096]),
]

ELEMSIZE = {
    "F32": 4, "BF16": 2, "F16": 2, "U8": 1,
    "F8_E4M3": 1,   # one byte per element
    "F8_E8M0": 1,   # one byte per element (exponent-only scale)
    "I8": 1,        # FP4 packed two-per-byte: the SHAPE is already in bytes
    "I64": 8,
}


def build(tensors, seed=1234):
    """Return (header_json_bytes, payload_bytes) laid out as safetensors does."""
    off, entries, blobs = 0, {}, []
    state = seed
    for name, dt, shape in tensors:
        n = 1
        for d in shape:
            n *= d
        nb = n * ELEMSIZE[dt]
        # deterministic, non-constant filler: a constant blob would let an
        # offset bug go unnoticed because every window looks identical.
        buf = bytearray(nb)
        for i in range(0, nb, 251):
            state = (state * 1103515245 + 12345) & 0xFFFFFFFF
            buf[i] = (state >> 16) & 0xFF
        blobs.append(bytes(buf))
        entries[name] = {"dtype": dt, "shape": shape, "data_offsets": [off, off + nb]}
        off += nb
    return json.dumps(entries, separators=(",", ":")).encode(), b"".join(blobs)


def write_shard(path, tensors):
    hdr, payload = build(tensors)
    pad = (-(8 + len(hdr))) % 8          # safetensors allows header padding
    hdr = hdr + b" " * pad
    with open(path, "wb") as f:
        f.write(struct.pack("<Q", len(hdr)))
        f.write(hdr)
        f.write(payload)
    return 8 + len(hdr) + len(payload)


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/st")
    out.mkdir(parents=True, exist_ok=True)

    n = write_shard(out / "model-00001-of-00001.safetensors", TENSORS)
    print(f"  {out}/model-00001-of-00001.safetensors  {n:,} bytes, {len(TENSORS)} tensors")

    dts = sorted({t[1] for t in TENSORS})
    print(f"  dtypes covered: {' '.join(dts)}")

    # A second directory holding one tensor of a dtype nothing implements. The
    # reader must refuse this outright rather than pick the nearest width.
    bad = out.parent / "st_bad"
    bad.mkdir(parents=True, exist_ok=True)
    write_shard(bad / "model-00001-of-00001.safetensors",
                [("layers.0.attn.w.weight", "F16", [8, 8])])
    # rewrite its header with a dtype that does not exist anywhere
    p = bad / "model-00001-of-00001.safetensors"
    raw = bytearray(p.read_bytes())
    ln = struct.unpack("<Q", raw[:8])[0]
    h = raw[8:8 + ln].replace(b'"F16"', b'"F4X"')
    h = h + b" " * (ln - len(h))
    raw[8:8 + ln] = h
    p.write_bytes(bytes(raw))
    print(f"  {bad}/model-00001-of-00001.safetensors  dtype 'F4X' -> must be REFUSED")


if __name__ == "__main__":
    main()
