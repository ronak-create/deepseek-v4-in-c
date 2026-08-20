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
    # One routed expert: FP4 packed 2-per-byte, exposed as I8, scales 1x32.
    # Only the SCALES here -- the weights are emitted far below. See the note on
    # expert layout at the end of this list.
    ("layers.2.ffn.experts.0.w1.scale",             "F8_E8M0", [2048, 128]),
    ("layers.2.ffn.experts.0.w2.scale",             "F8_E8M0", [4096, 64]),
    ("layers.2.ffn.experts.0.w3.scale",             "F8_E8M0", [2048, 128]),


    # --- layer 3: ratio 128, NO indexer, scored routing ------------------------
    # Deliberately complete, and deliberately WITHOUT any indexer.* tensor: the
    # binder must not request one here, and the gate checks that it does not.
    ("layers.3.attn_norm.weight",                   "BF16",    [4096]),
    ("layers.3.attn.q_norm.weight",                 "BF16",    [1024]),
    ("layers.3.attn.kv_norm.weight",                "BF16",    [512]),
    ("layers.3.attn.attn_sink",                     "F32",     [64]),
    ("layers.3.attn.wq_a.weight",                   "F8_E4M3", [1024, 4096]),
    ("layers.3.attn.wq_a.scale",                    "F8_E8M0", [8, 32]),
    ("layers.3.attn.wq_b.weight",                   "F8_E4M3", [32768, 1024]),
    ("layers.3.attn.wq_b.scale",                    "F8_E8M0", [256, 8]),
    ("layers.3.attn.wkv.weight",                    "F8_E4M3", [512, 4096]),
    ("layers.3.attn.wkv.scale",                     "F8_E8M0", [4, 32]),
    ("layers.3.attn.wo_a.weight",                   "F8_E4M3", [8192, 4096]),
    ("layers.3.attn.wo_a.scale",                    "F8_E8M0", [64, 32]),
    ("layers.3.attn.wo_b.weight",                   "F8_E4M3", [4096, 8192]),
    ("layers.3.attn.wo_b.scale",                    "F8_E8M0", [32, 64]),
    # ratio 128 -> coff 1, so the compressor is HALF the width of layer 2's
    ("layers.3.attn.compressor.ape",                "F32",     [128, 512]),
    ("layers.3.attn.compressor.norm.weight",        "BF16",    [512]),
    ("layers.3.attn.compressor.wkv.weight",         "BF16",    [512, 4096]),
    ("layers.3.attn.compressor.wgate.weight",       "BF16",    [512, 4096]),
    ("layers.3.hc_attn_fn",                         "F32",     [24, 16384]),
    ("layers.3.hc_attn_base",                       "F32",     [24]),
    ("layers.3.hc_attn_scale",                      "F32",     [3]),
    ("layers.3.hc_ffn_fn",                          "F32",     [24, 16384]),
    ("layers.3.hc_ffn_base",                        "F32",     [24]),
    ("layers.3.hc_ffn_scale",                       "F32",     [3]),
    ("layers.3.ffn.gate.weight",                    "BF16",    [256, 4096]),
    ("layers.3.ffn.gate.bias",                      "F32",     [256]),
    ("layers.3.ffn.shared_experts.w1.weight",       "F8_E4M3", [2048, 4096]),
    ("layers.3.ffn.shared_experts.w1.scale",        "F8_E8M0", [16, 32]),
    ("layers.3.ffn.shared_experts.w2.weight",       "F8_E4M3", [4096, 2048]),
    ("layers.3.ffn.shared_experts.w2.scale",        "F8_E8M0", [32, 16]),
    ("layers.3.ffn.shared_experts.w3.weight",       "F8_E4M3", [2048, 4096]),
    ("layers.3.ffn.shared_experts.w3.scale",        "F8_E8M0", [16, 32]),
    ("layers.3.ffn_norm.weight",                    "BF16",    [4096]),
    # A routed expert on a SECOND layer, so the cache gate can hold two distinct
    # entries and observe eviction order. One expert per layer is enough: the
    # cache keys on (layer, expert) and never on tensor content.
    ("layers.3.ffn.experts.0.w1.scale",             "F8_E8M0", [2048, 128]),
    ("layers.3.ffn.experts.0.w2.scale",             "F8_E8M0", [4096, 64]),
    ("layers.3.ffn.experts.0.w3.scale",             "F8_E8M0", [2048, 128]),
]

# Routed experts 1..5 on layer 2, so the CONCURRENT fetch has something to be
# concurrent about. With only expert 0 the comparison against the serial path is
# vacuous: dsv4_cache_get_many's parallel branch is guarded by `ntodo > 1` and
# would never be entered, and a gate that cannot fail is not a gate.
#
# Shapes are copied from expert 0 rather than derived, because the FP4 scale
# grid is 1x32 over the UNPACKED columns while the weight is packed 2 per byte:
# w1 [2048, 2048] bytes carries 4096 columns and so scales [2048, 128], and w2
# [4096, 1024] bytes carries 2048 columns and so scales [4096, 64]. Deriving
# that in one expression is exactly how the two grids get confused.
for _e in range(1, 6):
    TENSORS += [
        ("layers.2.ffn.experts.%d.w1.scale"  % _e, "F8_E8M0", [2048, 128]),
        ("layers.2.ffn.experts.%d.w2.scale"  % _e, "F8_E8M0", [4096, 64]),
        ("layers.2.ffn.experts.%d.w3.scale"  % _e, "F8_E8M0", [2048, 128]),
    ]

# AN EXPERT IS TWO SEPARATED RUNS ON DISK, AND THAT IS THE WHOLE POINT.
#
# The released checkpoint does NOT store an expert's six tensors together. All
# three scales sit in one contiguous run, and the three weights sit in another
# ~341 MB away, at a DIFFERENT 4096 alignment residue (measured on Flash shard 4:
# scales at residue 1176, weights at 2712). dsv4_cache coalesces by file offset,
# so a real expert always loads as two runs, and the O_DIRECT window for the
# second one starts before its payload and can reach back over the first.
#
# An earlier version of this fixture emitted the six interleaved and contiguous,
# which collapsed to ONE run. Every cache gate passed while the engine silently
# corrupted every expert it loaded from the real checkpoint. So the weights go
# here, after everything else, with odd-sized tensors in between to keep the two
# residues different -- the fixture is only useful to the extent that it has the
# shape of the thing it stands in for.
for _e in range(0, 6):
    TENSORS += [
        ("layers.2.ffn.experts.%d.w1.weight" % _e, "I8",      [2048, 2048]),
        ("layers.2.ffn.experts.%d.w2.weight" % _e, "I8",      [4096, 1024]),
        ("layers.2.ffn.experts.%d.w3.weight" % _e, "I8",      [2048, 2048]),
    ]
TENSORS += [
    ("layers.3.ffn.experts.0.w1.weight",            "I8",      [2048, 2048]),
    ("layers.3.ffn.experts.0.w2.weight",            "I8",      [4096, 1024]),
    ("layers.3.ffn.experts.0.w3.weight",            "I8",      [2048, 2048]),
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

    # The gate reads this rather than hard-coding a count. A literal in the test
    # goes stale every time a tensor is added, which trains the reader to edit
    # the number instead of asking why it changed.
    (out / "COUNT").write_text(f"{len(TENSORS)}\n")

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
