#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pack the dense trunk into one file where each layer is a single contiguous run.

WHY THIS EXISTS
    A decode step walks layers 0..N-1 in a fixed order, every token. If each
    layer's weights are scattered across a 160 GB checkpoint, that walk is
    thousands of small reads. Rewritten into trunk.bin, one layer is ONE pread
    at a known offset, and the memory requirement becomes a dial: pin a prefix,
    stream the rest.

THE ASSUMPTION K3 MADE THAT IS FALSE HERE
    kimi-k3-in-c's packer takes each layer's trunk to be a single contiguous run
    in its shard, and derives every tensor's in-slot offset as
    (absolute shard offset - run start).

    Measured on DeepSeek-V4-Flash, that is wrong. The checkpoint groups tensors
    by dtype, so a layer's trunk is TWO runs with all 256 routed experts between
    them:

        trunk x22  ->  EXPERT x768  ->  trunk x11  ->  EXPERT x768

    On layer 3 the first run is 13.28 MB (norms, hc_*, compressor, gate, and
    every attention .scale) and the second is 126 MB (the big F8_E4M3 .weight
    tensors), 192 MB apart. A weight and its own scale are in different runs.

    So this packer does NOT derive offsets from a single translation. It reads
    each tensor individually and records where it actually landed in the
    destination. The destination is still one contiguous run per layer, which is
    the only property the streamer needs -- and the streamer is unchanged.

WHAT IS DELIBERATELY NOT PACKED
    The routed experts. They are 137 GB of the 160, they are streamed on demand
    by the LRU cache, and copying them here would defeat the point.
"""
import argparse
import json
import os
import struct
import sys

ALIGN = 4096          # O_DIRECT needs offset, length and buffer all aligned

ELEMSIZE = {"F32": 4, "BF16": 2, "F16": 2, "U8": 1,
            "F8_E4M3": 1, "F8_E8M0": 1, "I8": 1, "I64": 8}


def read_header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def is_expert(name):
    return ".ffn.experts." in name


def index_checkpoint(model_dir):
    """name -> (shard_path, absolute_offset, nbytes, dtype), skipping experts."""
    idx = {}
    shards = sorted(f for f in os.listdir(model_dir) if f.endswith(".safetensors"))
    if not shards:
        sys.exit(f"no .safetensors in {model_dir}")
    for s in shards:
        p = os.path.join(model_dir, s)
        try:
            hdr, base = read_header(p)
        except Exception as e:
            print(f"  skipping {s}: {e}")
            continue
        for name, v in hdr.items():
            if is_expert(name):
                continue
            a, b = v["data_offsets"]
            idx[name] = (p, base + a, b - a, v["dtype"])
    return idx, len(shards)


def layer_tensors(idx, layer):
    pfx = f"layers.{layer}."
    return sorted(k for k in idx if k.startswith(pfx))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("out_dir")
    ap.add_argument("--layers", type=int, default=None,
                    help="pack only the first N layers (for testing)")
    args = ap.parse_args()

    cfg = json.load(open(os.path.join(args.model_dir, "config.json")))
    n_layers = cfg["num_hidden_layers"]
    if args.layers:
        n_layers = min(n_layers, args.layers)

    os.makedirs(args.out_dir, exist_ok=True)
    idx, nshard = index_checkpoint(args.model_dir)
    print(f"indexed {len(idx):,} non-expert tensors across {nshard} shards")

    bin_path = os.path.join(args.out_dir, "trunk.bin")
    manifest = {"n_layers": n_layers, "align": ALIGN, "layers": []}

    packed = skipped = 0
    with open(bin_path, "wb") as out:
        for L in range(n_layers):
            names = layer_tensors(idx, L)
            if not names:
                skipped += 1
                manifest["layers"].append(None)
                continue

            # Pad the layer's start so every run begins O_DIRECT-aligned.
            out.seek((out.tell() + ALIGN - 1) // ALIGN * ALIGN)
            run_start = out.tell()

            tensors = []
            for name in names:
                path, off, nb, dt = idx[name]
                with open(path, "rb") as f:
                    f.seek(off)
                    buf = f.read(nb)
                if len(buf) != nb:
                    sys.exit(f"short read on {name}: {len(buf)} of {nb}")
                # Offsets are RECORDED, not derived: the source runs are not
                # contiguous, so there is no single translation to apply.
                tensors.append({"name": name,
                                "off": out.tell() - run_start,
                                "nbytes": nb,
                                "dtype": dt})
                out.write(buf)
                # 8-align inside the run so a widened read is never misaligned.
                pad = (-out.tell()) % 8
                if pad:
                    out.write(b"\0" * pad)

            run_bytes = out.tell() - run_start
            manifest["layers"].append({"file_off": run_start,
                                       "nbytes": run_bytes,
                                       "tensors": tensors})
            packed += 1
            print(f"  layer {L:>3}: {len(tensors):>3} tensors, "
                  f"{run_bytes / 1048576:8.2f} MB at {run_start:,}")

        # Pad the file itself so an O_DIRECT read of the last run cannot run off
        # the end when it is widened out to an aligned window.
        out.seek((out.tell() + ALIGN - 1) // ALIGN * ALIGN - 1)
        out.write(b"\0")

    with open(os.path.join(args.out_dir, "trunk.json"), "w") as f:
        json.dump(manifest, f)

    total = os.path.getsize(bin_path)
    print(f"\npacked {packed} layers ({skipped} absent) -> {bin_path}")
    print(f"  {total:,} bytes = {total / 1073741824:.2f} GB")
    if packed:
        print(f"  mean {total / packed / 1048576:.2f} MB per layer, "
              f"one pread each")


if __name__ == "__main__":
    main()
