#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pack tokenizer.json into a flat binary the C loads with two reads.

WHY PACK IT AT ALL
    tokenizer.json is 128,000 vocab entries plus 127,741 merges plus 1,283 added
    tokens. Parsing that JSON at every process start costs more than loading the
    entire dense trunk of a small model, and the C would need a general JSON
    parser sized for it. The packed form is read in two calls and needs no
    parsing.

FORMAT  (all little-endian; the C mirrors this exactly)
    magic      "DSV4TOK\\0"          8 bytes
    version    u32 = 1
    n_vocab    u32
    n_merge    u32
    n_added    u32
    str_bytes  u64                   total size of the string blob
    -- then --
    vocab_off  u32 * n_vocab         offset into the blob for each id
    vocab_len  u16 * n_vocab         byte length of each token
    merge_a    u32 * n_merge         left  piece, as a VOCAB ID
    merge_b    u32 * n_merge         right piece, as a VOCAB ID
    merge_out  u32 * n_merge         the id the pair merges into
    added_id   u32 * n_added         ids that are special/added tokens
    blob       str_bytes             every token's bytes, concatenated

MERGES ARE STORED AS IDS, NOT STRINGS
    The reference stores merges as pairs of token strings. Resolving them to ids
    once here means the C's merge loop compares integers instead of doing string
    lookups in its inner loop, and a merge whose pieces are not in the vocab is
    caught HERE rather than silently never firing.

TOKEN BYTES ARE THE BYTE-LEVEL ALPHABET, NOT UTF-8 TEXT
    A ByteLevel BPE maps each of the 256 input bytes to a printable codepoint
    first, so a token like "Ġthe" means <space>the. The blob stores the token
    exactly as the vocab spells it; the C converts input bytes into the same
    alphabet before matching. Storing decoded UTF-8 here would make tokens that
    differ only in leading whitespace collide.
"""
import json
import os
import pathlib
import struct
import sys

MAGIC = b"DSV4TOK\0"


def byte_level_alphabet():
    """GPT-2's byte -> unicode map, the one HF ByteLevel uses."""
    bs = (list(range(ord("!"), ord("~") + 1))
          + list(range(0xA1, 0xAC + 1))
          + list(range(0xAE, 0xFF + 1)))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def main():
    model = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                               else "~/models/dsv4-flash")
    out = pathlib.Path(sys.argv[2] if len(sys.argv) > 2
                       else "tests/fixtures/tok/tokenizer.bin")
    out.parent.mkdir(parents=True, exist_ok=True)

    tj = json.loads(open(os.path.join(model, "tokenizer.json"),
                         encoding="utf-8").read())
    vocab = tj["model"]["vocab"]
    merges = tj["model"]["merges"]
    added = tj.get("added_tokens", [])

    n_vocab = max(max(vocab.values()), max((a["id"] for a in added), default=0)) + 1
    print(f"  vocab entries {len(vocab):,}, added {len(added):,}, "
          f"highest id {n_vocab - 1:,}")

    tokens = [None] * n_vocab
    for s, i in vocab.items():
        tokens[i] = s
    for a in added:
        tokens[a["id"]] = a["content"]

    holes = sum(1 for t in tokens if t is None)
    if holes:
        # Not fatal: a gap just means no input can ever produce that id. Filling
        # with an empty string keeps the offset table dense and the C simple.
        print(f"  {holes} id(s) have no token; filling with empty strings")
        tokens = [t if t is not None else "" for t in tokens]

    blob = bytearray()
    offs, lens = [], []
    index = {}
    for i, t in enumerate(tokens):
        b = t.encode("utf-8")
        if len(b) > 0xFFFF:
            sys.exit(f"token {i} is {len(b)} bytes, too long for a u16 length")
        offs.append(len(blob))
        lens.append(len(b))
        blob += b
        if t and t not in index:
            index[t] = i

    ma, mb, mo = [], [], []
    dropped = 0
    for m in merges:
        a, b = (m if isinstance(m, list) else m.split(" ", 1))
        ia, ib, io = index.get(a), index.get(b), index.get(a + b)
        if ia is None or ib is None or io is None:
            # Caught here rather than silently never firing at runtime.
            dropped += 1
            continue
        ma.append(ia); mb.append(ib); mo.append(io)
    if dropped:
        print(f"  {dropped} merge(s) reference tokens not in the vocab; dropped")

    added_ids = sorted(a["id"] for a in added)

    with open(out, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<IIII", 1, n_vocab, len(ma), len(added_ids)))
        f.write(struct.pack("<Q", len(blob)))
        f.write(struct.pack(f"<{n_vocab}I", *offs))
        f.write(struct.pack(f"<{n_vocab}H", *lens))
        f.write(struct.pack(f"<{len(ma)}I", *ma))
        f.write(struct.pack(f"<{len(mb)}I", *mb))
        f.write(struct.pack(f"<{len(mo)}I", *mo))
        f.write(struct.pack(f"<{len(added_ids)}I", *added_ids))
        f.write(bytes(blob))

    sz = os.path.getsize(out)
    print(f"\n  {out}  {sz:,} bytes")
    print(f"  {n_vocab:,} ids, {len(ma):,} merges, {len(added_ids):,} added, "
          f"blob {len(blob):,} bytes")


if __name__ == "__main__":
    main()
