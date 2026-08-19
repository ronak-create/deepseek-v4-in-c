#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Derive the bad-config fixtures from the released DeepSeek-V4-Flash config.

Each mutation is ONE realistic mistake, applied to a file that is otherwise
byte-for-byte the real thing. Deriving them rather than hand-writing them means
they cannot drift away from the schema the reader is actually meant to accept:
if DeepSeek changes a field, regenerating produces mutations of the NEW file.

Run from the repository root:  python3 tools/make_cfg_fixtures.py
"""
import json
import pathlib
import sys

CFG = pathlib.Path("tests/fixtures/cfg")
BASE = CFG / "dsv4_flash_config.json"


def drop(cfg, key):
    cfg.pop(key, None)
    return cfg


def mutations():
    """name -> (fn, one-line reason the reader must refuse it)"""
    return {
        # The key a reader written against a nested schema misses entirely. With
        # it absent every layer answers dense_attn -> a different model.
        "no_compress_ratios": (
            lambda c: drop(c, "compress_ratios"),
            "compress_ratios absent",
        ),
        # Parses fine, covers only the first 10 layers. The remaining 33 would
        # read past the list or fall back to a default.
        "short_compress_ratios": (
            lambda c: c.update(compress_ratios=c["compress_ratios"][:10]) or c,
            "compress_ratios shorter than num_hidden_layers",
        ),
        # The subtle one, and the mistake this test suite caught in its own first
        # draft. The released list is num_hidden_layers + num_nextn_predict_layers
        # long because it covers the MTP module too. Dropping that trailing entry
        # leaves a list that looks exactly right -- one per layer -- and is wrong.
        # The mirror mistake, KEEPING the tail and reading it as a decoder layer,
        # invents a dense-attention layer at the end, because the MTP entry is 0.
        "no_mtp_compress_tail": (
            lambda c: c.update(
                compress_ratios=c["compress_ratios"][: c["num_hidden_layers"]]
            ) or c,
            "compress_ratios exactly n_layers long, MTP entry dropped",
        ),
        # A ratio with no kernel behind it. Silently treating it as the nearest
        # implemented value would change the attention path for that layer.
        "bad_compress_value": (
            lambda c: c.update(
                compress_ratios=[7] + c["compress_ratios"][1:]
            ) or c,
            "compress ratio 7 has no implementation",
        ),
        # Both functions produce similarly-scaled routing weights, so this
        # reorders top-k on a minority of rows and degrades quality quietly.
        "bad_scoring_func": (
            lambda c: c.update(scoring_func="sigmoid") or c,
            "scoring_func sigmoid, not sqrtsoftplus",
        ),
        # Same tensor names, twice the bytes per weight. Reading fp4 offsets
        # against fp8 data produces garbage that still has finite magnitude.
        "bad_expert_dtype": (
            lambda c: c.update(expert_dtype="fp8") or c,
            "expert_dtype fp8, not fp4",
        ),
        # A reader that defaults this to 1 collapses mHC to an ordinary
        # residual stream. That is the natural guess and it is wrong.
        "no_hc_mult": (
            lambda c: drop(c, "hc_mult"),
            "hc_mult absent",
        ),
        # num_hash_layers must be a proper prefix. Setting it to n_layers would
        # mean no layer carries ffn.gate.bias, which is a different architecture.
        "hash_layers_all": (
            lambda c: c.update(num_hash_layers=c["num_hidden_layers"]) or c,
            "num_hash_layers == n_layers, leaving no scored layer",
        ),
        # Field names collide across DeepSeek generations; model_type does not.
        "bad_model_type": (
            lambda c: c.update(model_type="deepseek_v3") or c,
            "a sibling architecture",
        ),
        # The dangerous one: 10.0 IS the correct value, so a defaulting reader
        # produces a correct-looking activation and hides that it read nothing
        # else either.
        "no_swiglu_limit": (
            lambda c: drop(c, "swiglu_limit"),
            "swiglu_limit absent",
        ),
    }


def main():
    if not BASE.exists():
        sys.exit(f"missing {BASE}; fetch the released config first")
    base = json.loads(BASE.read_text())

    for name, (fn, why) in mutations().items():
        cfg = fn(json.loads(json.dumps(base)))
        out = CFG / f"{name}.json"
        out.write_text(json.dumps(cfg, indent=2) + "\n")
        print(f"  {out.name:<28} {why}")

    print(f"\n{len(mutations())} bad fixtures derived from {BASE.name}")


if __name__ == "__main__":
    main()
