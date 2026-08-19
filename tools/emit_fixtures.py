#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Run the PyTorch reference on random inputs and write fixtures for the C gates.

WHAT MAKES A FIXTURE USEFUL HERE
    Random inputs alone would pass on several of the traps this port has already
    hit -- a symmetric SwiGLU clamp only shows up on strongly negative gates, a
    missing router ReLU only on negative head votes. So each case below is shaped
    to EXERCISE the trap, and that intent is recorded next to it.

    The values are still produced by the reference, not by hand: the point is to
    catch a misreading of DeepSeek's semantics, and hand-computed expectations
    would only re-encode my own reading of them.

Usage:  python3 tools/emit_fixtures.py tests/fixtures/ref
"""
import json
import os
import pathlib
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dsv4_ref as R  # noqa: E402

torch.manual_seed(20260819)


def t2l(x):
    """Full float32 precision, NOT rounded to a fixed number of decimal places.

    An earlier version used round(v, 9), which is nine DECIMAL PLACES rather than
    nine significant figures. That is fine for values near 1 and destroys small
    ones: a silu output of 2e-9 keeps a single significant digit, and a rope
    component of 9e-6 keeps four. The C then disagreed with the reference by up
    to 3e-3 relative and looked like a kernel bug when it was the fixture.

    float() on a float32 tensor element already yields the exact value; json
    writes it with repr, which round-trips. So no rounding at all is correct
    here, and the fixtures are only a few KB larger.
    """
    return [float(v) for v in torch.as_tensor(x).flatten().tolist()]


CASES = {}


def case(name, **kv):
    CASES[name] = kv


# ---- RMSNorm -------------------------------------------------------------
# eps on the MEAN of squares. A tiny-variance case is where eps-on-mean and
# eps-on-sum diverge; a normal case checks the ordinary path.
for tag, scale in (("normal", 1.0), ("tiny", 1e-4)):
    n = 64
    x = torch.randn(n) * scale
    w = torch.randn(n) * 0.5 + 1.0
    case(f"rmsnorm_{tag}", n=n, eps=1e-6, x=t2l(x), w=t2l(w),
         y=t2l(R.rmsnorm(x, w, 1e-6)))

# ---- SwiGLU --------------------------------------------------------------
# Spread WIDE so both clamp edges are hit: gate must be clamped above only, up
# on both sides. Values beyond +/-limit on purpose.
n = 96
gate = torch.randn(n) * 20.0
up = torch.randn(n) * 20.0
case("swiglu_clamped", n=n, limit=10.0, gate=t2l(gate), up=t2l(up),
     y=t2l(R.swiglu(gate, up, 10.0)))
case("swiglu_unclamped", n=n, limit=0.0, gate=t2l(gate * 0.05), up=t2l(up * 0.05),
     y=t2l(R.swiglu(gate * 0.05, up * 0.05, 0.0)))

# ---- router --------------------------------------------------------------
# A bias large enough to REORDER the top-k, so gathering weights from the biased
# scores instead of the unbiased ones changes the answer.
ne, topk = 32, 6
logits = torch.randn(1, ne) * 2.0
bias = torch.zeros(ne)
bias[torch.randperm(ne)[:8]] = 4.0
w, idx = R.route(logits, bias, None, None, topk, 2.5)
case("router_scored", n_experts=ne, topk=topk, route_scale=2.5,
     logits=t2l(logits), bias=t2l(bias),
     weights=t2l(w), indices=[int(v) for v in idx.flatten().tolist()])

# Large logits: softplus must use torch's threshold or these overflow to NaN.
big = torch.tensor([[200.0, 100.0, 40.0, 25.0, 1.0, 0.5, 0.2, 0.1]])
w2, i2 = R.route(big, None, None, None, 3, 1.0)
case("router_large_logits", n_experts=8, topk=3, route_scale=1.0,
     logits=t2l(big), bias=None,
     weights=t2l(w2), indices=[int(v) for v in i2.flatten().tolist()])

# ---- RoPE ----------------------------------------------------------------
# Both regimes: dense layers disable YaRN (theta 10000), compressed layers
# enable it (theta 160000, orig 65536).
for tag, orig, theta in (("dense", 0, 10000.0), ("yarn", 65536, 160000.0)):
    rd, pos = 64, 37
    fc = R.freqs_cis(rd, 128, orig, theta, 16.0, 32.0, 1.0)
    v = torch.randn(rd)
    case(f"rope_{tag}", rd=rd, pos=pos, x=t2l(v),
         y=t2l(R.apply_rope(v, fc[pos])),
         y_inv=t2l(R.apply_rope(R.apply_rope(v, fc[pos]), fc[pos], inverse=True)))

# ---- mHC -----------------------------------------------------------------
hc, iters, eps = 4, 20, 1e-6
mix = (2 + hc) * hc
mixes = torch.randn(mix) * 1.5
hc_scale = torch.randn(3) * 0.5 + 1.0
hc_base = torch.randn(mix) * 0.3
pre, post, comb = R.hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps)
case("sinkhorn", hc=hc, iters=iters, eps=eps, mixes=t2l(mixes),
     hc_scale=t2l(hc_scale), hc_base=t2l(hc_base),
     pre=t2l(pre), post=t2l(post), comb=t2l(comb))

# hc_post with an ASYMMETRIC comb, so the transposed contraction differs.
d = 8
x = torch.randn(d)
resid = torch.randn(hc, d)
case("hc_post", hc=hc, d=d, x=t2l(x), residual=t2l(resid),
     post=t2l(post), comb=t2l(comb),
     y=t2l(R.hc_post(x, resid, post, comb, hc)))

# ---- sparse attention ----------------------------------------------------
# Mixed sinks per head, and some masked slots, so both behaviours are exercised.
h, dd, npos, topk_n = 6, 16, 24, 12
q = torch.randn(h, dd)
kv = torch.randn(npos, dd)
sink = torch.tensor([-8.0, -2.0, 0.0, 1.0, 3.0, 8.0])
idxs = torch.tensor([0, 3, -1, 7, 11, -1, 2, 19, 5, 23, 9, -1], dtype=torch.long)
case("sparse_attn", h=h, d=dd, n=npos, topk=topk_n, scale=float(dd ** -0.5),
     q=t2l(q), kv=t2l(kv), sink=t2l(sink),
     idxs=[int(v) for v in idxs.tolist()],
     o=t2l(R.sparse_attn(q, kv, sink, idxs, dd ** -0.5)))

# ---- CSA indexer ---------------------------------------------------------
# Head weights of both signs, and q/kv that produce negative per-head scores, so
# the ReLU is actually load-bearing.
ih, ihd, nt = 8, 16, 20
iq = torch.randn(ih, ihd)
ikv = torch.randn(nt, ihd)
iw = torch.randn(ih)
case("indexer", n_heads=ih, head_dim=ihd, n_pos=nt,
     q=t2l(iq), kv=t2l(ikv), weights=t2l(iw),
     score=t2l(R.indexer_score(iq, ikv, iw, ih, ihd)))

# ---- compressor pooling --------------------------------------------------
slots, ch = 8, 12
ks = torch.randn(slots, ch)
ss = torch.randn(slots, ch) * 3.0
ss[5] = float("-inf")                      # an unfilled slot
case("compress_pool", slots=slots, ch=ch, kv_state=t2l(ks),
     score_state=[None if v == float("-inf") else v for v in t2l(ss)],
     out=t2l(R.compress_pool(ks, ss)))


def main():
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else "tests/fixtures/ref")
    out.mkdir(parents=True, exist_ok=True)
    meta = {"torch": torch.__version__, "cases": sorted(CASES)}
    (out / "MANIFEST.json").write_text(json.dumps(meta, indent=2) + "\n")
    for name, body in CASES.items():
        (out / f"{name}.json").write_text(json.dumps(body) + "\n")
        print(f"  {name}.json")
    print(f"\n{len(CASES)} fixtures from torch {torch.__version__}")


if __name__ == "__main__":
    main()
