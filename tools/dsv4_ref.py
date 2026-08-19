#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Pure-PyTorch reference for the DeepSeek-V4 kernels.

WHY A REIMPLEMENTATION AND NOT A WRAPPER
    The plan was to wrap the checkpoint's own inference/model.py. It does not
    import on CPU: model.py -> kernel.py -> tilelang, a GPU compiler
    (requirements.txt pins tilelang==0.1.8, plus fast_hadamard_transform).

    So the six tilelang kernels are reimplemented here in plain torch. That is
    more work than wrapping and it is a STRONGER check: an independent
    reimplementation catches a misreading of the reference by disagreeing with
    the C, whereas a wrapper would share the misreading silently.

THE ONE RULE FOR THIS FILE
    Written from inference/model.py and inference/kernel.py, NEVER from the C in
    src/. If a C header records a trap, that is a hint about where to look in the
    Python, not a source to copy from. Two implementations that agree because one
    was transcribed from the other prove nothing.

WHAT IS DELIBERATELY NOT MODELLED
    act_quant / fp4_act_quant, the FP8 and FP4 activation simulation. They are
    quantisation-aware-training emulation: they perturb values slightly and can
    reorder positions at a top-k boundary. Modelling them would mean matching
    tilelang's rounding exactly, which is a bigger job than the thing being
    tested. Gates that could be affected say so.

    rotate_activation (Hadamard) is also skipped: it is orthogonal and applied to
    both sides of every inner product it touches, so it cancels in exact
    arithmetic. It exists to spread outliers before FP4 quantisation.
"""
import math

import torch
import torch.nn.functional as F


# ---------------------------------------------------------------- RMSNorm ---
def rmsnorm(x, weight, eps):
    """model.py RMSNorm.forward. eps is added to the MEAN of squares."""
    x = x.float()
    var = x.square().mean(-1, keepdim=True)
    return weight * (x * torch.rsqrt(var + eps))


# ----------------------------------------------------------- SwiGLU/clamp ---
def swiglu(gate, up, limit):
    """model.py Expert.forward.

    The clamp is ASYMMETRIC and that is not a typo in the reference:
        up   = clamp(up, min=-limit, max=+limit)
        gate = clamp(gate,           max=+limit)
    gate has no lower bound.
    """
    gate = gate.float()
    up = up.float()
    if limit > 0:
        up = torch.clamp(up, min=-limit, max=limit)
        gate = torch.clamp(gate, max=limit)
    return F.silu(gate) * up


# ------------------------------------------------------------------ router ---
def route(logits, bias, tid2eid, token_ids, topk, route_scale,
          score_func="sqrtsoftplus"):
    """model.py Gate.forward.

    Returns (weights, indices). The bias shifts scores for SELECTION only;
    original_scores is captured before it is added and is what the weights are
    gathered from.
    """
    scores = logits.float()
    if score_func == "softmax":
        scores = scores.softmax(dim=-1)
    elif score_func == "sigmoid":
        scores = scores.sigmoid()
    else:
        scores = F.softplus(scores).sqrt()

    original_scores = scores
    if bias is not None:
        scores = scores + bias

    if tid2eid is not None:
        indices = tid2eid[token_ids]
    else:
        indices = scores.topk(topk, dim=-1)[1]

    weights = original_scores.gather(1, indices)
    if score_func != "softmax":
        weights = weights / weights.sum(dim=-1, keepdim=True)
    return weights * route_scale, indices


# --------------------------------------------------------------- YaRN RoPE ---
def freqs_cis(dim, seqlen, original_seq_len, base, factor, beta_fast, beta_slow):
    """model.py precompute_freqs_cis."""
    def correction_dim(num_rot):
        return dim * math.log(original_seq_len / (num_rot * 2 * math.pi)) \
               / (2 * math.log(base))

    freqs = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
    if original_seq_len > 0:
        low = max(math.floor(correction_dim(beta_fast)), 0)
        high = min(math.ceil(correction_dim(beta_slow)), dim - 1)
        if low == high:
            high += 0.001
        ramp = (torch.arange(dim // 2, dtype=torch.float32) - low) / (high - low)
        smooth = 1 - torch.clamp(ramp, 0, 1)
        freqs = freqs / factor * (1 - smooth) + freqs * smooth

    t = torch.arange(seqlen, dtype=torch.float32)
    return torch.polar(torch.ones(seqlen, dim // 2), torch.outer(t, freqs))


def apply_rope(x, fc, inverse=False):
    """model.py apply_rotary_emb. Pairs are INTERLEAVED: view_as_complex over
    the last axis unflattened to (-1, 2), i.e. (x[0],x[1]), (x[2],x[3]), ..."""
    c = torch.view_as_complex(x.float().reshape(*x.shape[:-1], -1, 2))
    if inverse:
        fc = fc.conj()
    return torch.view_as_real(c * fc).flatten(-2)


# ------------------------------------------------------------------- mHC ----
def hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, eps):
    """kernel.py hc_split_sinkhorn_kernel.

    Note the loop shape: the FIRST pass is a row softmax, +eps, then a COLUMN
    normalise; the remaining iters-1 passes are row-then-column.
    """
    pre = torch.sigmoid(mixes[..., :hc] * hc_scale[0] + hc_base[:hc]) + eps
    post = 2 * torch.sigmoid(mixes[..., hc:2 * hc] * hc_scale[1] + hc_base[hc:2 * hc])
    comb = (mixes[..., 2 * hc:] * hc_scale[2] + hc_base[2 * hc:]).reshape(
        *mixes.shape[:-1], hc, hc)

    comb = comb.softmax(dim=-1) + eps
    comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        comb = comb / (comb.sum(dim=-1, keepdim=True) + eps)
        comb = comb / (comb.sum(dim=-2, keepdim=True) + eps)
    return pre, post, comb


def hc_pre(x, hc_fn, hc_scale, hc_base, hc, norm_eps, hc_eps, iters):
    """model.py Block.hc_pre. x is [hc, d]; the rsqrt spans the WHOLE flattened
    hc*d vector and uses norm_eps, not hc_eps."""
    shape = x.shape
    flat = x.reshape(-1).float()
    rsqrt = torch.rsqrt(flat.square().mean(-1, keepdim=True) + norm_eps)
    mixes = (hc_fn @ flat) * rsqrt
    pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc, iters, hc_eps)
    y = (pre.unsqueeze(-1) * x.reshape(shape)).sum(dim=0)
    return y, post, comb


def hc_post(x, residual, post, comb, hc):
    """model.py Block.hc_post.

        y = post[...,None]*x[...,None,:] + sum over dim -3 of
            comb[...,:,:,None] * residual[...,:,None,:]

    The contraction is over comb's FIRST index, so y[k] gathers comb[j][k].
    """
    return post.unsqueeze(-1) * x.unsqueeze(-2) \
         + (comb.unsqueeze(-1) * residual.unsqueeze(-2)).sum(dim=-3)


# -------------------------------------------------------- sparse attention ---
def sparse_attn(q, kv, attn_sink, topk_idxs, scale):
    """kernel.py sparse_attn_kernel, in plain (non-blocked) form.

    q [h, d]; kv [n, d] shared by every head; topk_idxs holds indices into kv
    with -1 for masked. The sink enters the DENOMINATOR only, and is not part of
    the running max.
    """
    h, d = q.shape
    out = torch.zeros(h, d, dtype=torch.float32)
    valid = topk_idxs >= 0
    idx = topk_idxs.clamp(min=0)
    gathered = kv[idx].float()                      # [topk, d]

    for i in range(h):
        s = (q[i].float() * gathered).sum(-1) * scale
        s = torch.where(valid, s, torch.full_like(s, float("-inf")))
        mx = s.max()
        if torch.isinf(mx) and mx < 0:
            e = torch.zeros_like(s)
        else:
            e = torch.exp(s - mx)
            e = torch.where(valid, e, torch.zeros_like(e))
        denom = e.sum() + torch.exp(attn_sink[i].float() - mx)
        out[i] = (e.unsqueeze(-1) * gathered).sum(0) / denom
    return out


# ----------------------------------------------------------- CSA indexer ----
def indexer_score(q, kv, weights, n_heads, head_dim):
    """model.py Indexer.forward scoring tail.

        weights    = weights_proj(x) * (head_dim^-0.5 * n_heads^-0.5)
        score      = einsum(bshd,btd->bsht)(q, kv)
        score      = (score.relu() * weights[..., None]).sum(dim=2)

    The ReLU sits BEFORE the head weighting.
    """
    scale = (head_dim ** -0.5) * (n_heads ** -0.5)
    s = torch.einsum("hd,td->ht", q.float(), kv.float())
    return (s.relu() * (weights.float() * scale).unsqueeze(-1)).sum(dim=0)


# ------------------------------------------------------------- compressor ---
def compress_pool(kv_state, score_state):
    """model.py Compressor: (kv_state * score_state.softmax(dim=slot)).sum(slot).

    The softmax runs over the SLOT axis, independently for every channel.
    """
    return (kv_state.float() * score_state.float().softmax(dim=0)).sum(dim=0)


# ==================================================================== block ==
# model.py Block.forward / Attention.forward / MoE.forward, decode path,
# compress_ratio 0 (no compressor, no indexer).
#
# Weights are bfloat16 so the C sees byte-identical values: the fixture carries
# the raw bf16 bit patterns and both sides widen them the same way. Anything
# else would compare two different models.

def attention_dense(x, W, cfg, kv_cache, fc, pos):
    """One token through a compress_ratio 0 attention block."""
    hd, rd, H = cfg["head_dim"], cfg["qk_rope"], cfg["n_heads"]
    eps = cfg["rms_eps"]

    qr = rmsnorm(x @ W["wq_a"].T.float(), W["q_norm"], eps)
    q = (qr @ W["wq_b"].T.float()).reshape(H, hd)
    # SECOND normalisation, per head and UNWEIGHTED -- not q_norm again.
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    q = torch.cat([q[:, :hd - rd], apply_rope(q[:, hd - rd:], fc[pos])], dim=-1)

    kv = rmsnorm(x @ W["wkv"].T.float(), W["kv_norm"], eps)
    kv = torch.cat([kv[:hd - rd], apply_rope(kv[hd - rd:], fc[pos])], dim=-1)
    kv_cache[pos % cfg["window"]] = kv

    n = min(pos + 1, cfg["window"])
    idxs = torch.tensor([(pos - k) % cfg["window"] for k in range(n)]
                        + [-1] * (cfg["window"] - n), dtype=torch.long)
    o = sparse_attn(q, kv_cache, W["sink"], idxs, hd ** -0.5)

    # de-rotate, then the GROUPED low-rank output projection
    o = torch.cat([o[:, :hd - rd],
                   apply_rope(o[:, hd - rd:], fc[pos], inverse=True)], dim=-1)
    g, gr = cfg["o_groups"], cfg["o_lora"]
    gw = H * hd // g
    o = o.reshape(g, gw)
    wo_a = W["wo_a"].float().reshape(g, gr, gw)
    o = torch.einsum("gd,grd->gr", o, wo_a).reshape(-1)
    return o @ W["wo_b"].T.float()


def moe_dense(x, W, cfg, token_id):
    """One token through the MoE half. The shared expert is added UNWEIGHTED
    after the routed sum."""
    logits = (x.unsqueeze(0) @ W["gate"].T.float())
    w, idx = route(logits, W.get("gate_bias"), W.get("tid2eid"),
                   torch.tensor([token_id]), cfg["topk"], cfg["route_scale"])
    y = torch.zeros(cfg["hidden"])
    for k in range(cfg["topk"]):
        e = int(idx[0, k])
        g_ = x @ W[f"e{e}_w1"].T.float()
        u_ = x @ W[f"e{e}_w3"].T.float()
        y = y + float(w[0, k]) * (swiglu(g_, u_, cfg["swiglu_limit"])
                                  @ W[f"e{e}_w2"].T.float())
    gs = x @ W["sh_w1"].T.float()
    us = x @ W["sh_w3"].T.float()
    y = y + swiglu(gs, us, cfg["swiglu_limit"]) @ W["sh_w2"].T.float()
    return y


def block_dense(h, W, cfg, kv_cache, fc, pos, token_id):
    """model.py Block.forward. h is [hc, hidden].

    Note the SECOND residual is taken after the attention half, not at the top.
    """
    hc, d = cfg["hc_mult"], cfg["hidden"]
    it, ne, he = cfg["sinkhorn_iters"], cfg["rms_eps"], cfg["hc_eps"]

    residual = h
    x, post, comb = hc_pre(h, W["hc_attn_fn"].float(), W["hc_attn_scale"],
                           W["hc_attn_base"], hc, ne, he, it)
    x = rmsnorm(x, W["attn_norm"], ne)
    x = attention_dense(x, W, cfg, kv_cache, fc, pos)
    h = hc_post(x, residual, post, comb, hc)

    residual = h
    x, post, comb = hc_pre(h, W["hc_ffn_fn"].float(), W["hc_ffn_scale"],
                           W["hc_ffn_base"], hc, ne, he, it)
    x = rmsnorm(x, W["ffn_norm"], ne)
    x = moe_dense(x, W, cfg, token_id)
    return hc_post(x, residual, post, comb, hc)


# ======================================================= compressed block ====
# model.py Compressor.forward, start_pos > 0 branch, plus the ratio != 0
# attention path. Written from that branch; a token-at-a-time loop starting at
# position 0 agrees with the seqlen==1 prefill branch, because there
# should_compress is `seqlen >= ratio` (false for one token) and the state write
# lands in the same slot.

class CompressorState:
    """The compressor's open window. score_state starts at -inf so an unfilled
    slot contributes nothing to the pooling softmax; zeros would give it uniform
    weight instead."""

    def __init__(self, ratio, head_dim):
        self.ratio = ratio
        self.overlap = (ratio == 4)
        self.coff = 1 + self.overlap
        w = self.coff * head_dim
        n = self.coff * ratio
        self.kv = torch.zeros(n, w)
        self.score = torch.full((n, w), float("-inf"))
        self.d = head_dim


def compress_step(cs, kv, score, ape, pos):
    """One decode step. Returns the pooled row, or None while the window is open."""
    r, d, ov = cs.ratio, cs.d, cs.overlap
    phase = pos % r
    score = score + ape[phase]

    slot = (r + phase) if ov else phase
    cs.kv[slot] = kv
    cs.score[slot] = score

    if (pos + 1) % r != 0:
        return None

    if ov:
        # first `ratio` slots contribute their FIRST d channels (the previous
        # window), the next `ratio` their SECOND d (the current one)
        k_sel = torch.cat([cs.kv[:r, :d], cs.kv[r:, d:]], dim=0)
        s_sel = torch.cat([cs.score[:r, :d], cs.score[r:, d:]], dim=0)
        out = (k_sel * s_sel.softmax(dim=0)).sum(dim=0)
        cs.kv[:r] = cs.kv[r:].clone()
        cs.score[:r] = cs.score[r:].clone()
    else:
        out = (cs.kv * cs.score.softmax(dim=0)).sum(dim=0)
    return out


def attention_compressed(x, W, cfg, cache, cs, fc, pos, n_comp):
    """compress_ratio != 0 attention. `cache` is window-first, compressed-after."""
    hd, rd, H = cfg["head_dim"], cfg["qk_rope"], cfg["n_heads"]
    eps, win, ratio = cfg["rms_eps"], cfg["window"], cfg["ratio"]

    qr = rmsnorm(x @ W["wq_a"].T.float(), W["q_norm"], eps)
    q = (qr @ W["wq_b"].T.float()).reshape(H, hd)
    q = q * torch.rsqrt(q.square().mean(-1, keepdim=True) + eps)
    q = torch.cat([q[:, :hd - rd], apply_rope(q[:, hd - rd:], fc[pos])], dim=-1)

    kv = rmsnorm(x @ W["wkv"].T.float(), W["kv_norm"], eps)
    kv = torch.cat([kv[:hd - rd], apply_rope(kv[hd - rd:], fc[pos])], dim=-1)
    cache[pos % win] = kv

    # the compressor sees the same normed x attention does
    ck = x @ W["c_wkv"].T.float()
    csc = x @ W["c_wgate"].T.float()
    pooled = compress_step(cs, ck, csc, W["c_ape"], pos)
    if pooled is not None:
        pooled = rmsnorm(pooled, W["c_norm"], eps)
        # stamped with the FIRST position of the window it summarises
        pooled = torch.cat([pooled[:hd - rd],
                            apply_rope(pooled[hd - rd:], fc[pos + 1 - ratio])], dim=-1)
        cache[win + n_comp[0]] = pooled
        n_comp[0] += 1

    n = min(pos + 1, win)
    idxs = [(pos - k) % win for k in range(n)] + [-1] * (win - n)
    idxs += [win + i for i in range(n_comp[0])]
    o = sparse_attn(q, cache, W["sink"], torch.tensor(idxs, dtype=torch.long),
                    hd ** -0.5)

    o = torch.cat([o[:, :hd - rd],
                   apply_rope(o[:, hd - rd:], fc[pos], inverse=True)], dim=-1)
    g, gr = cfg["o_groups"], cfg["o_lora"]
    gw = H * hd // g
    wo_a = W["wo_a"].float().reshape(g, gr, gw)
    o = torch.einsum("gd,grd->gr", o.reshape(g, gw), wo_a).reshape(-1)
    return o @ W["wo_b"].T.float()


def block_compressed(h, W, cfg, cache, cs, fc, pos, token_id, n_comp):
    hc = cfg["hc_mult"]
    it, ne, he = cfg["sinkhorn_iters"], cfg["rms_eps"], cfg["hc_eps"]

    residual = h
    x, post, comb = hc_pre(h, W["hc_attn_fn"].float(), W["hc_attn_scale"],
                           W["hc_attn_base"], hc, ne, he, it)
    x = rmsnorm(x, W["attn_norm"], ne)
    x = attention_compressed(x, W, cfg, cache, cs, fc, pos, n_comp)
    h = hc_post(x, residual, post, comb, hc)

    residual = h
    x, post, comb = hc_pre(h, W["hc_ffn_fn"].float(), W["hc_ffn_scale"],
                           W["hc_ffn_base"], hc, ne, he, it)
    x = rmsnorm(x, W["ffn_norm"], ne)
    x = moe_dense(x, W, cfg, token_id)
    return hc_post(x, residual, post, comb, hc)
