#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Ask whether DeepSeek-V4's expert routing is PREDICTABLE from the previous token.

WHY THIS EXISTS
    colibri reports "71.6% predictable one layer ahead" and ships a prefetcher
    on the strength of it. REFINEMENT_PLAN.md item 10 says the same idea is
    worth testing here before any engine code is written, because the objection
    raised in-thread is real: DeepSeek trains for load balance across experts on
    purpose, so routing may simply not repeat. That is a property of the model
    and the input, not something to reason about from first principles.

    This costs no engine change. Replay a route log and ask directly: if, while
    running token t, you prefetched exactly what token t-1 used, how much of
    token t's traffic would you have caught?

WHY THE PREDICTOR IS "PREVIOUS TOKEN, SAME LAYER"
    The router at layer L needs the hidden state after layer L-1, so a token's
    own route cannot be known ahead of itself. But the PREVIOUS token's full
    route is known before the current token starts -- all 43 layers of it. The
    expert cache holds 767 slots against the 258 experts one pass touches, so
    there is room to prefetch an entire predicted route. Lead time is therefore
    not the constraint; accuracy is. Hence this measurement.

    Note prefetching does NOT reduce bytes read. It moves a read off the
    critical path. The number that matters is therefore the share of MISSES it
    could have covered, not the share of requests.

USAGE
    dsv4 ... --batch 1 --route-log ~/route.log
    python3 tools/predict_route.py ~/route.log [--budgets 8,16,24]
"""
import argparse
import collections

EXPERT_BYTES = 13_369_344
GB = 1 << 30


def load_tokens(path):
    """Route log is 'layer expert' per request. A token is one full sweep of
    layers 0..n-1, so a new token starts whenever the layer index goes back
    down. Returns [ {layer: [experts]} ] in token order."""
    toks, cur, last = [], collections.defaultdict(list), -1
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) != 2:
                continue
            L, e = int(p[0]), int(p[1])
            if L < last:
                toks.append(cur)
                cur = collections.defaultdict(list)
            cur[L].append(e)
            last = L
    if cur:
        toks.append(cur)
    return toks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--budgets", default="8,16,24")
    a = ap.parse_args()

    toks = load_tokens(a.log)
    nlayer = max(max(t.keys()) for t in toks) + 1
    print("%d tokens, %d layers, %d requests"
          % (len(toks), nlayer, sum(len(v) for t in toks for v in t.values())))

    # ---- 1. raw predictability, ignoring the cache entirely ----------------
    # How much of token t's expert set at layer L also appears in token t-1's
    # set at the same layer? This is the ceiling on a previous-token predictor.
    for k in (1, 2, 4):
        hit = tot = 0
        per_layer = collections.defaultdict(lambda: [0, 0])
        for i in range(k, len(toks)):
            pred = collections.defaultdict(set)
            for j in range(1, k + 1):
                for L, es in toks[i - j].items():
                    pred[L] |= set(es)
            for L, es in toks[i].items():
                for e in es:
                    tot += 1
                    per_layer[L][1] += 1
                    if e in pred[L]:
                        hit += 1
                        per_layer[L][0] += 1
        cost = sum(len(v) for i in range(k, len(toks))
                   for v in [set().union(*[set(toks[i - j].get(L, []))
                                           for j in range(1, k + 1)])
                             for L in toks[i]])
        print("\nunion of last %d token(s): %.1f%% of requests predicted "
              "(%d/%d), prefetch set %.1f experts/layer"
              % (k, 100.0 * hit / tot, hit, tot,
                 cost / float(len(toks) - k) / nlayer))
        if k == 1:
            worst = sorted(per_layer.items(),
                           key=lambda kv: kv[1][0] / float(kv[1][1]))
            print("  worst layers: " + ", ".join(
                "L%d %.0f%%" % (L, 100.0 * c[0] / c[1]) for L, c in worst[:5]))
            print("  best  layers: " + ", ".join(
                "L%d %.0f%%" % (L, 100.0 * c[0] / c[1]) for L, c in worst[-5:]))

    # ---- 2. the number that actually matters: share of MISSES covered ------
    # Replay an LRU of the real size. A prediction is only worth anything when
    # the request would otherwise have missed.
    for gb in [float(x) for x in a.budgets.split(",")]:
        slots = int(gb * GB / EXPERT_BYTES)
        cache, clock = collections.OrderedDict(), 0
        req = hits = miss = pred_miss = 0
        for i, t in enumerate(toks):
            pred = {}
            if i > 0:
                for L, es in toks[i - 1].items():
                    pred[L] = set(es)
            for L in sorted(t.keys()):
                for e in t[L]:
                    clock += 1
                    req += 1
                    key = (L, e)
                    if key in cache:
                        hits += 1
                        cache.move_to_end(key)
                        continue
                    miss += 1
                    if e in pred.get(L, ()):
                        pred_miss += 1
                    cache[key] = clock
                    if len(cache) > slots:
                        cache.popitem(last=False)
        print("\n--budget %g GB  (%d slots)" % (gb, slots))
        print("  %d requests, %d hits (%.1f%%), %d misses"
              % (req, hits, 100.0 * hits / req, miss))
        print("  of those misses, %d (%.1f%%) were in the previous token's set "
              "at the same layer" % (pred_miss, 100.0 * pred_miss / miss))
        print("  -> upper bound on what a previous-token prefetcher could hide: "
              "%.1f%% of expert I/O" % (100.0 * pred_miss / miss))


if __name__ == "__main__":
    main()
