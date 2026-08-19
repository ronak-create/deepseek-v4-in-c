#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Replay a routing trace against several expert-cache policies.

WHY THIS EXISTS
    A measured run of DeepSeek-V4-Flash reported 2,580 expert requests and
    ZERO hits from a 1.99 GB LRU cache. That number is not noise and it is not
    bad luck. One forward pass touches 43 layers x 6 experts = 258 experts, or
    3.29 GB; the cache held 160. So between two visits to layer 0 the engine
    loads 252 other experts and evicts everything it will next ask for. It is
    the same cyclic-access pathology the trunk already avoids by pinning a
    prefix instead of running LRU.

    Whether a bigger or smarter cache helps depends entirely on how much
    routing REPEATS across tokens, which is a property of the model and its
    input, not something to be reasoned about from first principles. So:
    capture the trace, replay it, and let the numbers pick the policy.

USAGE
    dsv4 ... --route-log ~/route.log
    python3 tools/sim_cache.py ~/route.log [--budgets 2,4,8,16,24]

POLICIES
    lru       what ships today
    lfu-pin   pin the globally hottest experts, LRU the remainder
    belady    evict whatever is needed furthest in the future -- unattainable,
              but it bounds what any policy could achieve on this trace
"""
import argparse
import collections
import sys

EXPERT_BYTES = 13_369_344
GB = 1 << 30


def load(path):
    seq = []
    with open(path) as f:
        for line in f:
            p = line.split()
            if len(p) == 2:
                seq.append((int(p[0]), int(p[1])))
    return seq


def sim_lru(seq, nslot):
    cache, hits = collections.OrderedDict(), 0
    for k in seq:
        if k in cache:
            cache.move_to_end(k)
            hits += 1
        else:
            if len(cache) >= nslot:
                cache.popitem(last=False)
            cache[k] = 1
    return hits


def sim_lfu_pin(seq, nslot, pin_frac=0.5):
    """Pin the hottest experts outright, run LRU over what is left.

    The pinned set is chosen from the trace's own frequencies, which is
    cheating in the same way Belady cheats. It is reported to show whether
    skew is exploitable AT ALL. A shippable version would either learn the
    set from a warmup or take it from an offline profile -- routing skew in
    an MoE is a property of the trained weights and largely input-agnostic,
    which is exactly what a second trace can confirm.
    """
    freq = collections.Counter(seq)
    npin = int(nslot * pin_frac)
    pinned = {k for k, _ in freq.most_common(npin)}
    nlru, hits = nslot - len(pinned), 0
    cache = collections.OrderedDict()
    for k in seq:
        if k in pinned:
            hits += 1                      # after the first load; corrected below
        elif k in cache:
            cache.move_to_end(k)
            hits += 1
        else:
            if len(cache) >= nlru and nlru > 0:
                cache.popitem(last=False)
            if nlru > 0:
                cache[k] = 1
    return hits - len(pinned & set(seq))    # each pinned expert is read once


def sim_belady(seq, nslot):
    nxt = collections.defaultdict(collections.deque)
    for i, k in enumerate(seq):
        nxt[k].append(i)
    cache, hits = set(), 0
    for i, k in enumerate(seq):
        nxt[k].popleft()
        if k in cache:
            hits += 1
            continue
        if len(cache) >= nslot:
            # evict the resident whose next use is furthest away
            worst, at = None, -1
            for r in cache:
                d = nxt[r][0] if nxt[r] else float("inf")
                if d > at:
                    at, worst = d, r
            cache.discard(worst)
        cache.add(k)
    return hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--budgets", default="2,4,8,16,24",
                    help="cache sizes in GB")
    a = ap.parse_args()

    seq = load(a.trace)
    if not seq:
        sys.exit("empty trace")
    layers = len({l for l, _ in seq})
    per_pass = collections.Counter(l for l, _ in seq)
    npass = max(per_pass.values()) // 6 if per_pass else 0
    uniq = len(set(seq))

    print("trace: %d requests, %d distinct (layer,expert), %d layers, ~%d passes"
          % (len(seq), uniq, layers, npass))
    print("       one pass touches %d experts = %.2f GB"
          % (len(seq) // max(npass, 1),
             len(seq) / max(npass, 1) * EXPERT_BYTES / GB))
    print("       the whole trace's working set is %.2f GB"
          % (uniq * EXPERT_BYTES / GB))

    # How much reuse is even on the table? An expert seen once can never hit.
    freq = collections.Counter(seq)
    reusable = sum(v - 1 for v in freq.values())
    print("       reuse ceiling: %d of %d requests could hit (%.1f%%) -- an "
          "infinite cache" % (reusable, len(seq), 100.0 * reusable / len(seq)))
    top = freq.most_common(5)
    print("       hottest: %s" % ", ".join("L%d.e%d x%d" % (k[0], k[1], v)
                                           for k, v in top))
    print()

    hdr = "%8s %7s | %-18s %-18s %-18s" % (
        "budget", "slots", "lru", "lfu-pin", "belady (bound)")
    print(hdr)
    print("-" * len(hdr))
    for g in [float(x) for x in a.budgets.split(",")]:
        nslot = int(g * GB / EXPERT_BYTES)
        if nslot < 1:
            continue
        row = []
        for fn in (sim_lru, sim_lfu_pin, sim_belady):
            h = fn(seq, nslot)
            row.append("%6.1f%%  %5.1f GB" %
                       (100.0 * h / len(seq),
                        (len(seq) - h) * EXPERT_BYTES / GB))
        print("%6.1f GB %7d | %-18s %-18s %-18s" % (g, nslot, *row))


if __name__ == "__main__":
    main()
