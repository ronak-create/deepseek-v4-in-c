# Refinement plan from the Reddit feedback (r/LocalLLM, r/DeepSeek, r/OpenSourceAI)

Three launch threads, ~47K views, ~35 comments. This is what came back that is
worth acting on, what is already answered, and what should be declined.

Ground rule, unchanged from the README: **nothing here gets a performance claim
without an interleaved cold A/B**. Two of the items below exist precisely
because a plausible story was believed before it was measured.

| thread | comments with signal |
|---|---|
| [r/LocalLLM](https://www.reddit.com/r/LocalLLM/comments/1vuh471/) | colibri comparison; "will it do 5 t/s on my box" |
| [r/DeepSeek](https://www.reddit.com/r/DeepSeek/comments/1vuhwyi/) | expert prediction; NVMe striping; SSD wear x2; AMX/ACE; Qwen x2 |
| [r/OpenSourceAI](https://www.reddit.com/r/OpenSourceAI/comments/1vuidqb/) | Apple IFP; async prefetch/double buffering; KTransformers; "why not a 27B" |

---

## The single most important finding: colibri exists and is ahead

`Chips_fr_` linked [JustVugg/colibri](https://github.com/JustVugg/colibri) —
pure C, Apache-2.0, same bet (stream MoE experts off NVMe), and it **already
supports DeepSeek-V4-Flash** (167 GB container). It has shipped several things
this repo lists as future work:

| colibri has | this repo |
|---|---|
| router lookahead prefetch, "71.6% predictable" one layer ahead | declared impossible in `src/cache/dsv4_cache.h:143` |
| async I/O pool (loads misses while residents compute) | hard barrier, `src/model/dsv4_layer.c:392-401` |
| batch-union reads across batch positions | prefill is one token at a time |
| dual-SSD striping, bandwidth-weighted | single drive |
| O_DIRECT | have it |

Their published Flash number is ~1.6 tok/s (RTX 5080, 2 NVMe, 3k ctx) against
~0.6 tok/s here. **Different rigs, so it is not yet a comparison** — which is
the point of item 1.

The 71.6% figure directly contradicts the header comment in this repo. That
comment is right about *exact* prefetch and wrong as written about
*speculative* prefetch. Fixing it is item 5.

---

## Tier 1 — cheap, and it answers the questions people actually asked

### 1. A head-to-head against colibri, run properly
**From:** `Chips_fr_`. Already publicly committed to in-thread.

Same machine, same checkpoint, same prompt, same context length, cold boot,
both engines. Report tok/s, TTFT, peak RSS, GB read from disk, and token ids.
Publish it whichever way it falls — if colibri is faster, that is the more
useful post, and their prefetch design becomes the roadmap.

*Verify:* a `BENCHMARKS.md` carrying the exact commands, drive model, and two
runs each, interleaved. Do not compare a cold run against a warm one.

### 2. "What will I get on my hardware" — make it self-service
**From:** `Prize-Cut-9651`, `BoboThePirate`, `mrgreatheart`, `desexmachina` —
asked four separate times, answered by hand each time.

Add `tools/estimate.py`: takes RAM budget, VRAM, NVMe random-read GB/s and PCIe
gen, and prints an estimated s/token by scaling the measured breakdown in
[Where the time goes](README.md#where-the-time-goes) (disk 38%, attention 32%,
expert matmul 16.5%) against the hit-rate curve already in
[Choose `--budget`](README.md#choose---budget-deliberately). Label the output an
estimate from a model, not a measurement.

*Verify:* feed it this machine's specs and check it reproduces 1.62–1.74
s/token within ~15%.

### 3. A "when NOT to use this" section
**From:** `Revolutionary_Loan13` ("2 tokens a second is nonsensically slow"),
`prime-rick` and `LinuXperia` (both asked for Qwen-27B), `mrgreatheart` (asked
what beats llama.cpp on 48 GB VRAM).

Every one of these is the same misread: people think this competes with
llama.cpp. It does not — it competes with *not being able to run the model at
all*. Put a decision table near the top of the README:

| your situation | use |
|---|---|
| model fits in RAM/VRAM | llama.cpp — streaming is pure overhead |
| model fits across VRAM+RAM combined | KTransformers (~4x on decode) |
| model does not fit at all, and you want bit-exact output | this |

*Verify:* n/a, documentation.

### 4. An SSD-wear FAQ entry
**From:** `SlckOvrfl`, `Sea_Ear5201`, plus `Right_Simple_6813` answering it
half-right in-thread.

Answer it once, with this repo's own numbers: 60.87 GB read per 40-pass run,
**zero writes** on the inference path, TBW is a write rating, and the residual
mechanism is read-disturb-triggered internal relocation. State the size of that
effect only if it can be sourced or measured — otherwise say it is small and
unquantified. This engine's whole reputation is not overstating things.

*Verify:* `smartctl -A` before and after a run, reporting host writes and NAND
writes. That turns the FAQ into a measurement instead of an argument.

### 5. Correct the cross-layer prefetch comment
**From:** the colibri 71.6% figure against `src/cache/dsv4_cache.h:143`.

Rewrite it to separate three genuinely different cases:
- **scored layers, exact prefetch** — impossible, layer L+1 needs L's output;
- **hash layers (0–2 of 43), exact prefetch** — *possible*, `tid2eid` makes the
  experts a pure function of the token id (item 7);
- **speculative prefetch** — possible, costs correctness nothing, costs
  bandwidth on a miss (item 10).

*Verify:* n/a, but it must land before items 7 and 10 so the code and the
comment never disagree.

---

## Tier 2 — engine work with a bounded, measurable payoff

### 6. Overlap expert compute with expert I/O
**From:** `desexmachina` — "have you tried RAM as an intermediate streamer". The
honest answer given in-thread was that there is no double buffering: a miss
blocks until the read finishes.

That is exactly what `src/model/dsv4_layer.c:392-401` does — `get_many()` reads
all six experts, *then* the k-loop runs all six matmuls. Disk is 38.0% of wall
clock and expert matmul is 16.5%, strictly serialised. Overlapping them can hide
at most the smaller of the two, so the **ceiling is ~16% of wall clock** and the
realistic figure is lower, since a cache hit has nothing to overlap against.

Design that keeps bit-exactness: give each k its own accumulator (`s->expert_acc`
becomes `topk` buffers), let `dsv4_cache_get_many()` publish slots as each read
completes, run `expert_forward` for whichever k has landed, and **sum into `out`
in k order at the end**. Each expert's matmul is independent; only the weighted
sum is order-sensitive, and that stays serial. The 16-accumulator tree per row is
untouched.

*Verify:* the existing bit-exactness gates plus identical token ids against the
current build on the same prompt; then an interleaved cold A/B on s/token and on
`dsv4_prof.expert_io`.

### 7. Exact prefetch for the hash layers
**From:** the `tid2eid` exception this repo already documents but never uses.

Layers 0–2 route by static token-id lookup. The moment a token is sampled, the
experts for the *next* token's first three layers are known exactly — no
speculation. Issue those 18 reads (~230 MB) during sampling and the tail of the
current token.

Honest sizing: 3 of 43 layers, so ~7% of expert traffic, and only the misses
count. Small. But it is exact, it can never evict something it will not use, and
it is the cheapest possible test of the async machinery item 6 builds.

*Verify:* hit rate on layers 0–2 should reach ~100% after the first token; tokens
must be identical; A/B the wall clock and expect low single digits.

### 8. Find out whether the read path is actually queue-depth-bound
**From:** `ProfessionalJackals` — "buy a ton of NVMes and parallel read".

The in-thread answer had the right instinct: the limit is probably requests in
flight, not drive bandwidth. `src/cache/dsv4_cache.h:139` already records 2.0
GB/s in situ against 4.4 GB/s in the benchmark, at a queue depth of at most six.
**Measure before building anything:** extend `bench/cache_bw.c` to sweep queue
depth 1 → 32 on this drive.

- If throughput keeps climbing past 6, the fix is `io_uring` with a submission
  ring instead of OpenMP-parallel `pread`, and striping is a *second* step that
  only pays once the queue is deep enough to saturate one drive.
- If it plateaus at 6, striping is the only lever and io_uring buys nothing.
  Record that as a negative result and stop.

*Verify:* the QD sweep is the deliverable. Only then decide.

### 9. Prefill batching — still the biggest single win
**From:** the r/OpenSourceAI post's own contribution ask; unchanged by comments.

Already fully specified in
[What a batched GEMM would actually buy](README.md#what-a-batched-gemm-would-actually-buy):
reuse factor 7.99x at a 207-token prompt, 53,406 → 6,685 distinct expert loads,
estimated 2–3x on long-prompt prefill. At ~1.0 s per prompt token, this is the
number that makes the engine unusable for agent loops and code completion.

colibri's "batch-union reads" is the same idea already shipped, which is
corroboration that the work is worth doing — and item 1 will show how much it
buys in practice.

*Verify:* bit-exactness gates must hold with N > 1 (same 16-accumulator tree per
(row, token) pair); TTFT measured on the 15/207-token prompt pair already in the
README, so before/after is directly comparable.

---

## Tier 3 — research, only after Tier 2, and only if measured first

### 10. Speculative cross-token expert prediction
**From:** `Genericinquirer` (twice, including "train a small model for it"),
corroborated by colibri's 71.6% one-layer-ahead claim.

The objection raised in-thread is real and stands: DeepSeek's training spreads
load across experts on purpose, and a wrong guess both wastes a read and can
evict something useful. Both halves are testable **without touching the engine**,
using the route logs and `tools/sim_cache.py` that already exist:

1. From a stored `--route-log`, measure P(expert set at L+1 | expert set at L)
   and P(same expert at L on token t+1 | token t). If it is nowhere near 71.6%
   on this model, stop — that is a publishable negative result and it costs a
   day.
2. If it is high, prefetch into **dedicated non-evicting slots**, so a wrong
   guess can never displace a live entry. That removes the second objection
   entirely and reduces the downside to wasted bandwidth.

A trained predictor is out of scope. A frequency/recency table replayed through
`sim_cache.py` is the cheap version and answers the same question.

### 11. Per-prompt expert pruning (opt-in, explicitly non-exact)
**From:** `MatiAI` — pointed at Apple's on-device MoE and Instruction-Following
Pruning, which decides routing *per prompt* rather than per token because
NAND → DRAM bandwidth cannot sustain per-token expert swaps.

This is the one comment proposing a different point on the tradeoff curve rather
than an optimisation. It would be a `--prune-per-prompt` flag that selects an
expert working set once and runs against it: much faster, and **not bit-exact**,
which is this repo's entire claim to trust.

If it is built: separate flag, separate README section, output quality measured
rather than asserted, and every existing exactness claim scoped explicitly to the
default path. It must be impossible to read a benchmark table and not know which
mode produced it. Note it also composes with item 9 — a prefill union of experts
is the same computation, minus the approximation.

---

## Declined, with the reason recorded

- **AVX-512 / AMX / ACE** (`desexmachina`, `ProfessionalJackals`). ACE is real,
  spec locked, silicon ~2028. Irrelevant either way: disk is 38% and expert
  matmul 16.5%, so the matmuls are not the wall. Worth one README line so the
  suggestion is not re-litigated. AVX-512 would still be worth *measuring* if a
  machine with it ever shows up — as a kernel-throughput data point, not as a
  hoped-for speedup.
- **Porting to Qwen-27B / Qwen 3.x** (`prime-rick`, `LinuXperia`,
  `Revolutionary_Loan13` — the most-repeated request across all three threads).
  A 27B dense model in 12–16 GB is a solved problem and streaming is pure
  overhead there; that answer was already given in-thread and it is correct.
  What *would* be legitimate is a large Qwen **MoE** checkpoint that does not
  fit — but only after item 1 says whether this engine is the right vehicle at
  all. Interim deliverable: a short "porting to another MoE" note pointing at
  `include/dsv4/dsv4_cfg.h` and `src/model/dsv4_bind.c`, not a port.
- **Training a predictor from scratch, or baking prediction into the model**
  (`Genericinquirer`). Data-centre scale. Item 10 is the solo-feasible version.

---

## Suggested order

1. Items 3, 4, 5 — one afternoon of documentation, and they retire most of the
   repeated questions.
2. Item 1 — the colibri head-to-head. It reorders everything below it, so it goes
   early even though it produces no code.
3. Item 8 — the queue-depth sweep. Also code-free, and it decides whether
   io_uring or striping is worth building at all.
4. Items 6, then 7 — the async machinery, cheapest exact user first.
5. Item 2 — the estimator, once 6 and 7 have changed the numbers it models.
6. Item 9 — prefill batching, the big one.
7. Items 10, 11 — only with data from item 8 and a decision from item 1.
