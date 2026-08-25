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

### 6. Overlap expert compute with expert I/O — DONE
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

**Done 2026-08-25.** Built as designed: `dsv4_cache_get_many` split into
`begin`/`end`, misses handed to a pthread reader pool, one accumulator per `k`,
weighted sum still in `k` order. 21/21 gates green plus a new cache gate 8 for
the contract `begin()` adds; identical token ids and identical
hit/miss/eviction/bytes-read counts in every A/B run.

Measured over nine interleaved runs at `--budget 16 --gpu`, alternating which
build ran first: exposed I/O **-34%** (523.6 → 346.2 ms/pass), wall clock
**-10%** (1.45 → 1.31 s/token mean), and the spread cut from 1.13–1.86 down to
1.19–1.44. Below the estimated 16% ceiling, for a measured reason: a third
build isolating the per-`k` refactor from the overlap shows the refactor costs
nothing, while the overlap costs **+34% on the expert matmul** — NVMe DMA
contending with a memory-bound FP4 matmul. That is the same mechanism the README
names as the unmeasured suspect behind the GPU contention collapse, now observed
with no GPU in the picture.

**A narrower reader pool does not buy the contention back**, though the
argument that it should is a good one. Five interleaved rounds varying
`DSV4_READERS`: exposed I/O 667.9 / 488.5 / 418.3 / 375.4 / 351.3 ms/pass at
1 / 2 / 3 / 4 / 6 readers, and s/token 1.72 / 1.54 / 1.40 / 1.39 / 1.39.
Widest wins. The QD sweep measures *throughput* over 48 reads; what `end()`
waits for is the *makespan* of a batch of six against a fixed amount of
hit-matmul. Halving the readers triples the rounds without making the drive
busier. Recorded so the argument is not made again from scratch.

That same experiment sent me back to item 8's sweep, which collected regions
layer-major — near-contiguous, unlike the scattered reads real routing issues.
The sweep now shuffles by default. **The negative result survives**: shuffled
peaks at QD4 rather than QD2 and is equally flat to 32, so io_uring still buys
nothing. The ordered curve alone would have overstated how little depth is
worth, and `ordered` reproduces it for comparison.

**And it does nothing at `--budget 4`**, where the hit rate is exactly 0% so
there are no residents to compute during the reads. Measured, identical both
ways. The payoff is proportional to the hit rate — which is the argument for
item 7, since exact hash-layer prefetch manufactures overlappable work on
precisely the configurations that have none.

### 7. Exact prefetch for the hash layers — SIZED, NOT BUILT
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

**Sized 2026-08-25 from a real route log, and the sizing above was too
pessimistic for an interesting reason.** Replaying a `--budget 16 --gpu` trace
through an LRU of the engine's own 767 slots reproduces its 49.9% hit rate
exactly, and then splits it:

| | requests | hits | hit rate | misses |
|---|---|---|---|---|
| hash layers 0–2 | 360 | 60 | **16.7%** | 300 |
| scored layers 3–42 | 4,800 | 2,517 | 52.4% | 2,283 |

**The hash layers have three times the miss rate of the scored ones.** `tid2eid`
routing is a function of the token id, so it spreads across the vocabulary and
has far less reuse than content-based routing, which concentrates. So they are
7% of requests but **11.6% of all misses** — the plan's "~7% of expert traffic,
small" understated it.

Ceiling: removing 300 of 2,583 misses is 11.6% of expert I/O, which at 33% of
wall clock is **~3.9% of the wall clock**, and only if the moved reads land in
idle drive time (there is plenty: the drive is busy a third of the run).

Two versions, and they are not the same job:

- **Layer 0 only** — issue its `begin()` at sampling time, before the embedding
  and layer 0's attention, and let its MoE call `end()`. One batch, no new
  machinery, no reserved slots, and eviction between issue and use is
  impossible because nothing else runs in between. Worth ~1.3%.
- **All three layers** — needs more than one batch in flight in the reader pool,
  and at low budget it needs non-evicting reserved slots, because at
  `--budget 4` the LRU evicts everything before its layer comes round. Worth
  the full ~3.9%, at maybe a day.

**~3.9% is below what this machine can resolve** without a large number of
interleaved runs — identical builds swing ±20% with thermal and disk state. So
this is a decision, not an obvious next step: the work is well understood and
the payoff is real but small.

### 8. Find out whether the read path is actually queue-depth-bound — DONE, NEGATIVE
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

**Done 2026-08-25.** `cache_bw <model> qd` sweeps depth 1–32 with expert-sized
O_DIRECT reads over distinct regions of the real checkpoint. Three consecutive
runs agree to within 2%:

| QD | 1 | 2 | 3 | 4 | 6 | 8 | 12 | 16 | 24 | 32 |
|---|---|---|---|---|---|---|---|---|---|---|
| GB/s | 3.95 | **4.98** | 4.56 | 4.66 | 4.97 | 4.81 | 4.94 | 4.70 | 4.61 | 4.77 |

It plateaus at QD2. Concurrency is worth ~25% and the existing six-wide
`get_many` already collects all of it. **So the second branch above is the one
that holds: io_uring buys nothing, and striping is the only lever.** Recorded as
a negative result; neither is being built.

The sweep also retired the number that motivated the question. `dsv4_cache.h`
claimed 2.0 GB/s in situ against 4.4 benchmarked. Since the cache started
reading straight into its slot (`f185fee`), a live run moves 32.16 GB of experts
in 7.8–10.5 s = 3.1–4.1 GB/s, i.e. 75–95% of the drive's ceiling at any depth.
There is no large in-situ gap left. That also weakens item 6's expected payoff:
overlap can only hide I/O behind compute, and the I/O is no longer the
underperforming half.

### 9. Prefill batching — DONE, and worth 9%, not the 2–3x estimated here
**From:** the r/OpenSourceAI post's own contribution ask; unchanged by comments.

**Shipped** in `857f43e` (batched matmuls) and `7e50219` (end to end).
`--batch N`, default 32, bit-exact against the per-token path and gated by the
whole-model oracle at several chunk sizes.

**The estimate this section used to carry is retracted.** It projected 2–3x on
long-prompt prefill from the time breakdown. Measured on a 53-token prompt at
`--budget 16`, CPU only, interleaved and with both orderings run: **9%**, TTFT
81.7/85.8 s → 73.6/78.0 s. The GEMMs delivered exactly what `bench/gemm_bw.c`
priced them at — routed expert matmul −45%, shared expert −35% — and an I/O
regression ate most of it.

**Why: the union eats its own cache hits.** Deduplicating the batch's expert
requests removes precisely the repeat requests that were the cache's hits. Hit
rate by batch size, and these counters are bit-identical run to run:

    batch      1      2      4      8     16     32     64
    hits    50.1%  40.5%  29.2%   1.8%   2.1%   0.9%   0.9%

The cliff between 4 and 8 is exact: batch 4's per-layer union still fits the
767-slot / 9.55 GB cache and batch 8's does not. But **the cost cancels** —
batch 4 runs matmul 23.3–29.2 s against disk 9.3–10.0 s, batch 32 runs matmul
17.8–18.8 s against disk 15.0–15.2 s. Batching past 4 buys ~5 s of matmul and
pays ~5 s of exposed I/O for it, which is why TTFT is flat from 4 to 32 and why
the default was left at 32.

**Two claims made during this work did not survive re-measurement, and are
withdrawn:**

- A **+13% attention regression** on identical unbatched work, asserted in
  `7e50219`'s message. It does not reproduce: at matched thermal state batch-32
  attention is 43.1 s against batch-1's 41.1/45.3 s. The original figure came
  from comparing two arms measured at different thermal states. What may still
  be real, but rests on single runs, is attention rising above batch 8 —
  34.8 / 37.6 / 43.1 / 52.5 s at nt = 8 / 16 / 32 / 64.
- **Batch 4 as the optimum**, from a sweep showing 78.1 s against 84.3 s at 32.
  An interleaved A/B gave 32 → 80.5, 4 → 80.5, 32 → 85.4, 4 → 93.4 s: TTFT rose
  with position in the sequence, not with batch size.

**The method rule both of those earned:** on this machine the route and cache
counters are deterministic and can be trusted from a single run, but any timing
difference under ~10% needs interleaving *and* both orderings — and even then a
monotonic drift across the sequence can manufacture a trend. Restating it
because three separate conclusions died to it in one afternoon.

**Also fixed here:** `test_model_oracle` built its expert source with `.get`
alone, so neither the per-token overlap nor the batched one was ever reached by
the gate that claims to protect them. It now wires `begin`/`end`.

**Left open:** `dsv4_mmq_n` has no CUDA branch, so `--gpu` plus batching
silently runs every FP8 matmul on the CPU. And the attention half is still
per-token inside the batch — 46% of batched TTFT, and the only remaining 2x.

---

## Tier 3 — research, only after Tier 2, and only if measured first

### 10. Speculative cross-token expert prediction — MEASURED, and it is DEAD
**From:** `Genericinquirer` (twice, including "train a small model for it"),
corroborated by colibri's 71.6% one-layer-ahead claim.

**Tested with `tools/predict_route.py` on a real 65-token route log, no engine
change, exactly as this section proposed. It cannot help.**

The previous token's route predicts **36.2%** of the next token's expert
requests (44.7% from the last two tokens, 52.6% from the last four). That part
of colibri's premise holds — routing does repeat.

It buys nothing anyway, and the trace shows why with an exact identity:

    slots   hits            predictable misses
      240   0      ( 0.0%)  5978  (35.6%)
      281   5978  (35.6%)   0     ( 0.0%)

One forward pass touches 43 x 6 = **258 experts**. Below that, LRU holds nothing
and every prediction is useful but unaffordable. Above it, **every expert the
previous token used is still resident when the next token asks for it** — so
the 5978 predictable requests are *precisely* the 5978 cache hits. Prefetching
them re-fetches what the cache already has.

There is no budget where this wins. Below 258 slots the prefetched entry is
evicted before use; above it LRU already captured everything the predictor
knows. The misses that remain at `--budget 16` (41.6% of requests) are experts
that appear in *none* of the last four tokens — genuinely novel routing, which
no history-based predictor can reach by construction.

**Consequence for the declined item:** training a predictor from scratch was
already declined on cost. It is now declined on a stronger ground — a
history-based predictor has no headroom to recover, so a learned one would have
to beat the router by modelling the hidden state, not by learning usage
patterns. That is a research project, not an optimisation.

This also confirms the objection raised in-thread: DeepSeek balances expert load
across the vocabulary on purpose, and what repetition survives is already the
thing LRU is good at.

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
6. Item 9 — prefill batching. DONE, and it was not the big one: 9%, not 2-3x.
7. Items 10, 11 — only with data from item 8 and a decision from item 1.
