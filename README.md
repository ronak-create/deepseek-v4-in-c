# deepseek-v4-in-c

DeepSeek-V4 in C99, running a checkpoint far larger than RAM by streaming it off
NVMe. Flash (284B parameters, ~160 GB on disk) runs on a laptop from **3.2 GB of
RAM**, and at **1.6 – 1.7 s/token** when given 18 GB and the GPU.

It is the instruct model, so it holds a conversation and writes code — see
[Chat and code](#chat-and-code).

The streaming design is ported from Fareed Khan's
[kimi-k3-in-c](https://github.com/FareedKhan-dev/kimi-k3-in-c) (Apache-2.0),
which does the same trick for Kimi K3. Roughly 40% of that repo transfers — the
safetensors reader, the trunk streamer, the expert cache, the memory planner.
The other 60% is new: DeepSeek-V4 shares none of K3's math, so every kernel was
written from DeepSeek's own reference implementation.

## Status

| | |
|---|---|
| DeepSeek-V4-Flash, CPU | ✅ 2.14 – 2.21 s/token |
| DeepSeek-V4-Flash, `--gpu` | ✅ 1.62 – 1.74 s/token |
| Smallest machine it runs on | ✅ **3.23 GB peak RSS** |
| Chat + tool-call encoding | ✅ `--chat`, stops at end-of-turn |
| DeepSeek-V4-Pro | ⚠️ planned and gated, never run — needs ~865 GB of disk |
| Gates | 21, all green (`make test`) |

## What makes it correct, not just plausible

Fluent output is weak evidence. A model with a swapped nibble or a misindexed
scale still produces confident-sounding text. So the engine is checked against
PyTorch at three levels:

- **Per kernel** — 14 kernels against a PyTorch reference, agreeing to 5e-7.
- **Per block** — three layer kinds over 46 positions, same tolerance.
- **Whole model** — a structurally real tiny `deepseek_v4` checkpoint (same
  tensor names, dtypes and config schema as the released one) run end to end
  through the *actual* loader, agreeing to **2.9e-6 with identical argmax at
  every position**.

The reference is a pure-PyTorch reimplementation written from DeepSeek's
`inference/model.py`, never from this C — otherwise both could agree on the same
misreading. The shipped `model.py` cannot serve directly: it imports a GPU
compiler (`tilelang`) and will not run on CPU.

**Bit-exactness is a contract, not an aspiration.** Scalar, OpenMP and AVX2 paths
produce byte-identical output, enforced by a fixed 16-accumulator tree and
`-ffp-contract=off`. The matmul gate checks this at runtime by running both
instantiations on the same data and comparing bit patterns — not a tolerance.
The GPU cannot join that club (its reduction order depends on warp geometry), so
it is gated separately on relative error plus argmax identity, and the CPU path
stays the reference.

## Build

```sh
make            # CPU only
make test       # the gate ladder
make bench      # matmul_bw, gpu_contention, cache_bw
```

`make test` runs 20 gates. The 21st, the trunk gate, needs a real checkpoint and
a packed trunk and skips without them:

```sh
DSV4_MODEL=~/models/dsv4-flash DSV4_TRUNK=~/dsv4-trunk make test
```

CUDA is auto-detected. With `nvcc` present it compiles the GPU path and links
`-lcudart`; without it, a stub is linked and `--gpu` reports that no device is
available. A CPU-only build is complete, not degraded.

Requires: a C99 compiler with OpenMP, AVX2 recommended. CUDA 12.8+ for `--gpu`
(sm_120 / Blackwell was the development target).

## Running a model

Three steps. The model directory alone is not enough — the trunk and tokenizer
have to be packed first.

```sh
# 1. pack the dense trunk into one contiguous file (Flash: ~6.3 GB)
python3 tools/pack_trunk.py ~/models/dsv4-flash ~/dsv4-trunk

# 2. pack the tokenizer (vocab 129,280)
python3 tools/pack_tokenizer.py ~/models/dsv4-flash ~/dsv4_tok.bin

# 3. run
./bin/dsv4 ~/models/dsv4-flash \
    --trunk ~/dsv4-trunk \
    --tok   ~/dsv4_tok.bin \
    --prompt "The capital of France is" \
    --gen 25 --budget 16 --gpu
```

### Options

| flag | meaning |
|---|---|
| `<dir>` or `--model DIR` | the checkpoint directory (`config.json` + shards) |
| `--trunk DIR` | packed trunk from `pack_trunk.py` |
| `--tok FILE` | packed tokenizer from `pack_tokenizer.py` |
| `--prompt TEXT` | prompt |
| `--gen N` | how many tokens to generate |
| `--budget GB` | total RAM for trunk + expert cache (default 8) |
| `--gpu` | put the dense trunk's FP8 matrices in VRAM (opt-in) |
| `--route-log FILE` | record every expert request, for `tools/sim_cache.py` |

### It scales down to 3.2 GB

Every row below produces **identical tokens** — the budget buys speed, never
correctness. Measured on Flash, same prompt:

| `--budget` | peak RAM | s/token | layers pinned |
|---|---|---|---|
| 1 GB | **3.23 GB** | 4.64 | 5 / 43 |
| 2 GB | 4.23 GB | 3.95 | 12 / 43 |
| 3 GB | 5.23 GB | 3.42 | 18 / 43 |
| 4 GB | 6.23 GB | 3.12 | 25 / 43 |
| 8 GB | 10.08 GB | 2.37 | 43 / 43 |
| 16 GB | 18.08 GB | 2.14 – 2.21 | 43 / 43 |
| 16 GB + `--gpu` | 18.19 GB | **1.62 – 1.74** | 43 / 43 |

Ranges are two runs of the same command; single figures are one run. These were
measured before [the expert cache stopped copying](#the-expert-cache-reads-straight-into-its-slots),
which is worth about 12% and is not yet reflected here — the machine has to go
cold again first, and mixing a warm number into a cold table is exactly the
mistake this page already made once. These are
short runs, so the fixed cost of filling a cold cache is spread over 26 tokens
— treat the small-budget rows as an upper bound on what a small machine pays.

Below 8 GB the trunk no longer pins and streams through ring slots instead; the
3 GB row exists to exercise that path, and it produces the same tokens as the
rows that pin all 43 layers.

### Choose `--budget` deliberately

This is the setting that matters most, and the default is not enough for Flash.

One forward pass touches `n_layers × topk` experts — **258 for Flash, 3.21 GB**.
Below that, the LRU cache **cannot hit at all**: every entry is evicted before
its layer comes round again, and the memory is wasted. Measured at the old
default: 10,320 requests, **0 hits**, 128 GB read from disk.

The budget covers the trunk *and* the cache. Flash's trunk is 6.27 GB, so:

| `--budget` | expert cache | hit rate | disk read |
|---|---|---|---|
| 8 GB | 1.6 GB | **0%** | 128 GB |
| 12 GB | 5.6 GB | ~40% | ~80 GB |
| **16 GB** | 9.6 GB | **53%** | 61 GB |

The engine prints a warning naming the threshold when you are under it.

## Chat and code

`DeepSeek-V4-Flash` is the **instruct** model — it does multi-turn chat, tool
calling and extended thinking. The prompt format is not in `tokenizer_config.json`
where most tools look; it lives in the checkpoint's own `encoding/encoding_dsv4.py`.
`--chat` applies it for you:

```sh
./bin/dsv4 ~/models/dsv4-flash --trunk ~/dsv4-trunk --tok ~/dsv4_tok.bin \
    --chat --system "You are a helpful coding assistant." \
    --prompt "Write a Python function that reverses a string. Reply with only the code." \
    --gen 120 --budget 16 --gpu
```

```
def reverse_string(s):
    return s[::-1]
```

It stopped there, at the end-of-turn token, in 12 tokens.

`--think` opens a `<think>` block instead, so the model reasons before
answering. Both stop at `<｜end▁of▁sentence｜>`, whose id is looked up from the
tokenizer rather than hardcoded.

**Two things the special tokens taught us.** `<｜User｜>` is a single id, and the
BPE merge table contains nothing that would rebuild it from pieces — so it has
to be matched *before* the regex pre-tokeniser sees it. It was not, and the
prompt above tokenised to 37 ids instead of 17. The model never saw a turn
boundary and answered by continuing the text: fluent, and not an assistant.
There is now a gate on it.

For a code-completion measurement, a binary search written by the model passed
**2000 random cases against Python's own `list.index`** plus the empty-list and
not-found edges. Routing over that run: 35,088 expert requests, all served, 46%
cache hits, with an 83.3% reuse ceiling — code routes more repetitively than
prose, so the cache pays off better on it.

Not included: a tool-calling loop. The model emits DSML tool blocks and the
format is documented in `encoding/README.md`, but driving them is a layer above
this CLI.

## Where the time goes

`--budget 16 --gpu`, 40 forward passes on Flash (15 prompt + 25 generated),
98% of wall clock accounted for:

```
routed experts (disk)   38.0%   17.2 s   NVMe — cannot be moved anywhere
attention               32.2%   14.6 s   FP8, on the GPU when --gpu
routed experts (matmul) 16.5%    7.5 s   FP4, CPU — streamed, so it stays here
shared expert            7.6%    3.4 s
mHC + norms              3.0%    1.4 s
router                   0.5%    0.2 s
```

Disk is the largest single line. Note that 15 of these 40 passes are prompt
tokens — see [Prefill is the expensive half](#prefill-is-the-expensive-half-and-it-is-not-batched).

The routed experts stay on the CPU on purpose: 137 GB, none of it resident,
streamed per token. Sending them across PCIe would add a hop to a disk read.

**If you use `--gpu`, one CPU core is reserved automatically**, and an explicit
`OMP_NUM_THREADS` overrides that. In a microbenchmark the effect is enormous —
a CPU-only FP4 matmul collapses 30x while the device is kept busy, and recovers
completely when a core is held back (see [The GPU needs a spare
core](#the-gpu-needs-a-spare-core)).

On the real model it is much smaller than that, and smaller than this README
previously claimed. Measured cold, over two runs each:

| `--budget 16 --gpu` | s/token |
|---|---|
| one core reserved (default) | **1.62 – 1.74** |
| `OMP_NUM_THREADS=20`, no reservation | 1.88 – 1.91 |

That is 10–15%, not the 142 s versus 47 s recorded in the heat-soaked session.
The microbenchmark keeps the device saturated back to back; the model only
touches it a few times per layer, so the pathology is real but rarely at full
strength. The reservation is still the right default — it never loses — but it
is a modest win, not a 3x one.

## The expert cache reads straight into its slots

`O_DIRECT` needs the file offset, the length and the destination buffer all
4096-aligned. Every expert tensor in the released checkpoint is unaligned — the
data section starts at a fixed odd offset — so reads used to be widened to the
enclosing window, landed in a staging buffer, and were then **memcpy'd** into
the cache slot. A 12.75 MB copy, 4,887 times a run.

The old comment here justified that as "~1 ms against the ~10 ms the unbuffered
read saves". The read is not 10 ms. Measured on this disk: a 12.75 MB O_DIRECT
read takes **2.81 ms**, while the engine was spending **3.60 ms** per expert.
The copy was not a rounding error, it was 22% of the cost.

Slots are now allocated 4096-aligned with a little slack, and each coalesced run
is placed at the offset whose alignment residue matches its file offset. The
widened window is then read *straight into the slot* and the payload is already
where it belongs. No copy at all.

Measured as an A/B, the two builds interleaved so disk and thermal drift hit
both equally, two runs each:

| | s/token | routed-expert disk | GB read | peak RSS |
|---|---|---|---|---|
| staging buffer + memcpy | 1.84, 1.82 | 17.6 s, 17.3 s | 60.85 | 18.19 GB |
| straight into the slot | **1.67, 1.55** | **15.2 s, 14.1 s** | 60.87 | 17.96 GB |

Identical token ids, the same hit rate and the same bytes read — so the
difference is the copy and nothing else. **Disk time −16%, wall clock −12%**,
and peak RSS is slightly *lower* because the slack costs one slot.

### It was wrong first, and the gates said it was fine

The first version of this placed each run at the next offset with a matching
residue, which is not sufficient: a window begins up to 4095 bytes *before* its
payload, so the second run's read reached back over the first run's bytes and
corrupted them. The damage was silent and it looked like a triumph — routing
collapsed onto a handful of experts, so the cache hit rate went from 52.6% to
**95.5%**, disk reads from 60.85 GB to 5.76 GB, and the model appeared to run at
1.06 s/token. Only the token ids gave it away.

Every cache gate passed throughout, for two compounding reasons, both now fixed:

- The gates compared the cache **against itself** — serial against concurrent —
  and both paths corrupted the bytes identically. There is now a gate that
  compares against a plain buffered `pread` of the same tensors, which shares no
  code with the fast path.
- The synthetic fixture stored an expert's six tensors **contiguously**, so an
  expert loaded as one run and the second-run case never arose. The real
  checkpoint splits them: three scales in one run, three weights ~341 MB away at
  a different residue. The fixture now mirrors that, and the new gate fails
  without the fix.

## Prefill is the expensive half, and it is not batched

Prompt tokens go through the engine **one at a time**, exactly like generated
ones. That is a deliberate gap, and this is what it costs. Two runs of the same
command differing only in prompt length, `--budget 16 --gpu`:

| prompt | forward passes | wall | |
|---|---|---|---|
| 15 tokens | 40 | 42.1 s | |
| 207 tokens | 232 | 234.8 s | |
| **marginal** | **192** | **192.7 s** | **1.00 s per prompt token** |

So a prompt token costs about 1.0 s against 1.62 s for a generated one. It is
cheaper only because the attention context is shorter — the weight traffic is
identical, because each token still does its own GEMV against every matrix.

That sets time-to-first-token at roughly **a second per prompt token**: ~3.5
minutes for a 207-token prompt, and over half an hour for a 2,000-token one.
For anything with a long prompt — code completion, an agent loop, retrieval —
this, not the generation rate, is the number that hurts.

### What a batched GEMM would actually buy

The fix is to run N prompt tokens through a layer together so each weight is
read once and used N times, lifting arithmetic intensity from ~1 flop/byte
toward N. How much reuse is really available is a property of the routing, so
it can be measured rather than guessed — replaying a `--route-log` and counting
requests against distinct experts per layer over the prefill window:

| prefill length | tokens sharing one expert weight load |
|---|---|
| 15 | 2.26 |
| 32 | 2.93 |
| 64 | 3.94 |
| 128 | 5.80 |
| 207 | **7.99** |

Routed experts are the awkward case: top-6 of 256 spreads tokens thin, so a
15-token prompt gets almost nothing (2.26x) and the reuse only becomes real
past ~100 tokens. The shared expert and the dense trunk matmuls have no such
problem — every token passes through the same weights, so they batch N-wide
whatever N is.

At the 207-token prompt measured above, batching would cut distinct expert
weight loads from 53,406 to 6,685. Combined with the ~4x fewer disk reads that
implies, and N-wide GEMMs on the trunk and shared expert, the accounted time
(disk 38.7%, attention 33.4%, expert matmul 15.4%, shared expert 7.4%) suggests
prefill could get roughly **2-3x faster on a long prompt, and ~15% on a
25-token one**. That is an estimate from the breakdown, not a measurement —
the only measurements here are the timings and the reuse factors above.

It is not a small change: it needs a batched variant of every matmul, batched
attention, and the compressor's `start_pos == 0` branch, and each has to keep
the same 16-accumulator tree per (row, token) pair to stay bit-exact.

## DeepSeek-V4-Pro

Planned and gated, but **never run** — it needs ~865 GB of checkpoint plus a
~50 GB trunk. `make test` includes a Pro gate that plans it from its real
`config.json` with no weights, and it passes. What that gate establishes:

| | Flash | Pro |
|---|---|---|
| layers | 43 | 61 |
| hidden | 4096 | 7168 |
| experts | 256, top-6 | 384, top-6 |
| one expert | 12.75 MiB | 33.47 MiB |
| **per-pass working set** | 3.21 GB | **11.96 GB** |

So Pro needs **~12 GB of expert cache before the cache can hit at all**, and its
~50 GB trunk will not pin in 23 GB of RAM — it will stream through ring slots.
`--gpu` will help far less, since only a fraction of a 50 GB trunk fits in 8 GB
of VRAM.

That gate is not ceremony. Until recently the binder validated `wo_a` as
`[o_lora*o_groups, hidden]`. The correct column count is
`n_heads*head_dim/o_groups`. For Flash those are the same number — 64·512/8 =
4096 = hidden — so the wrong rule passed on both released checkpoints. For Pro
they are 4096 against a hidden of 7168, and it would have failed on layer 0. A
Flash-only test suite could not have caught it.

## Measured on

Every number in this README came from one machine, stated so the numbers can be
argued with:

| | |
|---|---|
| CPU | 20-core x86-64, AVX2 + FMA, no AVX-512 |
| GPU | RTX 5060 Laptop, 8 GB, sm_120 (Blackwell), CUDA 13.3 |
| Disk | PCIe Gen4 NVMe, 5.3 GB/s sequential O_DIRECT when cold |
| OS | WSL2 / Ubuntu 24.04, 23 GB available to the VM |

**Thermal state matters far more than previously claimed, and it is the
biggest caveat on this page.** Every figure here was re-measured on a freshly
rebooted, idle machine on AC power in performance mode. An earlier revision of
this README carried the same benchmarks taken after 24+ hours of sustained
load, and the gap is not the ±20% that revision quoted:

| | heat-soaked | cold | |
|---|---|---|---|
| `--budget 1` | 13.7 s/tok | **4.64 s/tok** | 2.95x |
| `--budget 16` | 2.65 s/tok | **2.21 s/tok** | 1.20x |
| `--budget 16 --gpu` | 1.81 s/tok | **1.74 s/tok** | 1.04x |

The pattern is consistent: the more disk-bound a configuration is, the more it
loses when the machine is hot. Sequential O_DIRECT reads measure 5.3 GB/s on the
cold machine, against 4.4 GB/s recorded during the earlier heat-soaked session,
and the low-budget rows are almost entirely disk. So treat any single timing here as ±20% *at a fixed thermal state*, and
up to 3x across states — which is why kernel changes are resolved with
`bench/matmul_bw.c` rather than by timing a generation.

## Kernel throughput

`bench/matmul_bw.c`, best of 15, at Flash's real matrix shapes. "bitwise equal"
is the scalar path against the AVX2 path — not a tolerance:

| kernel | scalar | AVX2 | speedup | bitwise equal |
|---|---|---|---|---|
| fp4 expert w1 2048x4096 | 45.0 GF/s | **108.8 GF/s** | 2.42x | yes |
| fp4 expert w2 4096x2048 | 52.2 GF/s | **115.4 GF/s** | 2.21x | yes |
| fp8 attn wo_b 4096x8192 | 14.9 GF/s | 19.7 GF/s | 1.32x | yes |
| fp8 attn wq_b 32768x1024 | 15.1 GF/s | 19.1 GF/s | 1.26x | yes |

FP8 barely moving is the informative part: those matrices are 33.5 MB and are
re-read from RAM every layer, so they are bandwidth-bound rather than
conversion-bound. That is the measurement that argues for putting the trunk in
VRAM instead of optimising it further on the CPU.

Per-call, including the PCIe round trip:

| | GPU | CPU (AVX2) |
|---|---|---|
| fp8 wo_b 4096x8192 | **1.348 ms** | 3.401 ms |
| fp8 wq_b 32768x1024 | **1.341 ms** | 3.522 ms |

There is a ~1.3 ms floor per call that barely varies with size, so the GPU wins
on the 33.5 MB matrices and loses on small ones — `wkv` is 4.2 MFLOP, which the
CPU finishes in 0.23 ms.

## The GPU needs a spare core

`bench/gpu_contention.c` times the same CPU-only FP4 matmul with the device idle
and busy:

| | CPU throughput |
|---|---|
| device idle | 119.1 GF/s |
| device busy, 20 OpenMP threads | **3.9 GF/s** — 30.5x collapse |
| device busy, 20 threads, blocking sync | 5.0 GF/s — 25.1x, no better |
| device busy, one core reserved | 118.5 GF/s — 1.00x, no contention |

The fix is real and the explanation was wrong. This bench was written to test
the hypothesis that CUDA's default synchronisation *spins*, so the waiting
thread competes with the OpenMP workers it is waiting behind. Selecting
`cudaDeviceScheduleBlockingSync` — where the thread sleeps instead — should then
have removed the collapse. It does not: 5.0 GF/s against 3.9, a 25.1x collapse
instead of 30.5x. So spinning is at most a small part of it, and the real cause
is still open; DMA traffic contending for DRAM bandwidth with a memory-bound FP4
matmul is the next suspect, unmeasured.

What *is* measured is that holding one core back removes the collapse entirely,
whichever sync mode is in use. `--gpu` reserves a core automatically; an
explicit `OMP_NUM_THREADS` is honoured instead.

## Expert routing

From a real code-completion run (`--route-log` + `tools/sim_cache.py`), 136
forward passes:

| | |
|---|---|
| expert requests | 35,088, all served |
| distinct experts used | 5,861 of 11,008 (43 layers x 256) |
| hottest expert | fired in 118 of 136 passes |
| used exactly once | 27% — can never be a cache hit |
| reuse ceiling (infinite cache) | **83.3%** |

| cache | LRU | frequency-pinned | Belady (bound) |
|---|---|---|---|
| 9.6 GB | 45.9% | 49.1% | 67.5% |
| 16 GB | 57.1% | 60.3% | 75.4% |
| 24 GB | 67.7% | 68.6% | 79.7% |

Frequency-pinning beats LRU by ~3 points and loses at small sizes, which is why
the policy stays LRU. These are cache-simulation results replayed from a stored
trace, so unlike every timing on this page they do not move with thermal state —
they are properties of the routing, not of the machine. Code routes more repetitively than prose — the same
measurement on an English prompt gives a 69.9% ceiling against 83.3% here — so
the cache pays off better on coding work.

## Benchmarks and diagnostics

```sh
make bench                 # builds all three
./bin/matmul_bw 15         # scalar vs AVX2 vs GPU, at real geometries
./bin/gpu_contention spin  # CPU throughput while the device is busy
./bin/gpu_contention block # the same with blocking sync, for comparison
./bin/cache_bw             # expert read throughput, cold
python3 tools/sim_cache.py route.log   # replay a routing trace vs LRU/LFU/Belady
```

`cache_bw` is the one that explains the low-`--budget` rows: a 12.75 MB expert
reads in **4.8 ms at 2.6 GB/s**, against 5.3 GB/s for the trunk's long
sequential reads. One forward pass needs 258 of them.

`matmul_bw` exists because the model run is a poor instrument: it takes ~70 s
and moves ±20% with disk and thermal state, which cannot resolve a 12% kernel
change.

## Things that were tried and did NOT work

Recorded so they are not rebuilt from the same wrong premise.

- **Frequency-pinning the expert cache.** Beats LRU by ~4 points at best and
  loses at small sizes. Plain LRU is right; the bug was the budget split.
- **8-wide SIMD FP4 dequant.** 81.5 GF/s against 80.9 for plain table lookups.
  FP4 was never dequant-bound — the win was hoisting the activation widening out
  of the row loop (2.8×).
- **Layer-major prefill.** Bit-exact, and 73.6 s against a 66–70 s baseline. The
  trunk is fully pinned so there is no disk read to save, and 165 MB per layer
  dwarfs the ~24 MB L3, so there is no cache reuse either. (Both figures are
  from the heat-soaked session, so they compare with each other but not with
  the cold timings elsewhere on this page. The conclusion is a ratio, and the
  ratio is what survives.)

## Licence

Apache-2.0. Portions derived from `kimi-k3-in-c`, also Apache-2.0.
