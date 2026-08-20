# deepseek-v4-in-c

DeepSeek-V4 in C99, running a checkpoint far larger than RAM by streaming it off
NVMe. Flash (284B parameters, ~160 GB on disk) runs on a laptop from **3.2 GB of
RAM**, and at **1.81 s/token** when given 18 GB and the GPU.

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
| DeepSeek-V4-Flash, CPU | ✅ 2.65 s/token |
| DeepSeek-V4-Flash, `--gpu` | ✅ 1.81 s/token |
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
make test       # the full gate ladder
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
| 1 GB | **3.23 GB** | 13.7 | 5 / 43 |
| 2 GB | 4.23 GB | 12.6 | 12 / 43 |
| 4 GB | 6.23 GB | 11.6 | 25 / 43 |
| 8 GB | 10.08 GB | 9.3 | 43 / 43 |
| 16 GB + `--gpu` | 18.2 GB | **1.81** | 43 / 43 |

(The slower rows are short runs, so cold-cache cost is spread over fewer
tokens; treat them as an upper bound on what a small machine pays.)

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

`--budget 16 --gpu`, 40 forward passes on Flash:

```
routed experts (disk)   27%   NVMe — cannot be moved anywhere
attention               33%   FP8, on the GPU when --gpu
routed experts (matmul) 25%   FP4, CPU — streamed, so it stays here
shared expert            7%
mHC + norms              2%
```

The routed experts stay on the CPU on purpose: 137 GB, none of it resident,
streamed per token. Sending them across PCIe would add a hop to a disk read.

**If you use `--gpu`, one CPU core is reserved automatically.** CUDA's default
sync spins, and a spinning thread on a pool with one OpenMP worker per core
competes with the threads it is waiting behind. Measured: CPU-side matmul
collapses from 114 GF/s to 5.0 GF/s (21.7×) — and returns to 0.99× the moment a
core is held back. On the real model that was 142 s versus 47 s.

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
| Disk | PCIe Gen4 NVMe, 4.4 GB/s sequential O_DIRECT |
| OS | WSL2 / Ubuntu 24.04, 23 GB available to the VM |

Two caveats worth applying to every figure. Run-to-run variance on the full
model is about **20%**, dominated by disk and thermal state — which is why
kernel changes are measured with `bench/matmul_bw.c` rather than by timing a
generation. And this is a laptop: the numbers here were taken on AC power in
performance mode, after hours of sustained load, so they are if anything
pessimistic.

## Kernel throughput

`bench/matmul_bw.c`, best of 15, at Flash's real matrix shapes. "bitwise equal"
is the scalar path against the AVX2 path — not a tolerance:

| kernel | scalar | AVX2 | speedup | bitwise equal |
|---|---|---|---|---|
| fp4 expert w1 2048x4096 | 36.3 GF/s | **103.5 GF/s** | 2.85x | yes |
| fp4 expert w2 4096x2048 | 41.5 GF/s | **114.3 GF/s** | 2.76x | yes |
| fp8 attn wo_b 4096x8192 | 14.3 GF/s | 18.1 GF/s | 1.27x | yes |
| fp8 attn wq_b 32768x1024 | 13.7 GF/s | 19.4 GF/s | 1.41x | yes |

FP8 barely moving is the informative part: those matrices are 33.5 MB and are
re-read from RAM every layer, so they are bandwidth-bound rather than
conversion-bound. That is the measurement that argues for putting the trunk in
VRAM instead of optimising it further on the CPU.

Per-call, including the PCIe round trip:

| | GPU | CPU (AVX2) |
|---|---|---|
| fp8 wo_b 4096x8192 | **1.50 ms** | 3.89 ms |
| fp8 wq_b 32768x1024 | **1.34 ms** | 3.40 ms |

There is a ~1.3 ms floor per call that barely varies with size, so the GPU wins
on the 33.5 MB matrices and loses on small ones — `wkv` is 4.2 MFLOP, which the
CPU finishes in 0.23 ms.

## The GPU needs a spare core

`bench/gpu_contention.c` times the same CPU-only FP4 matmul with the device idle
and busy:

| | CPU throughput |
|---|---|
| device idle | 114.2 GF/s |
| device busy, 20 OpenMP threads | **5.0 GF/s** — 21.7x collapse |
| device busy, one core reserved | 113.8 GF/s — no contention |

CUDA's default synchronisation spins, and a spinning thread on a pool that
already has one worker per core competes with the threads it waits behind. On
the real model that was 142 s versus 47 s. `--gpu` reserves a core
automatically; an explicit `OMP_NUM_THREADS` is honoured instead.

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
the policy stays LRU. Code routes more repetitively than prose — the same
measurement on an English prompt gives a 69.9% ceiling against 83.3% here — so
the cache pays off better on coding work.

## Benchmarks and diagnostics

```sh
./bin/matmul_bw 15        # scalar vs AVX2 vs GPU, at real geometries
./bin/gpu_contention spin # what a spinning CUDA thread does to OpenMP
./bin/cache_bw            # expert read throughput, cold
python3 tools/sim_cache.py route.log   # replay a routing trace vs LRU/LFU/Belady
```

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
  dwarfs the ~24 MB L3, so there is no cache reuse either.

## Licence

Apache-2.0. Portions derived from `kimi-k3-in-c`, also Apache-2.0.
