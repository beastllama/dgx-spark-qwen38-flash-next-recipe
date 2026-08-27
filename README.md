# Qwen3.8-Flash-Next NVFP4 on 2× DGX Spark — field notes

Serving **Qwen3.8-Flash-Next-NVFP4** across **two DGX Sparks (GB10 / sm_121)** with SGLang, TP=2
over ConnectX-7 RoCE (**single rail** — dual-rail is measured fabric capability, untested
under SGLang). Vision enabled. 262,144 context. MTP speculative decoding.

**Start here, then read the findings.** The base stack is MiaAI-Lab's — this repo is the config
delta on top of it plus everything that went wrong getting there and how it was fixed
(see [Credit](#credit)). What follows was *not* in either published recipe:
repeated node wedges (on unified memory, exhaustion does not error — it takes the whole box), a deadlock that only appears behind a default-deny
firewall, the first speculative-decode acceptance measurements we're aware of for this model, and three
optimisation avenues that turned out to be **dead ends** — documented as such, with numbers.

Everything here was measured on real hardware. Where a number is contested or unproven, it says so.

---

## Status

| | |
|---|---|
| Serving | ✅ TP=2 across 2 nodes, 262,144 context |
| Vision | ✅ verified end-to-end (see below) |
| Spec decode | ✅ NEXTN 3/1/4 (`num_steps`/`eagle_topk`/`num_draft_tokens`) — **the architectural maximum, not a default** |
| Decode | **~63 tok/s** single-stream on real generation |
| Concurrency | **306 tok/s** aggregate at 6 streams |
| Prefill | 3,050 tok/s (cache defeated) |
| Thermals | 52 °C / 35 W peak under load |

<sub>Conditions, because this repo insists on them: **decode ~63 tok/s** = real generation of a
10.7k-token HTML file, thinking off, temp 0.3, 2398 MHz idle / 2522 under load. **306 tok/s** =
aggregate across 6 concurrent streams, code prompt, 400 `max_tokens`, `ignore_eos`. **Prefill
3,050 tok/s** = ~7,450-token unique prompt per run, `cached_tokens=0` asserted, n=6. **Thermals** =
concurrency 4, 1 Hz sampling. A number without its prompt, token count and clock state is not
comparable to anything — including these.</sub>

---

## Performance

Two DGX Sparks. 180B params (125B backbone + 51B PLE), NVFP4, 262k context, vision on.

| | tok/s | conditions |
|---|---|---|
| **Single stream, real work** | **~63** | 10.7k-token HTML page, natural stop, 2522 MHz |
| Single stream, 400 tok | 63.7 | code prompt, `ignore_eos` |
| 2 concurrent | **104.9** agg | 52.8/stream |
| 4 concurrent | **178.7** agg | 45.2/stream |
| **6 concurrent** | **306.6** agg | 51.7/stream — still climbing |
| Prefill | **3,050** | ~7,450-token unique prompt, `cached_tokens=0` asserted, n=6 |
| Stress floor | 47.6 | `ignore_eos` + hard prompt + 800 tok — a deliberate FLOOR, see below |

**Power and heat, at concurrency 4:** **52 °C, 35.5 W** peak per node; 42 °C / 10.4 W idle. That is
roughly **half the draw** of a comparably-sized dense-ish MoE we previously ran on the same boxes
(88 °C / 65 W), at *higher* clocks. Cause: ~6B active params per token (10 of 512 experts) and only
12 of 48 layers are full attention — the rest are linear-attention GDN, so decode waits on memory
rather than burning watts. Practical effect: thermal guard stages sized for the older model are
unreachable by 43 °C, and two Sparks serve this at **~71 W combined under load**.

**Why two numbers for "single stream".** `ignore_eos` benchmarks force generation past the model's
natural stopping point into degenerate text. They're excellent for regression detection and useless
as a headline. Real generation of a complete HTML page runs at **~63 tok/s**; the same stack under
`ignore_eos` on a hard prompt reports **47.6**. Both are correct. Quote the one that matches what
you're doing, and say which.

**Speculative decoding is doing the heavy lifting.** The same stack without MTP measured ~20–21
tok/s. Turning it on is worth roughly **3×** — see §3 for why 3/1/4 is the ceiling and how close the
drafter gets to it.

---

## Recipe

**1. Base stack.** Clone [MiaAI-Lab's repo](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)
and follow it — it builds the SM121 QSA patch onto the public SGLang image and handles fabric
preflight, worker-first ordering and readiness waiting. Everything below is a delta on that.

**2. Stage NCCL on both nodes.** Both published recipes treat host-staged NCCL as required for
GB10 multi-node stability:

```bash
mkdir -p ~/nccl-2.30.7
cp /usr/lib/aarch64-linux-gnu/libnccl.so.2.30.7 ~/nccl-2.30.7/
ln -sf libnccl.so.2.30.7 ~/nccl-2.30.7/libnccl.so.2
```

**3. Apply the config delta** (table below) to the `.env`.

**4. Pin the control plane to the fabric** — the single most important line if you run a
default-deny firewall, and the one that cost us the longest debug:

```bash
-e SGLANG_HOST_IP=<this node's fabric IP>
```

**5. Evict page cache immediately before launch, and keep it bounded during the load:**

```bash
# before launch (no root needed — this is what scripts/cache-warden.py automates)
python3 -c "
import os,glob
for p in glob.glob(os.path.expanduser('~/.cache/huggingface/hub/**/*.safetensors'),recursive=True):
    rp=os.path.realpath(p)
    if os.path.exists(rp):
        fd=os.open(rp,os.O_RDONLY); os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)"

# during the load, on BOTH nodes
python3 scripts/cache-warden.py --model-dir ~/.cache/huggingface/hub \
    --interval 20 --stop-below-gb 25 --max-runtime 86400 --log ~/warden.jsonl
```

**6. Verify it took — at the point of effect, not in your config file:**

```bash
docker exec <container> env | grep -E 'SGLANG_HOST_IP|NCCL_IB_HCA|NCHANNELS'
curl -s localhost:8899/get_server_info | python3 -m json.tool | grep -E 'max_total|speculative|context'
```

A setting you did not confirm arrived is a setting you did not set. Ours silently disagreed with
the `.env` more than once.

---

## Stack (pin these when reproducing)

| | |
|---|---|
| Model | `RadixArk/Qwen3.8-Flash-Next-NVFP4` (206 shards, 135.2 GB) |
| Engine | SGLang, image `lmsysorg/sglang:qwen38flashnext` + MiaAI-Lab's SM121 QSA patch |
| Driver | NVIDIA 580.173.02 (open kernel module, aarch64) |
| NCCL | 2.30.7, host-staged and `LD_PRELOAD`ed (both recipes treat this as required for GB10 multi-node) |
| Hardware | 2× DGX Spark, GB10 / sm_121, 121 GB unified per node |

### Config delta vs. the MiaAI-Lab defaults

Everything else is hers. These are the only changes, and why:

| setting | value | why |
|---|---|---|
| `SGLANG_HOST_IP` | fabric IP, per node | the deadlock in §1 |
| `NCCL_MIN/MAX_NCHANNELS` | 4 | default negotiated **64 channels and hung** during init; tonyd2wild pins 4 |
| `CUDA_GRAPH_BS` | dense `1..8` | a ladder with **gaps** forces padding, and padded rows carry `decode_len=0`, which wedges the sparse indexer under spec decode. Dense inside range means nothing pads. tonyd2wild solves the same problem with `--disable-cuda-graph-padding`; we do both |
| `MAX_RUNNING_REQUESTS` | 8 | matches the graph ladder — 9–16 ran eager with transient workspace |
| `--max-total-tokens` | 600000 | pinned; unpinned "OOMs under sustained load" (tonyd2wild) |
| `MEM_FRACTION_STATIC` | 0.80 | leaves ~23 GB headroom |
| `NCCL_CROSS_NIC` | 0 | a multi-Spark report had `CROSS_NIC=1` **wedge within hours** under real traffic |

---

## The five things this repo adds

### 1. `SGLANG_HOST_IP` — multi-node SGLang deadlocks behind a default-deny firewall

**Symptom:** both ranks hang forever after `CustomAllreduce is disabled`. No error, no timeout, no
NCCL warning. NCCL itself completes fine — RoCE connects, channels build.

**Diagnosis** (`py-spy dump` on both ranks — this is what made it solvable):

```
rank 0 (head):    wait_until_ready (shm_broadcast.py:333)   <- writer awaiting subscriptions
rank 1 (worker):  wait_until_ready (shm_broadcast.py:344)   <- reader awaiting READY
```

A ZMQ PUB/SUB deadlock, not an NCCL problem. Root cause: SGLang's `get_local_ip_auto()` returns the
**default-route interface** — the management LAN address — and binds the cross-rank XPUB control
socket there. With `-P INPUT DROP` on that interface, the worker's subscription never arrives.

**Fix** — pin the control plane to the fabric, per node:

```bash
docker run ... -e SGLANG_HOST_IP=10.10.10.1   # head
docker run ... -e SGLANG_HOST_IP=10.10.10.2   # worker
```

The bug requires two conditions together: the fabric is not the default route, and the default-route
interface drops unsolicited inbound. Absent either, it never appears — which is presumably why the
published recipes don't mention it. vLLM's equivalent is `VLLM_HOST_IP`, which is what confirmed the fix was legitimate
rather than a workaround.

### 2. Page cache is half your memory budget — and it is the wedge

On GB10 the GPU and host share **one** physical pool. `safetensors` mmaps each shard, so file pages
and resident tensors compete for the same memory. Measured on an idle node:

```
reading 41.6 GB of shards  ->  MemFree 103.0 -> 64.1 GB    (~1:1)
```

A full load reads far more than that. When it runs out, **the NVIDIA driver fails before the kernel
reclaims** — from our own kernel log on a hung boot:

```
18:48:40  NVRM: Out of memory [NV_ERR_NO_MEMORY] .. _memdescAllocInternal @ mem_desc.c:1359
18:48:54  systemd-journald: Under memory pressure, flushing caches.
```

NVRM failed **14 seconds before** the kernel registered any pressure of its own, and that boot
contains no `page allocation failure`, no `order:` line, and no OOM-killer invocation. The cache was
resident; the driver could not have it. On unified memory this does not raise — **it wedges the
whole box**, and recovery is a physical power-cycle. Note: when it wedges this hard the **power
button is dead too** — no lights, no fans, no response to a long hold. Unplug and replug is the only
recovery we found.

**Consequences, all measured:**

- **Gate on `MemFree`, never `MemAvailable`.** MemAvailable counts reclaimable page cache the driver
  cannot use. Live example from a node in this state: `MemFree 1.9 GB` vs `MemAvailable 17.2 GB`.
- **`drop_caches` before launch is necessary but not sufficient** — the cache regrows during the
  135 GB read. See [`scripts/cache-warden.py`](scripts/cache-warden.py), which bounds it *during*
  and *after* load, needs **no root**, and requires no engine patch.
- **The same pressure silently costs throughput**, not just stability:

  | MemFree | median tok/s | CV | min |
  |---|---|---|---|
  | 1.7 GB | 58.77 | 9.7% | 44.96 |
  | 21 GB | 60.69 | **2.0%** | 58.74 |

  One root cause, two symptoms. Every benchmark in this repo evicts cache first.

  The warden's own A/B, identical 41.6 GB read:

  | | MemFree before → after |
  |---|---|
  | control (no warden) | 103.0 → 63.9 GB (**−39.1**) |
  | with warden | 102.8 → 102.6 GB (**−0.2**) |

  **Safe against a running engine.** `posix_fadvise(DONTNEED)` drops only *clean, unmapped* pages:
  pages the live engine has mmap'd are skipped by the kernel, dirty pages are never discarded, and
  the shards are read-only anyway. Worst case is a re-read from NVMe — latency, never corruption.
  It carries a hard self-limit, exits when its target process dies, and reports a loud `FATAL` on a
  bad directory and `UNVERIFIED` when it cannot evict. All three exit paths were proven by
  execution, not by reading the code.

- **`--max-total-tokens` is a ceiling, not a floor.** Requesting 600,000 with a warm cache silently
  yielded **557,120** — the engine under-fills the pool and does not warn.

### 3. Speculative decoding is at its architectural ceiling — and the drafter saturates it

`3/1/4` is not a conservative default. Raising it is **refused by the engine**:

```
NotImplementedError: Qwen QSA requires speculative_num_draft_tokens <= the QSA compress ratio (4):
the pending index-key ring holds one group; got 5
```

Qwen Sparse Attention's pending index-key ring holds exactly one group of 4. `num_draft_tokens`
can never exceed 4 **regardless of acceptance rate**.

**Per-run acceptance measurements — the first we're aware of for this model** — n=16, hard code prompt, per-run:

```
spec_accept_length   3.300 – 3.850   median 3.500     (HARD CEILING 4.0)
tok/s                52.55 – 62.56
correlation                          r = +0.786
```

Two findings:

**The throughput variance people see is acceptance, not noise.** It is not thermal, not clocks, not
page cache — all were flat/controlled. It is inherent and cannot be removed by environmental control.

**The drafter reaches 3.85 against a hard maximum of 4.0.** Net of the bonus token that is ~95% of
draft slots accepted on the best runs. The model would benefit from a larger draft budget and QSA
makes that impossible.

**External anchor.** The LMSYS day-0 writeup reports this model on **B200 TP4 NVFP4 at 540 tok/s
bs1 with MTP, accept length 3.3**. Our median is **3.500** on two GB10s — *higher* than the
reference implementation's published figure on far larger hardware. That's an independent
cross-check of both the number and the method, which matters because acceptance is easy to
measure wrongly (we did, three times, before getting it right).

LMSYS also names the mechanism behind the ceiling: **IndexShare MTP** reuses QSA selections across
draft steps, which is precisely why the pending index-key ring holds a single group.

⚠️ **Acceptance swings ~40 pp on prompt alone.** Measured on the same engine within one hour:
`3.5–3.7` on one code prompt, `2.475` on a chat+code mix. All arithmetically self-consistent — they
measure different prompt mixes. **Never quote an acceptance number without naming the prompt set.**

**Is there a way past the ceiling?** Not today. As of 2026-08-27 no DFlash / DSpark / EAGLE3
drafter exists for Flash-Next — z-lab's DFlash repo lists Muse-Glimmer-30B and Qwen3.8-**27B**
(a different model) and does not mention Flash-Next in supported models, roadmap or TODO. Two
things look like hits and are not: a HuggingFace repo named `…-MTP-Drafter-GGUF` is a repackaging
of the built-in MTP ("extracted … unmodified", 33 tensors), and SGLang's cookbook lists
`--speculative-algorithm DFLASH` because that's the engine-wide picker on every page — it needs a
`--speculative-draft-model-path` checkpoint that doesn't exist for this target. **The flag being
selectable is a label; the weights are the evidence.**

### 4. Vision works, and the self-review loop closes

Verified with a generated image of known content, not taken from the model card:

```
224×224 PNG, quadrants TL red / TR blue / BL green / BR yellow
answer: all four correct, image_tokens=64, 1.3 s
```

More useful — the full loop:

```
model writes HTML  →  headless Chrome renders at 1280px and 380px
                   →  model reads its own screenshots  →  critiques its own output
```

On a run truncated by too small a `max_tokens`, it reported *"the rendering is a complete failure…
just a dark background with a subtle grid pattern"* — describing what was on screen, not what it had
intended to write. On a complete run it found a font-size inconsistency and a checkmark-colour
mismatch that required zooming in to confirm. **It contradicts its own prior output**, which is the
property that makes self-review worth anything.

⚠️ **`max_tokens` ≥ 8000 for a full page.** At 2,600 the file truncated mid-CSS and produced a
*valid-looking* file that rendered blank. No error. Only the screenshot caught it.

### 5. Thinking mode: binary, helps reasoning, and fails catastrophically 30% of the time

There are **no effort levels**. The engine reports
`ReasoningToggleConfig(toggle_param='enable_thinking', default_enabled=True, effort_kwarg=None)`.

A/B on 8 reasoning problems with verifiable answers:

| | thinking OFF | thinking ON |
|---|---|---|
| score | 6/8 | **8/8** |
| time | 2.9 s | 14.8 s (5×) |
| tokens | 78 | 733, of which 651 thinking (9×) |

It fixes exactly the intuition traps: bat-and-ball `$0.10 → $0.05`, "Sally's sisters" `3 → 2`.

⚠️ **But do not default it on for code generation.** Same task, same config, temperature 0,
`max_tokens=14000`, n=10 each:

| | runaways (empty answer, budget exhausted) | completion tokens |
|---|---|---|
| thinking **ON** | **3 / 10** | 1,342 – 14,000 |
| thinking **OFF** | **0 / 10** | 222 – 287 |

**30% of thinking-on requests consumed the entire 14,000-token budget and returned zero characters
of content**, with everything in `reasoning_content` and `finish_reason: length`. Thinking off
solved the identical task in 222–287 tokens every single time — roughly **50× cheaper and
completely stable**.

Note the token range under thinking: 1,342 to 14,000, a **10× spread at temperature 0**. Greedy
decoding is not bit-reproducible on this stack (NVFP4 GEMM variance on sm_121 is the usual
explanation), and thinking amplifies that divergence into a coin-flip between "fine" and
"produces nothing at all".

**Practical guidance:**
- **Reasoning problems, short outputs** → thinking ON is a real win (6/8 → 8/8 on classic
  intuition traps: bat-and-ball `$0.10 → $0.05`, "Sally's sisters" `3 → 2`).
- **Code generation, long outputs** → thinking OFF. It is faster, ~50× cheaper in tokens, and does
  not silently return nothing.
- **If you must run thinking on unattended**, you need a guard: treat
  `finish_reason == "length"` or empty `content` as a retryable failure, not as a model answer.
  A harness without that guard will book 30% of its thinking-arm results as task failures and
  conclude "thinking hurts on code" — which is not what is happening.

⚠️ **Wherever thinking is on, `max_tokens` must be ≥ 2000** regardless. Thinking consumes the
*same* budget as the answer, so a small cap guarantees the empty-content outcome rather than
merely risking it.

Tool calling was **not** harmed by thinking in our testing (correct `tool_calls` at temp 0.0, 0.7 and
1.0). One recipe reports a token-0 `!!!!!` repetition loop for thinking+tools; we probed n=6 at
temp 1.0 and saw none, on the *riskier* configuration (flashinfer sampling, radix cache on).
**n=6 cannot prove absence of a rare probabilistic loop.** Keep it on the watch list.

---

## Dead ends — documented so you don't spend the time

**Clock headroom does not exist.** `clocks.max.sm` reports 3003 MHz; the GPU runs 2528 under load.
Locking `-lgc 2800,3003` yields **2528 MHz** — the floor does not take — and prefill changes by
**0.07%**:

| | clock | prefill |
|---|---|---|
| default | 2528 MHz | 3,055 tok/s |
| locked 2800–3003 | 2528 MHz | 3,053 tok/s |

GB10 is **memory-bandwidth bound**, not clock bound, for both prefill and decode. This also explains
the low power draw — the GPU is mostly waiting on memory.

**Raising the draft budget is impossible.** See §3.

**`--load-format dummy` should not be used on GB10** — the rule and the >150 GB transient figure are
Mia's; our contribution is only that it explains one of our own wedges. A "safe rehearsal" is more
dangerous than the real load.

---

## Thermals — much cooler than a comparable dense-ish MoE

Measured at concurrency 4, 1 Hz telemetry:

```
idle           42.0 °C · 10.4 W · 2398 MHz
peak (load)    52.0 °C · 35.5 W · 2522 MHz
```

For scale, a previous model on identical hardware peaked at **88 °C / 65 W** uncapped. Qwen runs
~36 °C cooler at ~45% the power — at *higher* clocks. Cause: ~6B active params/token (10 of 512
experts) and only 12 of 48 layers are full attention; the rest are linear-attention GDN.

**Practical effect:** thermal guard stages sized for the older model are effectively unreachable
(43 °C of margin), and a clock cap intended to control thermals has nothing left to control.

---

## Benchmark discipline

Two spectacular false results were produced and caught during this work. Both were **prefix-cache
artifacts**:

```
"72,000 tok/s prefill"   -> identical prompt repeated, radix cache hit
"46,388 tok/s prefill"   -> a shell function that never passed its seed argument
true prefill:  3,050 tok/s   (unique prompt per run, cached_tokens=0 asserted)
```

**Always assert `usage.prompt_tokens_details.cached_tokens == 0`** when measuring prefill. A 24×
speedup that appears without a config change is a cache hit, not a discovery.

Likewise, **`ignore_eos` benchmarks are a floor, not real-world throughput.** They force generation
past the natural stopping point into degenerate text:

| measurement | tok/s |
|---|---|
| real generation (10.7k tokens of HTML) | **62.9** |
| harness, `ignore_eos`, hard prompt, 800 tok | 47.6 |

Both correct; they measure different things. Name the prompt, token count and clock state on every
number, or it is not comparable to anything.

---

## Files

| | |
|---|---|
| [`scripts/cache-warden.py`](scripts/cache-warden.py) | bounds page cache during and after load; no root, no engine patch |
| [`bench/decode-bench.py`](bench/decode-bench.py) | decode benchmark that waits for idle, discards contended runs, reports medians, and names its conditions |

## Credit

This work stands on two recipes published first, and would not exist without them:

- **[MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks](https://github.com/MiaAI-Lab/Qwen3.8-Flash-Next-Dual-DGX-Sparks)** — the orchestration and the SM121 QSA Triton fallback kernel that makes this model run on sm_121 at all. Our deployment *is* this stack.
- **[tonyd2wild/qwen3.8-flash-next-nvfp4-dgx-spark](https://github.com/tonyd2wild/qwen3.8-flash-next-nvfp4-dgx-spark)** — `--disable-cuda-graph-padding`, NCCL channel pinning (which fixed a 64-channel init hang for us), KV pinning, and the rule that any fix making the model text-only is off the table.
- **[bird/GLM-spark](https://github.com/bird/GLM-spark)** — published the `posix_fadvise(DONTNEED)` page-cache mechanism first, as an in-loader vLLM patch. We arrived at it independently and measured it before finding theirs; `cache-warden.py` is an out-of-tree, no-root variant that also bounds cache *during* and *after* load.

## License

MIT.
