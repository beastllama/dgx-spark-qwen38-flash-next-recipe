#!/usr/bin/env python3
"""Single-stream decode benchmark for an OpenAI-compatible endpoint (vLLM or SGLang).

Discipline, because casual decode benchmarks are usually wrong:
  * waits for a genuinely idle engine and DISCARDS any run foreign traffic touched mid-flight
    (a background keepalive will otherwise silently poison your numbers)
  * forces generation with `ignore_eos` so the run actually decodes instead of stopping early
  * reports the MEDIAN of N clean runs, never the best
  * names prompt, token count and clock state on every result

NOTE: `ignore_eos` makes this a FLOOR, not real-world throughput -- it pushes generation past the
natural stopping point into degenerate text. Real generation on the same stack measured ~33% higher.

Engine metric namespaces differ (`vllm:*` vs `sglang:*`) and the names are NOT the same; both are
parsed BY NAME. If neither matches, in-flight reads as UNVERIFIED -- never as "idle".

Env: BENCH_HOST BENCH_MODEL BENCH_NS BENCH_RUNS BENCH_MAX_TOK BENCH_LABEL BENCH_CLOCK_CMD
"""
import json, os, re, subprocess, sys, threading, time, urllib.request

# Engine-agnostic 2026-08-27. DEFAULTS ARE UNCHANGED so a DeepSeek run is byte-identical to the
# runs that produced the recorded baseline (chat 40.41 / code 65.71). Override via env to point at
# another stack -- SAME code, SAME prompts, SAME protocol is the whole point of the comparison.
HOST = os.environ.get("BENCH_HOST", "http://127.0.0.1:8899")
MODEL = os.environ.get("BENCH_MODEL", "qwen38-flash-next")
# vLLM exports `vllm:*`, SGLang exports `sglang:*` and the in-flight metric NAMES DIFFER.
# Parsed BY NAME (never positionally). If neither namespace matches, inflight() returns None and
# contended() reports UNVERIFIED -- it must never silently read as "idle".
NS = os.environ.get("BENCH_NS", "sglang")
_RUNNING = {"vllm": "vllm:num_requests_running", "sglang": "sglang:num_running_reqs"}[NS]
_WAITING = {"vllm": "vllm:num_requests_waiting", "sglang": "sglang:num_queue_reqs"}[NS]
RUNS = int(os.environ.get("BENCH_RUNS", "5"))
MAX_TOK = int(os.environ.get("BENCH_MAX_TOK", "800"))
LABEL = os.environ.get("BENCH_LABEL", "unlabelled")
PROMPTS = {
    "chat": "Explain how consensus works in a distributed database, with examples.",
    "code": "Write a Python LRU cache with TTL expiry, thread-safe, with tests. Explain the design.",
}

def inflight():
    try:
        b = urllib.request.urlopen(HOST + "/metrics", timeout=8).read().decode()
        r = re.search(r"^" + re.escape(_RUNNING) + r"\{?[^}\s]*\}?\s+([0-9.]+)", b, re.M)
        w = re.search(r"^" + re.escape(_WAITING) + r"\{?[^}\s]*\}?\s+([0-9.]+)", b, re.M)
        if r is None or w is None:
            return None
        return float(r.group(1)) + float(w.group(1))
    except Exception:
        return None

def wait_idle(timeout=1800):
    end = time.time() + timeout; streak = 0
    while time.time() < end:
        if inflight() == 0:
            streak += 1
            if streak >= 2: return True
        else: streak = 0
        time.sleep(15)
    return False

def sample(stop, out):
    while not stop.is_set():
        f = inflight(); out.append(f if f is not None else -1.0); time.sleep(1.5)

def contended(samples):
    if not samples: return "UNVERIFIED: no samples"
    foreign = blind = 0
    for f in samples:
        if f < 0:
            blind += 1; foreign = 0
            if blind >= 2: return "UNVERIFIED: /metrics unreadable mid-run"
        else:
            blind = 0; foreign = foreign + 1 if f >= 2.0 else 0
            if foreign >= 2: return "foreign traffic mid-run"
    return ""

def one(prompt):
    body = json.dumps({"model": MODEL, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": MAX_TOK, "min_tokens": MAX_TOK, "ignore_eos": True,
                       "temperature": 0, "stream": False}).encode()
    s = []; stop = threading.Event()
    th = threading.Thread(target=sample, args=(stop, s), daemon=True); th.start()
    t0 = time.time()
    try:
        d = json.load(urllib.request.urlopen(
            urllib.request.Request(HOST + "/v1/chat/completions", body,
                                   {"Content-Type": "application/json"}), timeout=600))
    finally:
        stop.set(); th.join(timeout=5)
    dt = time.time() - t0
    tok = d["usage"]["completion_tokens"]
    if tok < MAX_TOK * 0.9:
        return None, f"only {tok}/{MAX_TOK} tok generated"
    return (dt, tok), contended(s)

def main():
    # Clock state is REQUIRED context for a throughput number -- prompt, token count and clock
    # state each move the result by more than the effects people try to measure with them.
    # It is site-specific, so it comes from an env var. Unset -> print UNVERIFIED, never an
    # empty string: a check that cannot verify must say so rather than look fine.
    _cmd = os.environ.get("BENCH_CLOCK_CMD", "")
    if _cmd:
        _p = subprocess.run(_cmd, shell=True, capture_output=True, text=True)
        clk = (_p.stdout.strip() or "UNVERIFIED") if _p.returncode == 0 else "UNVERIFIED"
    else:
        clk = "UNVERIFIED (set BENCH_CLOCK_CMD)"
    print(f"=== decode bench :: {LABEL} ===")
    print(f"  prompts=chat,code  max_tokens={MAX_TOK} (ignore_eos)  runs={RUNS}/prompt  clock={clk}")
    if inflight() != 0:
        print("  engine busy — waiting for idle...", flush=True)
        if not wait_idle():
            print("  ABORT: never idle. UNVERIFIED."); return 2
    out = {}
    for name, p in PROMPTS.items():
        res = []; att = 0
        while len(res) < RUNS and att < RUNS * 4:
            att += 1
            if inflight() != 0:
                if not wait_idle(900): print("    gave up waiting"); break
            r, why = one(p)
            if why or r is None:
                print(f"    [{name}] attempt {att}: DISCARDED — {why}", flush=True); continue
            dt, tok = r; res.append(tok / dt)
            print(f"    [{name}] clean {len(res)}/{RUNS}: {tok/dt:6.2f} tok/s  ({dt:.2f}s)", flush=True)
        if res:
            res.sort(); out[name] = res[len(res)//2]
    print()
    for k, v in out.items(): print(f"  {k:5} median {v:.2f} tok/s   (n clean runs above)")
    if out: print(f"  MEAN of medians: {sum(out.values())/len(out):.2f} tok/s")
    print(json.dumps({"label": LABEL, "clock": clk, "max_tok": MAX_TOK, "medians": out}))
    return 0

sys.exit(main())
