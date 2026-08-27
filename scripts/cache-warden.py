#!/usr/bin/env python3
"""cache-warden — bound safetensors page cache DURING a model load, without root.

WHY THIS EXISTS
---------------
On GB10 the GPU and host share one physical pool. `safetensors` mmaps each shard, so the file
pages and the resident tensors compete for the SAME memory. Measured on an idle node:

    reading 41.6 GB of shards  ->  Cached +38.7 GB, MemFree 103.0 -> 64.1 GB   (~1:1)

A full load reads far more than that, and the NVIDIA driver cannot force the kernel to reclaim
those pages -- our hung-boot kernel log shows NVRM failing 14 s BEFORE the kernel registered any
memory pressure of its own. So the cache wins the race and the load dies mid-flight.

`drop_caches` before launch only starts you clean; the cache re-grows during the read. This evicts
the clean pages of already-read shards WHILE the load runs. `posix_fadvise(POSIX_FADV_DONTNEED)`
needs no privileges -- verified: 38.7 of 38.7 GB reclaimed, 100%, as an unprivileged user.

PRIOR ART -- WE WERE NOT FIRST. `bird/GLM-spark` (MIT) patches vLLM's `weight_utils.py` to call
`posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED)` on each shard close, with the same diagnosis
(~67 GB page cache + ~67 GB CUDA tensors vs ~119 GB available -> OOM at 66% of load; peak drops to
~72 GB after). Convergent discovery: we diagnosed it independently from our own kernel logs (NVRM failing
14 s before the kernel registered pressure), but THEY PUBLISHED THE MECHANISM FIRST. The published DGX Spark recipes (MiaAI-Lab, tonyd2wild) instead mandate
`sync; echo 3 > /proc/sys/vm/drop_caches` before every launch.
What is arguably still additive about THIS implementation, and the reason it exists:
  * out-of-tree -- no engine patch, so it works against vLLM and SGLang unchanged
  * needs NO ROOT (drop_caches does), which matters where there is no NOPASSWD grant
  * bounds cache DURING the load, not only before it; drop_caches alone lets it regrow mid-read
Credit belongs to bird/GLM-spark for publishing the mechanism first.

SAFETY: evicting a shard the loader later re-touches costs a re-read from NVMe, never corruption.

Self-limiting BY DESIGN. A loop that re-asserts state with no owner, no lease and no self-limit is
how a test guard once kept re-applying an admission gate on behalf of a unit that should have been
dead. This one carries a hard deadline, exits when its target process is gone, and cleans up on
SIGTERM/SIGINT.

USAGE
    cache-warden.py --model-dir DIR [--interval 5] [--max-runtime 1800]
                    [--pid PID | --container NAME] [--stop-below-gb 0]
"""
import argparse, glob, json, os, signal, subprocess, sys, time

STOP = False


def _sig(signum, _frame):
    global STOP
    STOP = True


def meminfo(key):
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith(key + ":"):
                return int(line.split()[1]) / 1048576.0
    return float("nan")


def shard_paths(model_dir):
    """Resolved, existing shard files. Skips dangling symlinks (an interrupted HF download)."""
    out = []
    # Accept a single model dir OR a whole HF hub root. The recursive pattern matters: pointed at
    # `~/.cache/huggingface/hub` the non-recursive globs match NOTHING (real layout is
    # hub/models--X/snapshots/<hash>/*.safetensors) and the tool exits FATAL. Covering the hub root
    # is the useful case at runtime -- a RETIRED model's shards are unreferenced and fully
    # evictable, while the live model's are mmap'd and correctly cannot be dropped.
    for pat in ("snapshots/*/*.safetensors", "*.safetensors", "**/*.safetensors"):
        for p in glob.glob(os.path.join(model_dir, pat), recursive=True):
            rp = os.path.realpath(p)
            if os.path.exists(rp):
                out.append(rp)
    return sorted(set(out))


def evict(paths):
    """Drop clean page-cache for each file. Returns (files_ok, files_failed)."""
    ok = failed = 0
    for rp in paths:
        try:
            fd = os.open(rp, os.O_RDONLY)
        except OSError:
            failed += 1
            continue
        try:
            os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
            ok += 1
        except OSError:
            failed += 1
        finally:
            os.close(fd)
    return ok, failed


def target_alive(pid, container):
    """Liveness of whatever we are warding for. None == no target given."""
    if pid:
        try:
            os.kill(pid, 0)
            return True
        except OSError:
            return False
    if container:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container],
            capture_output=True, text=True)
        return r.returncode == 0 and r.stdout.strip() == "true"
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--interval", type=float, default=5.0)
    ap.add_argument("--max-runtime", type=float, default=1800.0,
                    help="hard self-limit in seconds; ALWAYS exits by this deadline")
    ap.add_argument("--pid", type=int, default=0, help="exit when this pid is gone")
    ap.add_argument("--container", default="", help="exit when this container stops running")
    ap.add_argument("--stop-below-gb", type=float, default=0.0,
                    help="only evict when MemFree is below this (0 = always evict)")
    ap.add_argument("--log", default="", help="append JSONL telemetry here (fsync'd per line)")
    a = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    paths = shard_paths(a.model_dir)
    if not paths:
        print(f"cache-warden: FATAL no readable shards under {a.model_dir}", file=sys.stderr)
        return 2

    total = sum(os.path.getsize(p) for p in paths)
    print(f"cache-warden: warding {len(paths)} shards, {total/1e9:.1f} GB, "
          f"interval={a.interval}s deadline={a.max_runtime:.0f}s", flush=True)

    logfh = open(a.log, "a") if a.log else None
    start = time.time()
    ticks = 0
    reclaimed_total = 0.0
    try:
        while not STOP:
            elapsed = time.time() - start
            if elapsed >= a.max_runtime:
                print(f"cache-warden: self-limit {a.max_runtime:.0f}s reached, exiting", flush=True)
                break
            alive = target_alive(a.pid, a.container)
            if alive is False:
                print("cache-warden: target gone, exiting", flush=True)
                break

            free_before, cached_before = meminfo("MemFree"), meminfo("Cached")
            did = False
            if a.stop_below_gb <= 0 or free_before < a.stop_below_gb:
                ok, failed = evict(paths)
                did = True
                if failed and ok == 0:
                    # Never degrade to a silent no-op: if we cannot evict anything, say so loudly.
                    print(f"cache-warden: UNVERIFIED — every fadvise failed ({failed} files); "
                          f"cache is NOT being bounded", file=sys.stderr, flush=True)
            free_after, cached_after = meminfo("MemFree"), meminfo("Cached")
            ticks += 1
            reclaimed_total += max(0.0, cached_before - cached_after)

            rec = dict(t=round(elapsed, 1), evicted=did,
                       memfree_before=round(free_before, 1), memfree_after=round(free_after, 1),
                       cached_before=round(cached_before, 1), cached_after=round(cached_after, 1))
            if logfh:
                logfh.write(json.dumps(rec) + "\n")
                logfh.flush()
                os.fsync(logfh.fileno())   # survive a node hang; the trace is the whole point
            if did and cached_before - cached_after > 0.5:
                print(f"cache-warden: t={elapsed:6.0f}s  MemFree {free_before:6.1f} -> "
                      f"{free_after:6.1f} GB  (reclaimed {cached_before-cached_after:.1f})",
                      flush=True)
            time.sleep(a.interval)
    finally:
        if logfh:
            logfh.close()
    print(f"cache-warden: done after {time.time()-start:.0f}s, {ticks} ticks, "
          f"~{reclaimed_total:.0f} GB reclaimed cumulatively", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
