"""
Workload 3A — Failure characterization under memory pressure.

LLaMA-2 13B, full FP16 (no quantization), single GPU, sequential (one prompt
at a time by default), sequence length 4096, 256 generated tokens, greedy. vLLM 0.5.4.

This is NOT a throughput test. ~26 GB of FP16 weights plus the KV cache for a
4096+256-token context pushes a 32 GB V100 to its memory limit. The signal is
the *behavior under that pressure*: a healthy device evicts/recomputes KV
blocks, eats the latency penalty, and finishes; a marginal device shows
scheduling/queue storms, pathological ITL tails, or a hard OOM. The pattern
matters more than the numbers.

Spec note: the recipe's batch=4 OOMs at *load* time on a 32 GB GPU (26 GB
weights + ~13 GB KV + activations > 32 GB) so the pressure behavior is never
observed. Batch and sequence length come from the GPU profile.

Honesty note: vLLM 0.5.4 has no per-prompt
"allocation retry" API. The truthful equivalents it *does* expose are captured
instead — per-request queue time and scheduler time, and the terminal status
(OK / OOM). Engine-wide preemption/swap totals are read from the scheduler
when reachable and reported as such (not faked per-prompt).

GPUs are exercised one at a time by default (--gpu-mode sequential).

Outputs (per run, in --output-dir):
  {run_id}_3A_metadata.json     run config, env, per-GPU + aggregate stats
  {run_id}_3A_throughput.csv    1 Hz tokens/sec per GPU (full run)
  {run_id}_3A_per_prompt.csv    one row per prompt attempt
"""

import argparse
import asyncio
import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.multiprocessing as mp

TIMEZONE = 0  # UTC for standardization

WORKLOAD_NUM = 11  # 3A in the orchestrator's numbering (run_all.sh phase 12)





# --smoke: a pre-delivery shakeout, NOT the memory-pressure test. It runs the
# exact vLLM 0.5.4 path this file depends on — token-id prompt input, async
# streaming, TTFT/ITL, the OOM/queue/preemption capture, stats and CSV/JSON
# writers — with a tiny UNGATED model and tiny sizes so it fits any GPU and
# needs no HF_TOKEN. It deliberately does NOT push memory; it only proves the
# integration before the real 13B run on the V100.
SMOKE_SHAPE = {"batch": 1, "seq": 64}
SMOKE_MODEL = "facebook/opt-125m"


def _shape(args):
    if getattr(args, "smoke", False):
        return SMOKE_SHAPE
    return {"batch": args.batch, "seq": args.seq}


def _percentiles(xs):
    if not xs:
        return {"p50": None, "p95": None, "p99": None}
    import numpy as np
    a = np.asarray(xs, dtype="float64")
    return {"p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}


def _stats(xs):
    if not xs:
        return {"count": 0, "mean": None, "variance": None, "stddev": None,
                "p50": None, "p95": None, "p99": None}
    import numpy as np
    a = np.asarray(xs, dtype="float64")
    return {"count": int(a.size), "mean": float(a.mean()),
            "variance": float(a.var(ddof=0)), "stddev": float(a.std(ddof=0)),
            **_percentiles(xs)}


def _corpus(path):
    with open(path, "rb") as f:
        raw = f.read()
    return hashlib.sha256(raw).hexdigest(), json.loads(raw)["base_texts"]


def _fixed_len(tok, text, n):
    ids = tok(text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [tok.eos_token_id or 0]
    while len(ids) < n:
        ids = ids + ids
    return ids[:n]


def _flag(value):
    return str(value).lower() in ("true", "1", "yes", "on")


def _read_engine_preemptions(engine):
    """Best-effort, truthful: vLLM 0.5.4 keeps no public per-prompt preemption
    counter. Sum what the scheduler exposes if the internal layout is present;
    otherwise return None (reported as 'not exposed', never fabricated)."""
    try:
        total = 0
        scheds = engine.engine.scheduler
        scheds = scheds if isinstance(scheds, (list, tuple)) else [scheds]
        for s in scheds:
            total += int(getattr(s, "num_cumulative_preemption", 0))
        return total
    except Exception:
        return None


_BARRIER = None


def _init_worker(barrier):
    global _BARRIER
    _BARRIER = barrier


def _worker(gpu_id, instance, args):
    try:
        return _run_on_gpu(gpu_id, instance, args)
    except BaseException:
        _BARRIER.abort()
        raise


def _run_on_gpu(gpu_id, instance, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    import torch as _torch
    cap = _torch.cuda.get_device_capability(0)
    dev_name = _torch.cuda.get_device_name(0)
    if cap != (7, 0):
        print(f"  [GPU {gpu_id}] WARNING expected sm_70, got sm_{cap[0]}{cap[1]} "
              f"({dev_name}); off-spec for this suite.", flush=True)

    try:
        import vllm  # noqa: F401
        import vllm._C  # noqa: F401
    except Exception as e:
        for _ in range(args.instances_per_gpu):
            _BARRIER.wait()
        return {"gpu_index": gpu_id, "instance_index": instance,
                "fatal": f"vllm import: {e!r}"}

    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
    from transformers import AutoTokenizer

    shape = _shape(args)
    # The VRAM-table value is the TOTAL context budget (max_model_len), i.e.
    # the full KV-cache size that drives memory pressure. Input length is
    # whatever is left after reserving the generated tokens. Llama-2-13B's
    # hard context is exactly 4096, so 4096 input + 256 output is impossible;
    # total=4096 (input 3840 + gen 256) keeps a full 4096-token KV cache —
    # same pressure — while producing valid, in-distribution, reproducible
    # output. Spec deviation, parallel to the batch=4->1 fix.
    max_len = shape["seq"]
    input_len = max(8, max_len - args.gen_tokens)
    model = SMOKE_MODEL if args.smoke else args.model
    revision = None if args.smoke else (args.revision or None)
    sha, texts = _corpus(args.corpus)
    tok = AutoTokenizer.from_pretrained(model, revision=revision, use_fast=True)
    prompts = [_fixed_len(tok, texts[i % len(texts)], input_len)
               for i in range(args.num_prompts)]

    # Wait for lower-numbered instances to finish loading.
    for _ in range(instance):
        _BARRIER.wait()

    # Engine init itself can OOM if weights + minimum KV don't fit — that is a
    # legitimate (hard-failure) characterization outcome, recorded as such.
    try:
        engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
            model=model,
            revision=revision,
            dtype="float16",
            quantization=None if args.smoke else args.quantization,
            kv_cache_dtype="auto" if args.smoke else args.kv_cache_dtype,
            seed=0,
            enforce_eager=args.enforce_eager,
            gpu_memory_utilization=args.gpu_mem_util / args.instances_per_gpu,
            max_model_len=max_len,
            max_num_seqs=shape["batch"],
            tensor_parallel_size=1,
            disable_log_stats=True,
            disable_log_requests=True,
            swap_space=args.swap_space_gb,
        ))
    except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
        msg = str(e).lower()
        kind = "OOM_AT_LOAD" if ("out of memory" in msg or
                                 isinstance(e, torch.cuda.OutOfMemoryError)) else "INIT_FAIL"
        print(f"  [GPU {gpu_id}.{instance}] {kind}: {e!r}", flush=True)
        for _ in range(args.instances_per_gpu - instance):
            _BARRIER.wait()
        return {"gpu_index": gpu_id, "instance_index": instance, "gpu_name": dev_name,
                "seq_len": input_len, "max_model_len": max_len,
                "corpus_sha256": sha, "load_status": kind,
                "load_error": repr(e)[:500], "kv_capacity_seqs": None,
                "t0_wall": None, "per_prompt": [], "per_second": []}

    cache = engine.engine.cache_config
    kv_capacity = cache.num_gpu_blocks * cache.block_size // max_len

    sampling = SamplingParams(temperature=0.0, max_tokens=args.gen_tokens,
                              ignore_eos=True, seed=0)

    async def drive():
        async for _ in engine.generate({"prompt_token_ids": prompts[0][:8]},
                                        SamplingParams(temperature=0.0, max_tokens=4),
                                        request_id=f"warm-{gpu_id}"):
            pass

        # Wait off the event loop until every instance in the group has loaded.
        loop = asyncio.get_running_loop()
        for _ in range(args.instances_per_gpu - instance):
            await loop.run_in_executor(None, _BARRIER.wait)

        t0 = time.perf_counter()
        t0_wall = time.time()
        rows = []
        tput_events = []
        next_idx = [0]

        async def run_prompt(i, ids):
            submit = time.perf_counter()
            last_t, last_n, ttft, itls = submit, 0, None, []
            status, err = "OK", ""
            qtime = stime = None
            try:
                async for out in engine.generate({"prompt_token_ids": ids},
                                                 sampling, request_id=f"g{gpu_id}-{i}"):
                    t = time.perf_counter()
                    n = len(out.outputs[0].token_ids)
                    if n > last_n:
                        if ttft is None:
                            ttft = t - submit
                        else:
                            itls.append((t - last_t) / (n - last_n) * 1000.0)
                        tput_events.append((t - t0, n - last_n))
                        last_t, last_n = t, n
                    if out.finished:
                        m = out.metrics
                        if m is not None:
                            qtime = getattr(m, "time_in_queue", None)
                            if m.first_scheduled_time and m.arrival_time:
                                stime = m.first_scheduled_time - m.arrival_time
                        break
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                em = str(e).lower()
                status = "OOM" if ("out of memory" in em or
                                   isinstance(e, torch.cuda.OutOfMemoryError)) else "ERROR"
                err = repr(e)[:300]
                print(f"  [GPU {gpu_id}] prompt {i}: {status}", flush=True)

            pj = _percentiles(itls)
            rows.append({
                "gpu_index": gpu_id,
                "corpus_index": i % len(texts),
                "status": status,
                "error": err,
                "ttft_ms": round(ttft * 1000.0, 4) if ttft is not None else None,
                "itl_p50_ms": pj["p50"], "itl_p95_ms": pj["p95"], "itl_p99_ms": pj["p99"],
                "output_tokens": last_n,
                "queue_time_s": round(qtime, 6) if qtime is not None else None,
                "sched_delay_s": round(stime, 6) if stime is not None else None,
                "engine_cumulative_preemptions": _read_engine_preemptions(engine),
                "instance_index": instance,
            })

        async def worker():
            while next_idx[0] < len(prompts):
                i = next_idx[0]
                next_idx[0] += 1
                await run_prompt(i, prompts[i])

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(shape["batch"])])
        return rows, tput_events, t0_wall

    rows, tput_events, t0_wall = asyncio.run(drive())

    max_s = int(tput_events[-1][0]) + 1 if tput_events else 0
    buckets = [0] * (max_s + 1)
    for el, d in tput_events:
        b = int(el)
        if 0 <= b <= max_s:
            buckets[b] += d
    per_second = [{"elapsed_s": s + 1, "tokens_per_sec": buckets[s]}
                  for s in range(max_s)]

    return {"gpu_index": gpu_id, "instance_index": instance, "gpu_name": dev_name,
            "seq_len": input_len, "max_model_len": max_len,
            "corpus_sha256": sha, "load_status": "OK",
            "kv_capacity_seqs": kv_capacity, "t0_wall": t0_wall,
            "per_prompt": rows, "per_second": per_second}


def _resolve_gpus(spec):
    n = torch.cuda.device_count()
    return list(range(n)) if spec == "all" else [int(x) for x in spec.split(",") if x.strip()]


def _run_group(tasks, args):
    """Run one process per (gpu, instance) task; all start timing together."""
    ctx = mp.get_context("spawn")
    barrier = ctx.Barrier(len(tasks))
    with ctx.Pool(processes=len(tasks), initializer=_init_worker,
                  initargs=(barrier,), maxtasksperchild=1) as pool:
        return pool.starmap(_worker, [(g, i, args) for g, i in tasks])


def _merge_instances(results):
    """Combine per-instance results into one result per GPU."""
    merged = {}
    for r in sorted(results, key=lambda r: (r["gpu_index"], r["instance_index"])):
        m = merged.get(r["gpu_index"])
        if m is None:
            merged[r["gpu_index"]] = {**r, "per_prompt": list(r["per_prompt"]),
                                      "per_second": [dict(b) for b in r["per_second"]],
                                      "kv_capacity_seqs": [r["kv_capacity_seqs"]]}
            continue
        m["per_prompt"] += r["per_prompt"]
        for b in r["per_second"]:
            if b["elapsed_s"] <= len(m["per_second"]):
                m["per_second"][b["elapsed_s"] - 1]["tokens_per_sec"] += b["tokens_per_sec"]
            else:
                m["per_second"].append(dict(b))
        m["kv_capacity_seqs"].append(r["kv_capacity_seqs"])
        m["t0_wall"] = m["t0_wall"] or r["t0_wall"]
        if r["load_status"] != "OK":
            m["load_status"], m["load_error"] = r["load_status"], r["load_error"]
    return list(merged.values())


def _summary(res):
    pp = res.get("per_prompt", [])
    ok = [r for r in pp if r["status"] == "OK"]
    ps = [b["tokens_per_sec"] for b in res.get("per_second", [])]
    return {
        "gpu_index": res["gpu_index"],
        "gpu_name": res.get("gpu_name"),
        "load_status": res.get("load_status"),
        "instances": len(res["kv_capacity_seqs"]),
        "kv_capacity_seqs": res["kv_capacity_seqs"],
        "attempts": len(pp),
        "completed": len(ok),
        "oom_events": sum(1 for r in pp if r["status"] == "OOM"),
        "error_events": sum(1 for r in pp if r["status"] == "ERROR"),
        "tokens_per_sec": _stats(ps),
        "ttft_ms": _stats([r["ttft_ms"] for r in ok if r["ttft_ms"] is not None]),
        "itl_p99_ms": _stats([r["itl_p99_ms"] for r in ok if r["itl_p99_ms"] is not None]),
        "queue_time_s": _stats([r["queue_time_s"] for r in ok if r["queue_time_s"] is not None]),
        "degradation": (
            "hard_oom" if res.get("load_status") != "OK" or
            any(r["status"] == "OOM" for r in pp)
            else "graceful"
        ),
    }


def main():
    ap = argparse.ArgumentParser(description="Workload 3A — LLaMA-2 13B FP16 memory-pressure")
    ap.add_argument("--model", default="meta-llama/Llama-2-13b-hf")
    ap.add_argument("--revision", default="", help="commit SHA to pin the repo")
    ap.add_argument("--corpus", default="/workspace/frozen_prompts.json")
    ap.add_argument("--max-vram-gb", type=float, default=32.0,
                    help="Per-GPU VRAM cap (passed by orchestrator). "
                         ">=24 -> 32gb config, >=12 -> 16gb config.")
    ap.add_argument("--duration", type=float, default=None,
                    help="Accepted from the orchestrator for uniformity; "
                         "3A is prompt-bounded (--num-prompts), so this is "
                         "ignored.")
    ap.add_argument("--batch", type=int, default=1,
                    help="prompts in flight per engine")
    ap.add_argument("--seq", type=int, default=2048,
                    help="prompt sequence length")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--num-prompts", type=int, default=16)
    ap.add_argument("--gpus", default="all", help='"all" or e.g. "0,3"')
    ap.add_argument("--gpu-mem-util", type=float, default=0.97,
                    help="deliberately high — this test is about the limit. "
                         "0.97 pushes measured memory utilization toward the "
                         "spec's 95%+; if this OOMs at load it is a recorded "
                         "characterization outcome, not a crash.")
    ap.add_argument("--swap-space-gb", type=int, default=1,
                    help="CPU swap space for KV eviction; small reserve so "
                         "graceful degradation can still happen but more of "
                         "the working set stays resident on the GPU.")
    ap.add_argument("--instances-per-gpu", type=int, default=1,
                    help="independent vLLM engines per GPU")
    ap.add_argument("--gpu-mode", choices=["parallel", "sequential"], default="sequential",
                    help="run all GPUs at once, or one GPU at a time")
    ap.add_argument("--quantization", default=None, help="vLLM quantization kernel")
    ap.add_argument("--kv-cache-dtype", default="auto", help="vLLM KV cache dtype")
    ap.add_argument("--enforce-eager", type=_flag, default=True,
                    help="true disables CUDA graphs")
    ap.add_argument("--mps", type=_flag, default=False,
                    help="run the CUDA MPS daemon for the duration of the run")
    ap.add_argument("--output-dir", default="/workspace/results")
    ap.add_argument("--smoke", action="store_true",
                    help="pre-delivery shakeout: tiny ungated model "
                         f"({SMOKE_MODEL}), tiny sizes, no HF_TOKEN, no memory "
                         "pressure. Validates the vLLM token-id streaming path "
                         "on commodity GPUs. NOT the 13B memory-pressure test.")
    args = ap.parse_args()

    if args.smoke:
        args.gen_tokens = min(args.gen_tokens, 16)
        args.num_prompts = min(args.num_prompts, 2)
        args.gpu_mem_util = min(args.gpu_mem_util, 0.5)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices. Run with --gpus all.")
    gpus = _resolve_gpus(args.gpus)
    shape = _shape(args)
    node_id = os.environ.get("NODE_ID", "")
    # seq = TOTAL context budget (max_model_len); input = total - gen_tokens.
    # Must mirror _run_on_gpu exactly.
    max_len = shape["seq"]
    input_len = max(8, max_len - args.gen_tokens)

    print(f"\n{'='*64}")
    if args.smoke:
        print("  Workload 3A — SMOKE (NOT the memory test): vLLM path shakeout")
        print(f"  model      : {SMOKE_MODEL} (ungated)")
    else:
        print("  Workload 3A — LLaMA-2 13B FP16 memory-pressure characterization")
    print(f"  shape      : batch={shape['batch']} seq={shape['seq']} "
          f"+ gen={args.gen_tokens} = max_model_len={max_len}")
    print(f"  GPUs       : {gpus} ({args.gpu_mode}, {args.instances_per_gpu} "
          f"independent vLLM per GPU)")
    print(f"  gpu_mem_util={args.gpu_mem_util}  swap_space={args.swap_space_gb}GB")
    print(f"{'='*64}\n", flush=True)

    run_dt = datetime.now(timezone(timedelta(hours=TIMEZONE)))
    run_id = run_dt.strftime("%y-%m-%d_%H-%M")

    instances = range(args.instances_per_gpu)
    if args.gpu_mode == "sequential":
        groups = [[(g, i) for i in instances] for g in gpus]
    else:
        groups = [[(g, i) for g in gpus for i in instances]]

    if args.mps:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
    try:
        results = [r for group in groups for r in _run_group(group, args)]
    finally:
        if args.mps:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True)

    fatal = [r for r in results if r.get("fatal")]
    results = _merge_instances([r for r in results if not r.get("fatal")])
    if not results:
        for r in fatal:
            print(f"  GPU {r['gpu_index']}.{r['instance_index']} fatal: {r['fatal']}", flush=True)
        sys.exit(1)

    per_gpu = [_summary(r) for r in results]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    meta = {
        "node_id": node_id,
        "run_id": run_id,
        "workload_num": WORKLOAD_NUM,
        "workload": "3A_llama2_13b_fp16_memory_pressure",
        "smoke": args.smoke,
        "spec": {
            "model": f"{SMOKE_MODEL} (SMOKE shakeout — ungated)" if args.smoke else args.model,
            "precision": "fp16-smoke" if args.smoke else "fp16 (no quantization)",
            "decoding": "greedy (temperature=0, ignore_eos)",
            "batch": shape["batch"],
            "instances_per_gpu": args.instances_per_gpu,
            "gpu_mode": args.gpu_mode,
            "quantization": None if args.smoke else args.quantization,
            "kv_cache_dtype": "auto" if args.smoke else args.kv_cache_dtype,
            "input_seq_len": input_len,
            "gen_tokens": args.gen_tokens,
            "max_model_len": max_len,
            "context_note": (
                "seq is the TOTAL context budget (max_model_len); input = "
                "total - gen_tokens. Llama-2-13B's hard context is 4096, so "
                "the recipe's 4096 input + 256 output is physically impossible "
                "in one window. total=4096 keeps a full 4096-token KV cache "
                "(identical memory pressure) with valid, reproducible output. "
                "Spec deviation, parallel to the batch=4->1 fix."
            ),
            "num_prompts": args.num_prompts,
            "gpu_memory_utilization": args.gpu_mem_util,
            "swap_space_gb": args.swap_space_gb,
            "intent": "failure characterization under memory pressure, "
                      "graceful-degradation vs hard-OOM",
        },
        "env": {
            "vllm": __import__("importlib.metadata", fromlist=["version"]).version("vllm"),
            "torch": torch.__version__, "cuda": torch.version.cuda,
            "enforce_eager": args.enforce_eager, "mps": args.mps, "seed": 0,
        },
        "counter_honesty_note": (
            "vLLM 0.5.4 exposes no per-prompt allocation-retry counter. "
            "Recorded instead: per-prompt queue_time_s, sched_delay_s, terminal "
            "status (OK/OOM/ERROR), and engine_cumulative_preemptions when the "
            "scheduler exposes it (else null — never fabricated)."
        ),
        "corpus_sha256": results[0]["corpus_sha256"],
        "per_gpu_summary": per_gpu,
        "failed_gpus": fatal,
    }
    mpath = Path(args.output_dir) / f"{run_id}_workload{WORKLOAD_NUM}_metadata.json"
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)

    tpath = Path(args.output_dir) / f"{run_id}_workload{WORKLOAD_NUM}_throughput.csv"
    with open(tpath, "w") as f:
        f.write("timestamp,elapsed_s,gpu_index,tokens_per_sec\n")
        for r in results:
            for b in r.get("per_second", []):
                ts = datetime.fromtimestamp(r["t0_wall"] + b["elapsed_s"],
                                            run_dt.tzinfo).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts},{b['elapsed_s']},{r['gpu_index']},{b['tokens_per_sec']}\n")

    ppath = Path(args.output_dir) / f"{run_id}_workload{WORKLOAD_NUM}_per_prompt.csv"
    with open(ppath, "w") as f:
        f.write("gpu_index,corpus_index,status,ttft_ms,itl_p50_ms,itl_p95_ms,"
                "itl_p99_ms,output_tokens,queue_time_s,sched_delay_s,"
                "engine_cumulative_preemptions,error,instance_index\n")
        for r in results:
            for p in r.get("per_prompt", []):
                err = (p["error"] or "").replace(",", ";").replace("\n", " ")
                f.write(f"{p['gpu_index']},{p['corpus_index']},{p['status']},"
                        f"{p['ttft_ms']},{p['itl_p50_ms']},{p['itl_p95_ms']},"
                        f"{p['itl_p99_ms']},{p['output_tokens']},{p['queue_time_s']},"
                        f"{p['sched_delay_s']},{p['engine_cumulative_preemptions']},{err},"
                        f"{p['instance_index']}\n")
            if r.get("load_status") not in (None, "OK") and not r.get("per_prompt"):
                f.write(f"{r['gpu_index']},,{r['load_status']},,,,,,,,,"
                        f"{repr(r.get('load_error',''))[:200].replace(',',';')},"
                        f"{r['instance_index']}\n")

    print(f"\n{'='*64}")
    print("  3A SUMMARY")
    for g in per_gpu:
        print(f"  GPU {g['gpu_index']} ({g['gpu_name']})  load={g['load_status']}  "
              f"completed={g['completed']}/{g['attempts']}  oom={g['oom_events']}  "
              f"-> {g['degradation']}")
        tp = g["tokens_per_sec"]; il = g["itl_p99_ms"]
        if tp["mean"] is not None:
            print(f"    tok/s: mean={tp['mean']:.1f}  p50={tp['p50']:.1f}")
        if il["mean"] is not None:
            print(f"    ITL p99 ms: mean={il['mean']:.2f} max-tail p99={il['p99']:.2f}")
    print(f"{'='*64}")
    print(f"  metadata -> {mpath}")
    print(f"  throughput csv -> {tpath}")
    print(f"  per-prompt csv -> {ppath}\n", flush=True)


if __name__ == "__main__":
    main()
