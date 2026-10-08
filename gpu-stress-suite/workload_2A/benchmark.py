"""
Workload 2A — Integrated diagnostic inference (the primary diagnostic phase).

LLaMA-2 13B at GPTQ-INT8 (produced by quantize.py from Meta's official
weights — same artifact pipeline as 1C, V100-validated end-to-end), batch
16, fixed input sequence length 2048, 256 generated tokens per prompt,
greedy decoding. KV cache is FP8 (`kv_cache_dtype="fp8_e5m2"`) so that the
spec batch and sequence length actually fit a 32 GB V100 — see README for
the memory math. vLLM 0.5.4 (last release with a working Volta sm_70 wheel).

METHOD DEVIATION: recipe specifies AWQ. autoawq versions split into two
incompatible groups vs the pinned stack: <=0.2.6 pins torch<2.4 (clashes
with vLLM 0.5.4's torch==2.4); >=0.2.9 pins transformers>=4.45 (clashes
with vLLM 0.5.4's era-pinned transformers 4.43.4). No version satisfies
both. Using AutoGPTQ-INT8 instead — same INT8 precision, same diagnostic
intent. See spec.method_deviation_note in metadata.

One or more independent vLLM instances per visible GPU (replicated, NOT
tensor-parallel) so each device is graded on its own. The recipe positions 2A as
the *primary diagnostic signal*: a genuinely larger model than 1C at higher
batch and longer sequences, with HBM ~85% full and the memory bus sustained
near saturation for 13 minutes.

Run is 15 minutes total. The first 120 s (thermal ramp) is written to the
CSV but excluded from the JSON summary statistics, which are scored on the
final 13 minutes.

If the spec batch does not fit at engine init (OOM), the per-GPU worker
halves the batch and re-attempts down to 1; the actual batch used per GPU
is recorded in metadata as a deviation.

Outputs (per run, in --output-dir):
  {run_id}_2A_metadata.json          run config, env, per-GPU + aggregate stats
  {run_id}_2A_throughput.csv         1 Hz tokens/sec per GPU (full run, unsliced)
  {run_id}_2A_per_prompt.csv         one row per completed prompt
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

WORKLOAD_NUM = 9  # 2A in the orchestrator's numbering (run_all.sh phase 6)





# --smoke: pre-delivery shakeout — tiny ungated model, no quantization, tiny
# sizes, no HF_TOKEN. Exercises the exact vLLM 0.5.4 runtime path (token-id
# prompt input, async per-token streaming, TTFT/ITL, stats, CSV/JSON
# writers). Not a benchmark.
SMOKE_SHAPE = {"batch": 2, "seq": 64}
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
    return {
        "p50": float(np.percentile(a, 50)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


def _stats(xs):
    """mean / variance / stddev / p50 / p95 / p99. No 'mode'."""
    if not xs:
        return {"count": 0, "mean": None, "variance": None, "stddev": None,
                "p50": None, "p95": None, "p99": None}
    import numpy as np
    a = np.asarray(xs, dtype="float64")
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "variance": float(a.var(ddof=0)),
        "stddev": float(a.std(ddof=0)),
        **_percentiles(xs),
    }


def _corpus_sha_and_texts(path):
    with open(path, "rb") as f:
        raw = f.read()
    return hashlib.sha256(raw).hexdigest(), json.loads(raw)["base_texts"]


def _fixed_length_ids(tokenizer, text, seq_len):
    """Tile + truncate token ids to EXACTLY seq_len. Same algorithm as 1C/3A."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    while len(ids) < seq_len:
        ids = ids + ids
    return ids[:seq_len]


def _flag(value):
    return str(value).lower() in ("true", "1", "yes", "on")


def _build_engine(model, batch, seq, opts, attempts):
    """Build an AsyncLLMEngine, halving batch on OOM-at-load until it fits.
    Halving is skipped when several instances share the GPU.
    Returns (engine, actual_batch); appends failed tries to attempts."""
    from vllm import AsyncLLMEngine, AsyncEngineArgs
    cur = batch
    while cur >= 1:
        try:
            args = dict(
                model=model,
                tokenizer=model,
                dtype="float16",
                seed=0,
                enforce_eager=opts.enforce_eager,
                gpu_memory_utilization=opts.gpu_mem_util / opts.instances_per_gpu,
                max_model_len=seq + opts.gen_tokens,
                max_num_seqs=cur,
                tensor_parallel_size=1,
                disable_log_stats=True,
                disable_log_requests=True,
            )
            if not opts.smoke:
                args["quantization"] = opts.quantization
                args["kv_cache_dtype"] = opts.kv_cache_dtype
            engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(**args))
            return engine, cur
        except (torch.cuda.OutOfMemoryError, RuntimeError, ValueError) as e:
            msg = str(e).lower()
            is_oom = ("out of memory" in msg or "no available memory" in msg
                      or isinstance(e, torch.cuda.OutOfMemoryError))
            attempts.append({"attempted_batch": cur,
                             "outcome": "OOM_AT_LOAD" if is_oom else "INIT_FAIL",
                             "error": repr(e)[:300]})
            if not is_oom or cur == 1 or opts.instances_per_gpu > 1:
                raise
            cur = cur // 2
    raise RuntimeError(f"engine init failed at every batch down to 1: {attempts}")


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

    dev_name = _torch.cuda.get_device_name(0)

    try:
        import vllm  # noqa: F401
        import vllm._C  # noqa: F401
    except Exception as e:
        print(f"  [GPU {gpu_id}] FATAL: vLLM C extension import failed ({e!r}).", flush=True)
        for _ in range(args.instances_per_gpu):
            _BARRIER.wait()
        return {"gpu_index": gpu_id, "instance_index": instance,
                "fatal": f"vllm import: {e!r}"}

    from vllm import SamplingParams
    from transformers import AutoTokenizer

    shape = _shape(args)
    batch, seq = shape["batch"], shape["seq"]
    model = SMOKE_MODEL if args.smoke else args.model_dir
    corpus_sha, texts = _corpus_sha_and_texts(args.corpus)
    tok = AutoTokenizer.from_pretrained(model, use_fast=True)
    prompt_ids = [_fixed_length_ids(tok, texts[i % len(texts)], seq)
                  for i in range(len(texts))]

    # Wait for lower-numbered instances to finish loading.
    for _ in range(instance):
        _BARRIER.wait()

    init_attempts = []
    try:
        engine, actual_batch = _build_engine(model, batch, seq, args, init_attempts)
    except Exception as e:
        print(f"  [GPU {gpu_id}.{instance}] FATAL engine init: {e!r}", flush=True)
        for _ in range(args.instances_per_gpu - instance):
            _BARRIER.wait()
        return {"gpu_index": gpu_id, "instance_index": instance,
                "fatal": f"engine init: {e!r}", "init_attempts": init_attempts}

    cache = engine.engine.cache_config
    kv_capacity = cache.num_gpu_blocks * cache.block_size // (seq + args.gen_tokens)
    if kv_capacity < actual_batch:
        print(f"  [GPU {gpu_id}.{instance}] WARNING KV cache holds {kv_capacity} "
              f"sequences, batch is {actual_batch}; requests will be preempted.", flush=True)

    if actual_batch != batch:
        print(f"  [GPU {gpu_id}] DEVIATION: spec batch={batch} did not fit, "
              f"using batch={actual_batch} (recorded in metadata).", flush=True)

    sampling = SamplingParams(temperature=0.0, max_tokens=args.gen_tokens,
                              ignore_eos=True, seed=0)

    async def drive():
        # Warm engine once (NOT measured); the *thermal* ramp is still
        # included in the CSV and discarded by the 120 s rule.
        async for _ in engine.generate({"prompt_token_ids": prompt_ids[0][:8]},
                                        SamplingParams(temperature=0.0, max_tokens=4),
                                        request_id=f"warm-{gpu_id}"):
            pass

        # Wait off the event loop until every instance in the group has loaded.
        loop = asyncio.get_running_loop()
        for _ in range(args.instances_per_gpu - instance):
            await loop.run_in_executor(None, _BARRIER.wait)

        t0 = time.perf_counter()
        t0_wall = time.time()
        deadline = t0 + args.duration
        next_idx = [0]
        rid = [0]
        per_prompt = []
        tput_events = []  # (elapsed_s, delta_tokens)

        async def worker():
            while True:
                now = time.perf_counter()
                if now >= deadline:
                    return
                ci = next_idx[0] % len(prompt_ids)
                next_idx[0] += 1
                my_rid = f"g{gpu_id}-{rid[0]}"
                rid[0] += 1
                ids = prompt_ids[ci]
                submit = time.perf_counter()
                last_t = submit
                last_n = 0
                ttft = None
                itls = []
                async for out in engine.generate({"prompt_token_ids": ids},
                                                 sampling, request_id=my_rid):
                    t = time.perf_counter()
                    n = len(out.outputs[0].token_ids)
                    if n <= last_n:
                        continue
                    if ttft is None:
                        ttft = t - submit
                    else:
                        itls.append((t - last_t) / (n - last_n) * 1000.0)
                    tput_events.append((t - t0, n - last_n))
                    last_t, last_n = t, n
                    if out.finished:
                        break
                done = time.perf_counter()
                if ttft is not None:
                    pj = _percentiles(itls)
                    per_prompt.append({
                        "gpu_index": gpu_id,
                        "corpus_index": ci,
                        "completion_elapsed_s": round(done - t0, 4),
                        "ttft_ms": round(ttft * 1000.0, 4),
                        "itl_p50_ms": pj["p50"], "itl_p95_ms": pj["p95"],
                        "itl_p99_ms": pj["p99"],
                        "output_tokens": last_n,
                        "instance_index": instance,
                    })

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(actual_batch)])
        return per_prompt, tput_events, t0_wall

    per_prompt, tput_events, t0_wall = asyncio.run(drive())

    max_s = int(args.duration)
    buckets = [0] * (max_s + 1)
    for el, d in tput_events:
        b = int(el)
        if 0 <= b <= max_s:
            buckets[b] += d
    per_second = [{"elapsed_s": s + 1, "tokens_per_sec": buckets[s]}
                  for s in range(max_s)]

    return {
        "gpu_index": gpu_id,
        "instance_index": instance,
        "gpu_name": dev_name,
        "spec_batch": batch,
        "actual_batch": actual_batch,
        "init_attempts": init_attempts,
        "seq_len": seq,
        "gen_tokens": args.gen_tokens,
        "corpus_sha256": corpus_sha,
        "kv_capacity_seqs": kv_capacity,
        "t0_wall": t0_wall,
        "per_prompt": per_prompt,
        "per_second": per_second,
    }


def _resolve_gpus(spec):
    n = torch.cuda.device_count()
    if spec == "all":
        return list(range(n))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


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
        for b, rb in zip(m["per_second"], r["per_second"]):
            b["tokens_per_sec"] += rb["tokens_per_sec"]
        m["kv_capacity_seqs"].append(r["kv_capacity_seqs"])
    return list(merged.values())


def _summarize(gpu_result, discard_s):
    pp = [r for r in gpu_result["per_prompt"]
          if r["completion_elapsed_s"] >= discard_s]
    ps = [b["tokens_per_sec"] for b in gpu_result["per_second"]
          if b["elapsed_s"] > discard_s]
    return {
        "gpu_index": gpu_result["gpu_index"],
        "gpu_name": gpu_result["gpu_name"],
        "spec_batch": gpu_result["spec_batch"],
        "actual_batch": gpu_result["actual_batch"],
        "instances": len(gpu_result["kv_capacity_seqs"]),
        "kv_capacity_seqs": gpu_result["kv_capacity_seqs"],
        "scored_prompts": len(pp),
        "discarded_ramp_s": discard_s,
        "tokens_per_sec": _stats(ps),
        "ttft_ms": _stats([r["ttft_ms"] for r in pp]),
        "itl_p99_ms": _stats([r["itl_p99_ms"] for r in pp if r["itl_p99_ms"] is not None]),
    }


def main():
    ap = argparse.ArgumentParser(description="Workload 2A — LLaMA-2 13B AWQ integrated inference")
    ap.add_argument("--model-dir", default="/workspace/model_int8")
    ap.add_argument("--corpus", default="/workspace/frozen_prompts.json")
    ap.add_argument("--max-vram-gb", type=float, default=32.0,
                    help="Per-GPU VRAM cap (passed by orchestrator). "
                         ">=24 -> 32gb config, >=12 -> 16gb config.")
    ap.add_argument("--batch", type=int, default=4,
                    help="prompts in flight per GPU")
    ap.add_argument("--seq", type=int, default=1024,
                    help="prompt sequence length")
    ap.add_argument("--gen-tokens", type=int, default=256)
    ap.add_argument("--duration", type=int, default=900, help="seconds (spec: 15 min)")
    ap.add_argument("--discard-ramp", type=int, default=120, help="seconds excluded from stats (spec: 2 min)")
    ap.add_argument("--gpus", default="all", help='"all" or e.g. "0,1,2"')
    ap.add_argument("--gpu-mem-util", type=float, default=0.95,
                    help="GPU memory fraction shared by all instances on a GPU")
    ap.add_argument("--instances-per-gpu", type=int, default=1,
                    help="independent vLLM engines per GPU")
    ap.add_argument("--quantization", default="gptq", help="vLLM quantization kernel")
    ap.add_argument("--kv-cache-dtype", default="fp8_e5m2", help="vLLM KV cache dtype")
    ap.add_argument("--enforce-eager", type=_flag, default=True,
                    help="true disables CUDA graphs")
    ap.add_argument("--mps", type=_flag, default=False,
                    help="run the CUDA MPS daemon for the duration of the run")
    ap.add_argument("--output-dir", default="/workspace/results")
    ap.add_argument("--smoke", action="store_true",
                    help="pre-delivery shakeout: tiny ungated model "
                         f"({SMOKE_MODEL}), no quantization, tiny sizes, no "
                         "HF_TOKEN. Validates the vLLM token-id streaming path "
                         "on commodity GPUs. NOT a benchmark.")
    args = ap.parse_args()

    if args.smoke:
        args.gen_tokens = min(args.gen_tokens, 16)
        args.duration = min(args.duration, 20)
        args.discard_ramp = 0
        args.gpu_mem_util = min(args.gpu_mem_util, 0.5)

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices. Run the container with --gpus all.")
    gpus = _resolve_gpus(args.gpus)
    shape = _shape(args)
    node_id = os.environ.get("NODE_ID", "")

    print(f"\n{'='*64}")
    if args.smoke:
        print("  Workload 2A — SMOKE (NOT a benchmark): vLLM path shakeout")
        print(f"  model      : {SMOKE_MODEL} (ungated, unquantized)")
    else:
        print("  Workload 2A — LLaMA-2 13B GPTQ-INT8 integrated diagnostic")
    print(f"  shape      : batch={shape['batch']} seq={shape['seq']}")
    print(f"  gen tokens : {args.gen_tokens}   duration: {args.duration}s "
          f"(discard first {args.discard_ramp}s)")
    print(f"  GPUs       : {gpus} ({args.instances_per_gpu} independent vLLM per GPU)")
    print(f"{'='*64}\n", flush=True)

    run_dt = datetime.now(timezone(timedelta(hours=TIMEZONE)))
    run_id = run_dt.strftime("%y-%m-%d_%H-%M")

    if args.mps:
        subprocess.run(["nvidia-cuda-mps-control", "-d"], check=True)
    try:
        results = _run_group([(g, i) for g in gpus
                              for i in range(args.instances_per_gpu)], args)
    finally:
        if args.mps:
            subprocess.run(["nvidia-cuda-mps-control"], input="quit\n", text=True)

    fatal = [r for r in results if r.get("fatal")]
    results = _merge_instances([r for r in results if not r.get("fatal")])
    if not results:
        print("  All GPU workers failed:", flush=True)
        for r in fatal:
            print(f"   GPU {r['gpu_index']}.{r['instance_index']}: {r['fatal']}", flush=True)
        sys.exit(1)

    per_gpu_stats = [_summarize(r, args.discard_ramp) for r in results]

    def _agg(field, sub):
        import numpy as np
        vals = [g[field][sub] for g in per_gpu_stats if g[field][sub] is not None]
        return float(np.mean(vals)) if vals else None

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    meta = {
        "node_id": node_id,
        "run_id": run_id,
        "workload_num": WORKLOAD_NUM,
        "workload": "2A_llama2_13b_gptq_int8_integrated_diagnostic",
        "smoke": args.smoke,
        "spec": {
            "model": (f"{SMOKE_MODEL} (SMOKE shakeout — ungated, unquantized)"
                      if args.smoke else
                      "meta-llama/Llama-2-13b-hf (GPTQ-INT8 in-container)"),
            "precision": "fp16-smoke" if args.smoke else "gptq-int8 + fp8_e5m2 kv",
            "decoding": "greedy (temperature=0, ignore_eos)",
            "batch": shape["batch"], "seq_len": shape["seq"],
            "gen_tokens": args.gen_tokens,
            "instances_per_gpu": args.instances_per_gpu,
            "quantization": None if args.smoke else args.quantization,
            "kv_cache_dtype": None if args.smoke else args.kv_cache_dtype,
            "duration_s": args.duration, "discarded_ramp_s": args.discard_ramp,
            "scored_window_s": args.duration - args.discard_ramp,
            "method_deviation_note": (
                "Recipe specifies AWQ. autoawq versions split into two "
                "incompatible groups vs the pinned stack: <=0.2.6 pin "
                "torch<2.4 (clashes with vLLM 0.5.4's torch==2.4.0); >=0.2.9 "
                "pin transformers>=4.45 (clashes with vLLM 0.5.4's era-pinned "
                "transformers 4.43.4). No autoawq version satisfies both "
                "constraints simultaneously. 2A uses AutoGPTQ-INT8 instead — "
                "same INT8 precision, same diagnostic intent (push HBM at "
                "batch 16 on a 13B model), same validated code path as 1C. FP8 KV cache preserves the recipe's "
                "'~12-14 GB KV at batch 16 seq 2304' memory profile."
            ),
        },
        "env": {
            "vllm": __import__("importlib.metadata", fromlist=["version"]).version("vllm"),
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "enforce_eager": args.enforce_eager, "mps": args.mps, "seed": 0,
        },
        "determinism_note": (
            "Greedy + fixed seed + enforce_eager + token-exact inputs. vLLM "
            "continuous batching can still reorder scheduling across runs; "
            "residual run-to-run variation under identical inputs is the "
            "hardware signal this suite is designed to surface."
        ),
        "corpus_sha256": results[0]["corpus_sha256"],
        "per_gpu_stats": per_gpu_stats,
        "per_gpu_init_attempts": [{"gpu_index": r["gpu_index"],
                                    "spec_batch": r["spec_batch"],
                                    "actual_batch": r["actual_batch"],
                                    "init_attempts": r["init_attempts"]}
                                   for r in results],
        "aggregate": {
            "gpu_count": len(results),
            "tokens_per_sec_mean": _agg("tokens_per_sec", "mean"),
            "ttft_ms_mean": _agg("ttft_ms", "mean"),
            "itl_p99_ms_mean": _agg("itl_p99_ms", "mean"),
        },
        "failed_gpus": fatal,
    }
    mpath = Path(args.output_dir) / f"{run_id}_workload{WORKLOAD_NUM}_metadata.json"
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)

    tpath = Path(args.output_dir) / f"{run_id}_workload{WORKLOAD_NUM}_throughput.csv"
    with open(tpath, "w") as f:
        f.write("timestamp,elapsed_s,gpu_index,tokens_per_sec\n")
        for r in results:
            for b in r["per_second"]:
                ts = datetime.fromtimestamp(r["t0_wall"] + b["elapsed_s"],
                                            run_dt.tzinfo).strftime("%Y-%m-%d %H:%M:%S")
                f.write(f"{ts},{b['elapsed_s']},{r['gpu_index']},{b['tokens_per_sec']}\n")

    ppath = Path(args.output_dir) / f"{run_id}_workload{WORKLOAD_NUM}_per_prompt.csv"
    with open(ppath, "w") as f:
        f.write("gpu_index,corpus_index,completion_elapsed_s,ttft_ms,"
                "itl_p50_ms,itl_p95_ms,itl_p99_ms,output_tokens,instance_index\n")
        for r in results:
            for p in r["per_prompt"]:
                f.write(f"{p['gpu_index']},{p['corpus_index']},{p['completion_elapsed_s']},"
                        f"{p['ttft_ms']},{p['itl_p50_ms']},{p['itl_p95_ms']},"
                        f"{p['itl_p99_ms']},{p['output_tokens']},{p['instance_index']}\n")

    print(f"\n{'='*64}")
    print("  2A SUMMARY (scored window only)")
    for g in per_gpu_stats:
        tp = g["tokens_per_sec"]; tt = g["ttft_ms"]; il = g["itl_p99_ms"]
        bdev = "" if g["actual_batch"] == g["spec_batch"] else f"  (spec batch={g['spec_batch']}, actual={g['actual_batch']})"
        print(f"  GPU {g['gpu_index']} ({g['gpu_name']}) prompts={g['scored_prompts']}{bdev}")
        if tp['mean'] is not None:
            print(f"    tokens/s  mean={tp['mean']:.1f} sd={tp['stddev']:.2f}")
        if tt['mean'] is not None:
            print(f"    TTFT ms   mean={tt['mean']:.1f} p99={tt['p99']:.1f}")
        if il['mean'] is not None:
            print(f"    ITL p99   mean={il['mean']:.2f}")
    print(f"{'='*64}")
    print(f"  metadata -> {mpath}")
    print(f"  throughput csv -> {tpath}")
    print(f"  per-prompt csv -> {ppath}\n", flush=True)


if __name__ == "__main__":
    main()
