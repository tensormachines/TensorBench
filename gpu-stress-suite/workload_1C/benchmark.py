"""
Workload 1C — Integrated inference baseline.

LLaMA-3 8B, INT8 (GPTQ, produced by quantize.py from Meta's official weights),
batch 16, fixed input sequence length 1024, 128 generated tokens per prompt,
greedy decoding. vLLM 0.5.4 (last release with a working Volta sm_70 wheel).

One or more independent vLLM instances per visible GPU (replicated, NOT tensor-parallel)
so each device is graded on its own — a coupled tensor-parallel run would mask
per-GPU variance, which is exactly the signal the health classifier needs.

Run is 7 minutes. The first 60 s (thermal ramp) is written to the CSV but
excluded from the JSON summary statistics, which are scored on the final
6 minutes.

Outputs (per run, in --output-dir):
  {run_id}_1C_metadata.json          run config, env, per-GPU + aggregate stats
  {run_id}_1C_throughput.csv         1 Hz tokens/sec per GPU (full run, unsliced)
  {run_id}_1C_per_prompt.csv         one row per completed prompt
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

WORKLOAD_NUM = 8  # 1C in the orchestrator's numbering (run_all.sh phase 4)


# Silent compute error (SDC) probe: each GPU runs the same fixed-token-id
# prompt SDC_PROBE_REPS times with greedy decoding + fixed seed, generating
# SDC_PROBE_TOKENS tokens per rep. main() then cross-checks token-by-token:
# (a) within-GPU self-consistency, and (b) cross-GPU majority-vote consensus.
# Approach mirrors Meta's "Silent Data Corruptions at Scale" cross-replica
# voting; takes ~3-5 s per GPU before the main timed window.
SDC_PROBE_REPS = 3
SDC_PROBE_TOKENS = 64




# --smoke: a pre-delivery shakeout, NOT a benchmark. It exercises the exact
# vLLM 0.5.4 runtime path this file depends on — token-id prompt input
# ({"prompt_token_ids": ...}), AsyncLLMEngine per-token streaming, TTFT/ITL
# capture, stats and CSV/JSON writers — using a tiny UNGATED model, no
# quantization, and tiny sizes. Fits any GPU, needs no HF_TOKEN. The whole
# point is to flush integration breakage on commodity hardware before the
# real run on the V100.
SMOKE_SHAPE = {"batch": 2, "seq": 64}
SMOKE_MODEL = "facebook/opt-125m"


def _shape(args):
    if getattr(args, "smoke", False):
        return SMOKE_SHAPE
    return {"batch": args.batch, "seq": args.seq}


# --------------------------------------------------------------------------- #
# helpers (pure, no CUDA)
# --------------------------------------------------------------------------- #
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
    p = _percentiles(xs)
    return {
        "count": int(a.size),
        "mean": float(a.mean()),
        "variance": float(a.var(ddof=0)),
        "stddev": float(a.std(ddof=0)),
        **p,
    }


def _corpus_sha_and_texts(path):
    with open(path, "rb") as f:
        raw = f.read()
    return hashlib.sha256(raw).hexdigest(), json.loads(raw)["base_texts"]


def _fixed_length_ids(tokenizer, text, seq_len):
    """Tile + truncate token ids to EXACTLY seq_len. Deterministic; this is
    what makes 'fixed input sequence length' a hard guarantee rather than an
    average."""
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if not ids:
        ids = [tokenizer.eos_token_id or 0]
    while len(ids) < seq_len:
        ids = ids + ids
    return ids[:seq_len]


def _flag(value):
    return str(value).lower() in ("true", "1", "yes", "on")


# --------------------------------------------------------------------------- #
# per-instance worker (runs in its own spawned process with one visible device)
# --------------------------------------------------------------------------- #
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
    # Pin the device BEFORE importing vllm/torch CUDA so the child sees exactly
    # one GPU as cuda:0.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")

    import torch as _torch

    dev_name = _torch.cuda.get_device_name(0)

    # Stops early when the vLLM wheel cannot load on this GPU.
    try:
        import vllm  # noqa: F401
        import vllm._C  # noqa: F401
    except Exception as e:  # pragma: no cover
        print(f"  [GPU {gpu_id}] FATAL: vLLM C extension import failed "
              f"({e!r}). The wheel is not usable on this GPU.", flush=True)
        for _ in range(args.instances_per_gpu):
            _BARRIER.wait()
        return {"gpu_index": gpu_id, "instance_index": instance,
                "fatal": f"vllm import: {e!r}"}

    from vllm import AsyncLLMEngine, AsyncEngineArgs, SamplingParams
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

    engine = AsyncLLMEngine.from_engine_args(AsyncEngineArgs(
        model=model,
        tokenizer=model,
        quantization=(None if args.smoke else args.quantization),
        dtype="float16",
        seed=0,
        enforce_eager=args.enforce_eager,
        gpu_memory_utilization=args.gpu_mem_util / args.instances_per_gpu,
        max_model_len=seq + args.gen_tokens,
        max_num_seqs=batch,
        tensor_parallel_size=1,
        disable_log_stats=True,
        disable_log_requests=True,
    ))
    cache = engine.engine.cache_config
    kv_capacity = cache.num_gpu_blocks * cache.block_size // (seq + args.gen_tokens)
    if kv_capacity < batch:
        print(f"  [GPU {gpu_id}.{instance}] WARNING KV cache holds {kv_capacity} "
              f"sequences, batch is {batch}; requests will be preempted.", flush=True)
    sampling = SamplingParams(temperature=0.0, max_tokens=args.gen_tokens,
                              ignore_eos=True, seed=0)

    async def _sdc_probe_reps():
        """Run the same fixed-token-id prompt 3 times with greedy decoding +
        fixed seed. Records the output token sequence per rep. Healthy GPUs
        should produce **bit-identical** outputs across reps (within-GPU
        self-consistency) and **identical token-by-token** sequences to the
        other GPUs (cross-GPU consensus, computed later in main()).
        Single-sequence at a time so vLLM continuous-batching can't reorder."""
        fixed_prompt = prompt_ids[0][:128]
        reps = []
        for rep in range(SDC_PROBE_REPS):
            sp = SamplingParams(temperature=0.0, max_tokens=SDC_PROBE_TOKENS,
                                ignore_eos=True, seed=0)
            final_token_ids = []
            async for out in engine.generate(
                {"prompt_token_ids": fixed_prompt}, sp,
                request_id=f"sdc-g{gpu_id}-r{rep}",
            ):
                if out.finished:
                    final_token_ids = list(out.outputs[0].token_ids)
                    break
            reps.append(final_token_ids)
        return reps

    async def drive():
        # SDC probe (cross-GPU consensus + within-GPU self-consistency).
        # Default-on; --no-sdc-probe + --smoke skip it. Runs BEFORE the
        # warm-up so the timed window isn't affected.
        sdc_runs = None
        if instance == 0 and not args.no_sdc_probe and not args.smoke:
            sdc_runs = await _sdc_probe_reps()

        # warm the engine once (NOT measured) so request #1 doesn't pay
        # one-time init; the *thermal* ramp is still included and discarded
        # later via the 60 s rule.
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
        tput_events = []  # (elapsed_s_float, delta_tokens)

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

        await asyncio.gather(*[asyncio.create_task(worker()) for _ in range(batch)])
        return per_prompt, tput_events, sdc_runs, t0_wall

    per_prompt, tput_events, sdc_runs, t0_wall = asyncio.run(drive())

    sdc_self_consistent = None
    if sdc_runs is not None and len(sdc_runs) >= 2:
        sdc_self_consistent = all(sdc_runs[0] == sdc_runs[i]
                                  for i in range(1, len(sdc_runs)))

    # 1 Hz throughput buckets over the full run.
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
        "batch": batch,
        "seq_len": seq,
        "gen_tokens": args.gen_tokens,
        "corpus_sha256": corpus_sha,
        "kv_capacity_seqs": kv_capacity,
        "t0_wall": t0_wall,
        "per_prompt": per_prompt,
        "per_second": per_second,
        "sdc_runs": sdc_runs,                       # list[list[int]] or None
        "sdc_self_consistent": sdc_self_consistent, # bool or None (None if probe skipped)
    }


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #
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
    """Stats on the scored window only (completion at/after discard_s)."""
    pp = [r for r in gpu_result["per_prompt"]
          if r["completion_elapsed_s"] >= discard_s]
    ps = [b["tokens_per_sec"] for b in gpu_result["per_second"]
          if b["elapsed_s"] > discard_s]
    return {
        "gpu_index": gpu_result["gpu_index"],
        "gpu_name": gpu_result["gpu_name"],
        "instances": len(gpu_result["kv_capacity_seqs"]),
        "kv_capacity_seqs": gpu_result["kv_capacity_seqs"],
        "scored_prompts": len(pp),
        "discarded_ramp_s": discard_s,
        "tokens_per_sec": _stats(ps),
        "ttft_ms": _stats([r["ttft_ms"] for r in pp]),
        "itl_p99_ms": _stats([r["itl_p99_ms"] for r in pp if r["itl_p99_ms"] is not None]),
    }


def _sdc_cross_gpu(results):
    """Per-GPU SDC verdict from cross-GPU majority-vote consensus + each
    GPU's within-GPU self-consistency flag. Returns dict keyed by gpu_index.

    For each generated-token position, the majority token across all GPUs
    that ran the probe is the consensus. A GPU's `cross_gpu_agreement_pct`
    is how often its token matches the consensus over the probe window;
    `first_disagreement_pos` is the first index it diverges. A GPU is
    flagged `suspect_compute_corruption` if either:
      - its 3 reps disagree (within-GPU self-consistency failed), or
      - its cross-GPU agreement is < 100%.
    Otherwise: `agree`.
    """
    from collections import Counter
    # Take rep 0 from each GPU as that GPU's canonical sequence (after the
    # self-consistency check already failed/passed for the within-GPU view).
    seqs = {r["gpu_index"]: r["sdc_runs"][0]
            for r in results
            if r.get("sdc_runs") and len(r["sdc_runs"]) >= 1
            and len(r["sdc_runs"][0]) > 0}
    if not seqs:
        return {}
    min_len = min(len(s) for s in seqs.values())
    consensus = []
    for pos in range(min_len):
        toks = [s[pos] for s in seqs.values()]
        consensus.append(Counter(toks).most_common(1)[0][0])

    out = {}
    by_gpu = {r["gpu_index"]: r for r in results}
    for gidx, s in seqs.items():
        agree = sum(1 for pos in range(min_len) if s[pos] == consensus[pos])
        first_dis = next((pos for pos in range(min_len)
                          if s[pos] != consensus[pos]), None)
        agreement_pct = (agree / min_len) * 100.0 if min_len else None
        self_cons = by_gpu[gidx].get("sdc_self_consistent")
        verdict = ("agree" if (agreement_pct == 100.0 and self_cons is True)
                   else "suspect_compute_corruption")
        out[gidx] = {
            "gpu_index": gidx,
            "self_consistent": self_cons,
            "cross_gpu_agreement_pct": round(agreement_pct, 4) if agreement_pct is not None else None,
            "first_disagreement_pos": first_dis,
            "verdict": verdict,
            "probe_tokens": min_len,
        }
    return out


def main():
    ap = argparse.ArgumentParser(description="Workload 1C — LLaMA-3 8B INT8 inference baseline")
    ap.add_argument("--model-dir", default="/workspace/model_int8")
    ap.add_argument("--corpus", default="/workspace/frozen_prompts.json")
    ap.add_argument("--max-vram-gb", type=float, default=32.0,
                    help="Per-GPU VRAM cap (passed by orchestrator). "
                         ">=24 -> 32gb config, >=12 -> 16gb config.")
    ap.add_argument("--batch", type=int, default=8,
                    help="prompts in flight per GPU")
    ap.add_argument("--seq", type=int, default=512,
                    help="prompt sequence length")
    ap.add_argument("--gen-tokens", type=int, default=128)
    ap.add_argument("--duration", type=int, default=420, help="seconds (spec: 7 min)")
    ap.add_argument("--discard-ramp", type=int, default=60, help="seconds excluded from stats")
    ap.add_argument("--gpus", default="all", help='"all" or e.g. "0,1,2"')
    ap.add_argument("--gpu-mem-util", type=float, default=0.90,
                    help="GPU memory fraction shared by all instances on a GPU")
    ap.add_argument("--instances-per-gpu", type=int, default=1,
                    help="independent vLLM engines per GPU")
    ap.add_argument("--gpu-mode", choices=["parallel", "sequential"], default="parallel",
                    help="run all GPUs at once, or one GPU at a time")
    ap.add_argument("--quantization", default="gptq", help="vLLM quantization kernel")
    ap.add_argument("--enforce-eager", type=_flag, default=True,
                    help="true disables CUDA graphs")
    ap.add_argument("--mps", type=_flag, default=False,
                    help="run the CUDA MPS daemon for the duration of the run")
    ap.add_argument("--output-dir", default="/workspace/results")
    ap.add_argument("--no-sdc-probe", action="store_true",
                    help="Skip the silent-compute-error (SDC) probe. By "
                         "default, each GPU runs a short deterministic "
                         f"probe ({SDC_PROBE_REPS} reps x {SDC_PROBE_TOKENS} "
                         "tokens) before the main run; main() cross-checks "
                         "within-GPU self-consistency and cross-GPU "
                         "majority-vote agreement to flag silent compute "
                         "corruption. Auto-skipped under --smoke.")
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
        print("  Workload 1C — SMOKE (NOT a benchmark): vLLM path shakeout")
        print(f"  model      : {SMOKE_MODEL} (ungated, unquantized)")
    else:
        print("  Workload 1C — LLaMA-3 8B INT8 (GPTQ) integrated baseline")
    print(f"  shape      : batch={shape['batch']} seq={shape['seq']}")
    print(f"  gen tokens : {args.gen_tokens}   duration: {args.duration}s "
          f"(discard first {args.discard_ramp}s)")
    print(f"  GPUs       : {gpus} ({args.gpu_mode}, {args.instances_per_gpu} "
          f"independent vLLM per GPU)")
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
        print("  All GPU workers failed:", flush=True)
        for r in fatal:
            print(f"   GPU {r['gpu_index']}.{r['instance_index']}: {r['fatal']}", flush=True)
        sys.exit(1)

    per_gpu_stats = [_summarize(r, args.discard_ramp) for r in results]

    def _agg(field, sub):
        import numpy as np
        vals = [g[field][sub] for g in per_gpu_stats if g[field][sub] is not None]
        return float(np.mean(vals)) if vals else None

    # SDC probe cross-GPU consensus (None if probe skipped on all GPUs).
    sdc_per_gpu = _sdc_cross_gpu(results) if not args.no_sdc_probe and not args.smoke else {}
    sdc_meta = {
        "enabled": (not args.no_sdc_probe) and (not args.smoke),
        "reps_per_gpu": SDC_PROBE_REPS,
        "tokens_per_rep": SDC_PROBE_TOKENS,
        "method": "cross-GPU majority-vote consensus + within-GPU self-consistency",
        "per_gpu": [sdc_per_gpu[g] for g in sorted(sdc_per_gpu.keys())],
        "all_pass": (bool(sdc_per_gpu)
                     and all(v["verdict"] == "agree" for v in sdc_per_gpu.values())),
        "suspected_gpus": [v["gpu_index"] for v in sdc_per_gpu.values()
                            if v["verdict"] != "agree"],
    } if sdc_per_gpu else {
        "enabled": (not args.no_sdc_probe) and (not args.smoke),
        "reps_per_gpu": SDC_PROBE_REPS,
        "tokens_per_rep": SDC_PROBE_TOKENS,
        "method": "cross-GPU majority-vote consensus + within-GPU self-consistency",
        "per_gpu": [], "all_pass": None, "suspected_gpus": [],
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    meta = {
        "node_id": node_id,
        "run_id": run_id,
        "workload_num": WORKLOAD_NUM,
        "workload": "1C_llama3_8b_int8_integrated_baseline",
        "smoke": args.smoke,
        "spec": {
            "model": (f"{SMOKE_MODEL} (SMOKE shakeout — ungated, unquantized)"
                      if args.smoke else
                      "meta-llama/Meta-Llama-3-8B-Instruct (quantized in-container, AutoGPTQ INT8)"),
            "precision": "fp16-smoke" if args.smoke else "int8-gptq",
            "decoding": "greedy (temperature=0, ignore_eos)",
            "batch": shape["batch"], "seq_len": shape["seq"],
            "gen_tokens": args.gen_tokens,
            "instances_per_gpu": args.instances_per_gpu,
            "gpu_mode": args.gpu_mode,
            "quantization": None if args.smoke else args.quantization,
            "duration_s": args.duration, "discarded_ramp_s": args.discard_ramp,
            "scored_window_s": args.duration - args.discard_ramp,
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
        "sdc_probe": sdc_meta,
        "per_gpu_stats": per_gpu_stats,
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
    print("  1C SUMMARY (scored window only)")
    for g in per_gpu_stats:
        tp = g["tokens_per_sec"]; tt = g["ttft_ms"]; il = g["itl_p99_ms"]
        print(f"  GPU {g['gpu_index']} ({g['gpu_name']}) prompts={g['scored_prompts']}")
        print(f"    tokens/s  mean={tp['mean']:.1f} sd={tp['stddev']:.2f}"
              if tp['mean'] is not None else "    tokens/s  n/a")
        print(f"    TTFT ms   mean={tt['mean']:.1f} p99={tt['p99']:.1f}"
              if tt['mean'] is not None else "    TTFT ms   n/a")
        print(f"    ITL p99   mean={il['mean']:.2f}"
              if il['mean'] is not None else "    ITL p99   n/a")

    if sdc_meta.get("enabled") and sdc_meta.get("per_gpu"):
        print(f"\n  SDC PROBE (cross-GPU consensus + within-GPU self-consistency)")
        for v in sdc_meta["per_gpu"]:
            print(f"    GPU {v['gpu_index']}  self_consistent={v['self_consistent']}  "
                  f"cross_gpu_agreement={v['cross_gpu_agreement_pct']}%  "
                  f"verdict={v['verdict']}")
        if sdc_meta["all_pass"]:
            print(f"    -> all 8 GPUs agree (no silent compute corruption detected)")
        else:
            print(f"    -> SUSPECTED GPUs: {sdc_meta['suspected_gpus']}")
    elif sdc_meta.get("enabled"):
        print(f"\n  SDC PROBE: enabled but no probe data collected.")

    print(f"{'='*64}")
    print(f"  metadata -> {mpath}")
    print(f"  throughput csv -> {tpath}")
    print(f"  per-prompt csv -> {ppath}\n", flush=True)


if __name__ == "__main__":
    main()
