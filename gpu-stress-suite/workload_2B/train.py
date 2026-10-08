"""
Workload 2B — LLM fine-tuning (write-path diagnostic).

Fine-tunes a Llama-3 family model: by default LoRA adapters on a frozen FP16
base, or every parameter with --training full. With LoRA the frozen base is
read heavily but never written; the adapters and AdamW optimizer state are
the write-heavy components. Gradient checkpointing recomputes activations on
the backward pass — additional sustained HBM write traffic. That's the
diagnostic point of 2B: a GPU with a degraded HBM write path can look healthy
in 1C/2A/3A but will misbehave here.

One independent training process per visible GPU (replicated, NOT DDP /
NOT FSDP). Per-GPU health signal is preserved exactly as in 1C/2A/3A.

Determinism: torch.use_deterministic_algorithms(True, warn_only=True) +
fixed seed + sequential dataset slice. Bitwise-identical loss curve is the
target; residual variance under identical inputs is the hardware signal.

Outputs (per run, in --output-dir):
  {run_id}_2B_metadata.json     run config, env, per-GPU + aggregate stats, dataset SHA
  {run_id}_2B_per_step.csv      one row per training step
"""

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import torch
import torch.multiprocessing as mp

TIMEZONE = 0  # UTC for standardization
SEED = 0

WORKLOAD_NUM = 10  # 2B in the orchestrator's numbering (run_all.sh phase 7)




# Recipe spec: Llama-3 8B + batch 64 seq 1024 grad-accum 4 (effective batch
# 256). The base model and shape come from the workload_2B block of each GPU
# profile under hardware/gpu/, sized to that GPU's memory. The defaults below
# fit a 16 GB GPU: Llama-3.2-3B-Instruct (same Llama-3 family, ~6 GB FP16) at
# seq 256. A shorter seq shrinks the lm_head logits tensor (128 256-token
# vocabulary, promoted to fp32 by the cross-entropy loss), the dominant
# memory hot-spot.

# --smoke: 10 steps on a tiny ungated model with tiny shape, no HF_TOKEN.
# Exercises the full training plumbing (load + LoRA wrap + forward +
# backward + AdamW step + CSV write) without needing the 8B INT8 base.
SMOKE_SHAPE = {"batch": 2, "seq": 64, "grad_accum": 1, "steps": 10}
SMOKE_MODEL = "facebook/opt-125m"

# Alpaca slice — 4 800 samples cycled deterministically across 300 steps × 256
# effective-batch = 76 800 samples. Cycling is sequential (no shuffle); the
# slice's SHA-256 is recorded in metadata for reproducibility.
ALPACA_SLICE_END = 4800
LR = 2e-4
FULL_LR = 1e-5
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.0
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALPACA_TEMPLATE = (
    "Below is an instruction that describes a task. Write a response that "
    "appropriately completes the request.\n\n"
    "### Instruction:\n{instruction}\n\n"
    "{input_block}"
    "### Response:\n{output}"
)


def _shape(args):
    if getattr(args, "smoke", False):
        return dict(SMOKE_SHAPE)
    return {"batch": args.batch, "seq": args.seq,
            "grad_accum": args.grad_accum, "steps": args.steps}


def _seed_everything(seed: int) -> None:
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _format_alpaca(sample: dict) -> str:
    instr = sample.get("instruction", "")
    inp = sample.get("input", "")
    out = sample.get("output", "")
    input_block = f"### Input:\n{inp}\n\n" if inp else ""
    return ALPACA_TEMPLATE.format(instruction=instr, input_block=input_block, output=out)


def _slice_sha(samples: list[dict]) -> str:
    h = hashlib.sha256()
    for s in samples:
        h.update(_format_alpaca(s).encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()


def _make_smoke_corpus(n: int) -> list[dict]:
    """Tiny synthetic Alpaca-like dataset for --smoke. No external download."""
    return [
        {"instruction": f"Add {i} and {i + 1}.", "input": "",
         "output": f"The answer is {i + i + 1}."}
        for i in range(n)
    ]


def _tokenize_batch(tokenizer, samples: list[dict], seq_len: int):
    texts = [_format_alpaca(s) for s in samples]
    enc = tokenizer(texts, padding="max_length", truncation=True,
                    max_length=seq_len, return_tensors="pt")
    enc["labels"] = enc["input_ids"].clone()
    # Mask padding from the loss (-100 = ignored).
    enc["labels"][enc["attention_mask"] == 0] = -100
    return enc


def _load_dataset(args):
    """Returns (samples_list, slice_sha) — deterministic Alpaca slice."""
    if args.smoke:
        corpus = _make_smoke_corpus(64)
        return corpus, _slice_sha(corpus)
    from datasets import load_dataset
    ds = load_dataset("tatsu-lab/alpaca", split=f"train[:{ALPACA_SLICE_END}]")
    samples = [dict(s) for s in ds]
    return samples, _slice_sha(samples)


def _load_base_model(args, gpu_id: int):
    """Returns the FP16 base model on CUDA, ready for LoRA wrapping.

    PRECISION DEVIATION (forced by stack incompatibility):
      Recipe asks for INT8 base + LoRA. auto-gptq 0.7.1's GPTQLoraModel
      does not compose with peft 0.11.1 (RecursionError during wrap), and
      bitsandbytes has unreliable Volta sm_70 support. We use the FP16
      base directly. 2B's diagnostic value is the HBM **write path**
      (gradients, optimizer state, activation-checkpoint recompute) — the
      base is frozen/read-only, so the write-path signal is identical
      regardless of base precision. Memory still fits a 32 GB V100
      comfortably (~16 GB FP16 base + ~3 GB activations w/ grad
      checkpointing + tiny LoRA/AdamW = ~20 GB).
    """
    from transformers import AutoModelForCausalLM
    model_id = SMOKE_MODEL if args.smoke else args.base_model
    revision = None if args.smoke else (args.revision or None)
    m = AutoModelForCausalLM.from_pretrained(
        model_id, revision=revision, torch_dtype=torch.float16,
    )
    return m.to("cuda")


def _prepare_for_training(base, args):
    from peft import LoraConfig, get_peft_model
    cfg = LoraConfig(
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
        target_modules=("all-linear" if args.smoke else LORA_TARGETS),
        bias="none", task_type="CAUSAL_LM",
    )
    if not args.smoke:
        # Gradient checkpointing fits batch 64 seq 1024 activations within
        # 32 GB on a V100. enable_input_require_grads lets gradient flow
        # through the frozen embedding into the LoRA adapters.
        base.gradient_checkpointing_enable()
        base.enable_input_require_grads()
    # Full fine-tuning trains every base parameter; LoRA trains only the adapters.
    model = base if args.training == "full" else get_peft_model(base, cfg)
    # PEFT's default LoRA adapter dtype follows the base (fp16 here). AdamW
    # state then inherits fp16 too, and AdamW on fp16 params is numerically
    # unstable: small grad^2 terms underflow to 0, the resulting update
    # `m / sqrt(v + eps)` blows up, params go NaN. Standard QLoRA fix is
    # to keep trainable params in fp32 so AdamW state runs in fp32.
    if not args.smoke:
        for p in model.parameters():
            if p.requires_grad:
                p.data = p.data.to(torch.float32)
    return model


def _run_on_gpu(gpu_id, args):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    os.environ.setdefault("DO_NOT_TRACK", "1")
    # CUBLAS_WORKSPACE_CONFIG is also set in the Dockerfile ENV; reinforced here
    # so a bare `python train.py` (no docker) is also deterministic.
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    import torch as _torch
    _torch.manual_seed(SEED)
    _torch.cuda.manual_seed_all(SEED)
    # warn_only=True so an auto-gptq kernel without a deterministic CUDA path
    # warns rather than crashes; we treat residual variation as the hardware
    # signal (same posture as 1C/2A's continuous-batching nondeterminism).
    _torch.use_deterministic_algorithms(True, warn_only=True)

    dev_name = _torch.cuda.get_device_name(0)

    from transformers import AutoTokenizer
    tok_src = SMOKE_MODEL if args.smoke else args.base_model
    tok_rev = None if args.smoke else (args.revision or None)
    tokenizer = AutoTokenizer.from_pretrained(tok_src, revision=tok_rev, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        base = _load_base_model(args, gpu_id)
        model = _prepare_for_training(base, args)
    except Exception as e:
        print(f"  [GPU {gpu_id}] FATAL load/wrap: {e!r}", flush=True)
        return {"gpu_index": gpu_id, "fatal": f"load_or_wrap: {e!r}"}

    model.train()
    shape = _shape(args)
    batch, seq, grad_accum, steps = (shape["batch"], shape["seq"],
                                      shape["grad_accum"], shape["steps"])

    # Deterministic sequential cycling over the dataset slice.
    samples, slice_sha = _load_dataset(args)
    print(f"  [GPU {gpu_id}] dataset slice: {len(samples)} samples sha256={slice_sha[:16]}...",
          flush=True)

    full = args.training == "full"
    # Fused AdamW avoids foreach temporaries the size of the full model.
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=FULL_LR if full else LR, betas=(0.9, 0.95), fused=True if full else None,
    )

    records = []
    total_samples_consumed = 0

    def _next_batch():
        nonlocal total_samples_consumed
        idx0 = total_samples_consumed
        idxs = [(idx0 + i) % len(samples) for i in range(batch)]
        total_samples_consumed += batch
        return _tokenize_batch(tokenizer, [samples[i] for i in idxs], seq)

    print(f"  [GPU {gpu_id}] starting {steps} steps  batch={batch} seq={seq} "
          f"grad_accum={grad_accum} effective_batch={batch * grad_accum}",
          flush=True)

    for step in range(steps):
        t0 = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(grad_accum):
            enc = _next_batch()
            enc = {k: v.to("cuda", non_blocking=True) for k, v in enc.items()}
            out = model(**enc)
            loss = out.loss / grad_accum
            loss.backward()
            step_loss += loss.item() * grad_accum
        # grad norm (norm over ALL trainable params, before clipping)
        with torch.no_grad():
            sq = 0.0
            for p in model.parameters():
                if p.grad is not None:
                    sq += float((p.grad.detach() ** 2).sum().item())
            grad_norm = sq ** 0.5
        optimizer.step()
        step_time = time.perf_counter() - t0
        avg_loss = step_loss / grad_accum
        records.append({"step": step, "step_time_s": round(step_time, 6),
                         "loss": round(avg_loss, 8),
                         "grad_norm": round(grad_norm, 8)})
        if step == 0 or (step + 1) % 25 == 0 or step == steps - 1:
            print(f"  [GPU {gpu_id}] step {step + 1}/{steps}  "
                  f"loss={avg_loss:.4f}  step_time={step_time:.3f}s  "
                  f"grad_norm={grad_norm:.4f}", flush=True)

    return {
        "gpu_index": gpu_id,
        "gpu_name": dev_name,
        "batch": batch, "seq_len": seq, "grad_accum": grad_accum,
        "effective_batch": batch * grad_accum,
        "steps": steps,
        "dataset_slice_sha256": slice_sha,
        "dataset_slice_size": len(samples),
        "records": records,
    }


def _resolve_gpus(spec):
    n = torch.cuda.device_count()
    if spec == "all":
        return list(range(n))
    return [int(x) for x in spec.split(",") if x.strip() != ""]


def _stats(xs):
    if not xs:
        return {"count": 0, "mean": None, "variance": None, "stddev": None,
                "p50": None, "p95": None, "p99": None}
    import numpy as np
    a = np.asarray(xs, dtype="float64")
    return {"count": int(a.size), "mean": float(a.mean()),
            "variance": float(a.var(ddof=0)), "stddev": float(a.std(ddof=0)),
            "p50": float(np.percentile(a, 50)),
            "p95": float(np.percentile(a, 95)),
            "p99": float(np.percentile(a, 99))}


def _summarize(res, discard_steps):
    rs = res["records"]
    scored = rs[discard_steps:] if discard_steps else rs
    return {
        "gpu_index": res["gpu_index"],
        "gpu_name": res["gpu_name"],
        "steps_run": len(rs),
        "scored_steps": len(scored),
        "discarded_steps": discard_steps,
        "step_time_s": _stats([r["step_time_s"] for r in scored]),
        "loss": _stats([r["loss"] for r in scored]),
        "grad_norm": _stats([r["grad_norm"] for r in scored]),
        "loss_first": rs[0]["loss"] if rs else None,
        "loss_last": rs[-1]["loss"] if rs else None,
    }


def main():
    ap = argparse.ArgumentParser(description="Workload 2B — LLaMA-3 8B INT8 LoRA fine-tuning")
    ap.add_argument("--base-model", default="meta-llama/Llama-3.2-3B-Instruct",
                    help="HF repo ID or local path to base model (FP16).")
    ap.add_argument("--revision", default=None,
                    help="Pin the source repo to a commit SHA.")
    ap.add_argument("--max-vram-gb", type=float, default=32.0,
                    help="Per-GPU VRAM cap (passed by the orchestrator).")
    ap.add_argument("--duration", type=float, default=None,
                    help="Accepted from the orchestrator for uniformity; "
                         "2B is step-bounded, so this is ignored "
                         "(runtime is controlled by --steps).")
    ap.add_argument("--gpus", default="all", help='"all" or e.g. "0,1,2"')
    ap.add_argument("--discard-warmup-steps", type=int, default=10,
                    help="exclude the first N steps from summary stats "
                         "(JIT compile, allocator warmup)")
    ap.add_argument("--seq", type=int, default=256,
                    help="tokenizer max_length")
    ap.add_argument("--batch", type=int, default=16,
                    help="micro-batch per GPU")
    ap.add_argument("--grad-accum", type=int, default=8,
                    help="gradient accumulation steps")
    ap.add_argument("--steps", type=int, default=200,
                    help="optimizer steps to run")
    ap.add_argument("--training", choices=["lora", "full"], default="lora",
                    help="lora trains adapters on a frozen base; full trains every parameter")
    ap.add_argument("--output-dir", default="/workspace/results")
    ap.add_argument("--smoke", action="store_true",
                    help="pre-delivery shakeout: tiny ungated model "
                         f"({SMOKE_MODEL}), tiny sizes, synthetic dataset, "
                         "no HF_TOKEN, 10 steps. NOT training.")
    args = ap.parse_args()

    if args.smoke:
        # Smoke runs only 10 steps; the default 10-step discard would leave
        # nothing scored. Use 0 so the smoke summary reflects all 10 steps.
        args.discard_warmup_steps = 0

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices. Run the container with --gpus all.")
    gpus = _resolve_gpus(args.gpus)
    shape = _shape(args)
    node_id = os.environ.get("NODE_ID", "")

    print(f"\n{'='*64}")
    if args.smoke:
        print("  Workload 2B — SMOKE (NOT training): plumbing shakeout")
        print(f"  model      : {SMOKE_MODEL} (ungated, unquantized)")
    else:
        print(f"  Workload 2B — LLaMA-3 8B + {'full' if args.training == 'full' else 'LoRA'} fine-tuning")
    print(f"  shape      : batch={shape['batch']} "
          f"seq={shape['seq']} grad_accum={shape['grad_accum']}")
    print(f"  steps      : {shape['steps']}  (effective batch {shape['batch'] * shape['grad_accum']})")
    print(f"  GPUs       : {gpus} (one independent process per GPU)")
    print(f"{'='*64}\n", flush=True)

    run_dt = datetime.now(timezone(timedelta(hours=TIMEZONE)))
    run_id = run_dt.strftime("%y-%m-%d_%H-%M")

    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=len(gpus)) as pool:
        results = pool.starmap(_run_on_gpu, [(g, args) for g in gpus])

    fatal = [r for r in results if r.get("fatal")]
    results = [r for r in results if not r.get("fatal")]
    if not results:
        for r in fatal:
            print(f"  GPU {r['gpu_index']} fatal: {r['fatal']}", flush=True)
        sys.exit(1)

    discard = min(args.discard_warmup_steps, max(0, shape["steps"] - 1))
    per_gpu = [_summarize(r, discard) for r in results]

    def _agg(field, sub):
        import numpy as np
        vals = [g[field][sub] for g in per_gpu if g[field][sub] is not None]
        return float(np.mean(vals)) if vals else None

    full = args.training == "full"
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    meta = {
        "node_id": node_id,
        "run_id": run_id,
        "workload_num": WORKLOAD_NUM,
        "workload": "2B_llama3_8b_full_finetune" if full else "2B_llama3_8b_fp16_lora_finetune",
        "smoke": args.smoke,
        "spec": {
            "model": (f"{SMOKE_MODEL} (SMOKE shakeout — ungated)"
                      if args.smoke else
                      f"{args.base_model} (full fine-tune)" if full else
                      f"{args.base_model} (FP16 base, frozen) + PEFT LoRA"),
            "base_precision": "fp16-smoke" if args.smoke else
                              "fp32 (trainable)" if full else "fp16 (frozen)",
            "adapter": "none (full fine-tune)" if full else
                       "LoRA fp16" if args.smoke else
                       f"LoRA fp16 (r={LORA_R}, alpha={LORA_ALPHA}, targets={LORA_TARGETS})",
            "optimizer": f"AdamW fused lr={FULL_LR}" if full else f"AdamW lr={LR}",
            "batch": shape["batch"], "seq_len": shape["seq"],
            "grad_accum": shape["grad_accum"],
            "effective_batch": shape["batch"] * shape["grad_accum"],
            "steps": shape["steps"],
            "gradient_checkpointing": not args.smoke,
            "deviation_notes": [
                "Recipe specifies Llama-3 8B INT8 base + LoRA. Implementation "
                "uses FP16 base + LoRA (INT8 path unavailable: auto-gptq "
                "0.7.1's GPTQLoraModel does not compose with peft 0.11.x — "
                "RecursionError; bitsandbytes Volta sm_70 support unreliable). "
                "The base model comes from the GPU profile: "
                "Llama-3-8B-Instruct where it fits, Llama-3.2-3B-Instruct "
                "(same Llama-3 family) on 16 GB GPUs, since the 8B FP16 base "
                "alone is 16 GB. The 2B diagnostic signal is the HBM write "
                "path (gradient, optimizer state, activation-checkpoint "
                "recompute).",
                "Shape comes from the GPU profile. Sequence lengths below the "
                "recipe's 1024 shrink the lm_head logits tensor (Llama-3's "
                "128 256-token vocabulary, fp32-promoted by the cross-entropy "
                "loss), the dominant memory hot-spot.",
                "Determinism: torch.use_deterministic_algorithms(True, "
                "warn_only=True). Residual step-to-step variance under "
                "identical inputs is the hardware signal this suite is "
                "designed to surface.",
                "Hardware telemetry (power, temperature, memory, ECC) is "
                "recorded by the suite's host-side loggers, not by this "
                "workload.",
            ] + (["Full fine-tuning (--training full): every parameter is "
                  "trainable in fp32; no LoRA adapters."] if full else []),
        },
        "env": {
            "torch": torch.__version__,
            "cuda": torch.version.cuda,
            "seed": SEED,
            "cublas_workspace_config": os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
        },
        "dataset": {
            "source": "synthetic-smoke" if args.smoke else "tatsu-lab/alpaca",
            "slice": "first 64 synthetic" if args.smoke else f"train[:{ALPACA_SLICE_END}]",
            "slice_sha256": results[0]["dataset_slice_sha256"],
            "slice_size": results[0]["dataset_slice_size"],
            "cycling": "sequential modulo slice size",
        },
        "per_gpu_summary": per_gpu,
        "aggregate": {
            "gpu_count": len(results),
            "step_time_s_mean": _agg("step_time_s", "mean"),
            "loss_mean": _agg("loss", "mean"),
            "grad_norm_mean": _agg("grad_norm", "mean"),
        },
        "failed_gpus": fatal,
    }
    mpath = Path(args.output_dir) / f"{run_id}_workload{WORKLOAD_NUM}_metadata.json"
    with open(mpath, "w") as f:
        json.dump(meta, f, indent=2)

    cpath = Path(args.output_dir) / f"{run_id}_workload{WORKLOAD_NUM}_per_step.csv"
    with open(cpath, "w") as f:
        f.write("gpu_index,step,step_time_s,loss,grad_norm\n")
        for r in results:
            for rec in r["records"]:
                f.write(f"{r['gpu_index']},{rec['step']},"
                        f"{rec['step_time_s']},{rec['loss']},{rec['grad_norm']}\n")

    print(f"\n{'='*64}")
    print("  2B SUMMARY (scored window only)")
    for g in per_gpu:
        st = g["step_time_s"]; ls = g["loss"]; gn = g["grad_norm"]
        print(f"  GPU {g['gpu_index']} ({g['gpu_name']}) "
              f"steps={g['steps_run']} (scored {g['scored_steps']})")
        if st["mean"] is not None:
            print(f"    step_time s  mean={st['mean']:.3f} sd={st['stddev']:.4f} "
                  f"p99={st['p99']:.3f}")
        if ls["mean"] is not None:
            print(f"    loss         first={g['loss_first']:.4f} last={g['loss_last']:.4f} "
                  f"mean(scored)={ls['mean']:.4f}")
        if gn["mean"] is not None:
            print(f"    grad_norm    mean={gn['mean']:.4f}")
    print(f"{'='*64}")
    print(f"  metadata -> {mpath}")
    print(f"  per-step csv -> {cpath}\n", flush=True)


if __name__ == "__main__":
    main()
