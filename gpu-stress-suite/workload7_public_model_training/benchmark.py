"""
Workload 7: Public Dataset Open-Model Training
Category  : Model-based training
Purpose   : Run deterministic causal-LM training using a public Hugging Face
            dataset and an open model ID.

The workload is adaptive: it tries to fit the requested model within available
GPU memory by reducing batch size and sequence length. If the requested model is
too large, it can fall back to smaller configured model IDs.
"""

import argparse
import copy
import gc
import json
import statistics
import time
from datetime import datetime, timezone
from pathlib import Path

import torch


def utc_stamp():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def get_gpu_info():
    info = []
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        info.append(
            {
                "index": i,
                "name": props.name,
                "total_memory_gb": round(props.total_memory / 1024**3, 2),
                "cuda_capability": f"{props.major}.{props.minor}",
            }
        )
    return info


def write_result(payload, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_id = payload["run_id"]
    target = output_path / f"workload7_{run_id}.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Results saved -> {target}", flush=True)


def skipped_payload(run_id, args, reason, attempts=None):
    return {
        "run_id": run_id,
        "workload": "workload7_public_model_training",
        "description": "Public dataset open-model training",
        "status": "skipped",
        "skip_reason": reason,
        "benchmark_args": vars(args),
        "gpu_hardware": get_gpu_info(),
        "attempts": attempts or [],
        "results": [],
    }


def import_hf():
    try:
        from datasets import load_dataset
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:
        raise RuntimeError(
            "Missing Hugging Face dependencies. Rebuild with INSTALL_HF_DEPS=1 "
            "or install transformers, datasets, accelerate, sentencepiece, and pyarrow."
        ) from exc
    return load_dataset, AutoModelForCausalLM, AutoTokenizer


def load_dataset_texts(load_dataset, args):
    dataset = load_dataset(args.dataset_id, args.dataset_config, split=args.dataset_split)
    texts = []
    for row in dataset:
        text = str(row.get(args.text_column, "")).strip()
        if text:
            texts.append(text)
        if len(texts) >= args.max_samples:
            break
    if not texts:
        raise RuntimeError(f"No non-empty text found in dataset column '{args.text_column}'.")
    return texts


def model_ids_to_try(args):
    ids = [args.model_id]
    for item in args.fallback_model_ids:
        if item not in ids:
            ids.append(item)
    return ids


def size_candidates(value, minimum):
    candidates = []
    current = max(value, minimum)
    while current >= minimum:
        candidates.append(current)
        if current == minimum:
            break
        current = max(minimum, current // 2)
    return candidates


def prepare_batches(tokenizer, texts, batch_size, block_size, device, vocab_size=None):
    encoded = tokenizer(
        texts,
        truncation=True,
        padding="max_length",
        max_length=block_size,
        return_tensors="pt",
    )
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    if vocab_size is not None:
        invalid_mask = (input_ids < 0) | (input_ids >= vocab_size)
        if bool(invalid_mask.any()):
            print(
                f"  Token IDs exceeded model vocab ({vocab_size}); remapping for tiny-model compatibility.",
                flush=True,
            )
            input_ids = torch.remainder(input_ids, vocab_size)

    batches = []
    for start in range(0, input_ids.shape[0], batch_size):
        end = start + batch_size
        if end > input_ids.shape[0]:
            break
        batches.append(
            {
                "input_ids": input_ids[start:end].to(device),
                "attention_mask": attention_mask[start:end].to(device),
            }
        )
    if not batches:
        raise RuntimeError("No full training batches were created after adaptive sizing.")
    return batches


def load_model_and_tokenizer(model_id, AutoModelForCausalLM, AutoTokenizer):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    try:
        model = AutoModelForCausalLM.from_pretrained(model_id, attn_implementation="eager")
    except TypeError:
        model = AutoModelForCausalLM.from_pretrained(model_id)
    return tokenizer, model


def cleanup_cuda(*objects):
    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def train_attempt(model_id, texts, args, batch_size, block_size, AutoModelForCausalLM, AutoTokenizer):
    device = torch.device("cuda:0")
    tokenizer, model = load_model_and_tokenizer(model_id, AutoModelForCausalLM, AutoTokenizer)
    model.to(device)
    model.train()

    vocab_size = getattr(model.config, "vocab_size", None)
    effective_block_size = block_size
    max_positions = getattr(model.config, "max_position_embeddings", None)
    if max_positions is not None and effective_block_size > int(max_positions):
        effective_block_size = max(8, int(max_positions))
        print(
            f"  Requested block={block_size} exceeds model position limit; clamping to {effective_block_size}.",
            flush=True,
        )

    batches = prepare_batches(
        tokenizer,
        texts,
        batch_size,
        effective_block_size,
        device,
        vocab_size,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)

    step_times = []
    losses = []
    deadline = time.perf_counter() + args.duration
    step = 0

    while step < args.max_steps and time.perf_counter() < deadline:
        batch = batches[step % len(batches)]
        torch.cuda.synchronize()
        start = time.perf_counter()

        labels = batch["input_ids"].clone()
        labels[batch["attention_mask"].eq(0)] = -100

        outputs = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=labels,
        )
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        step_times.append(elapsed)
        losses.append(float(loss.detach().cpu()))
        step += 1

    tokens_per_step = batch_size * effective_block_size
    total_tokens = tokens_per_step * step
    total_train_time = sum(step_times)
    ordered_times = sorted(step_times)
    p95_index = int(round((len(ordered_times) - 1) * 0.95)) if ordered_times else 0

    result = {
        "model_id": model_id,
        "dataset_id": args.dataset_id,
        "dataset_config": args.dataset_config,
        "dataset_split": args.dataset_split,
        "steps": step,
        "batch_size": batch_size,
        "requested_block_size": block_size,
        "block_size": effective_block_size,
        "total_tokens": total_tokens,
        "tokens_per_second": round(total_tokens / total_train_time, 3) if total_train_time > 0 else 0.0,
        "step_time_mean_s": round(statistics.mean(step_times), 6) if step_times else 0.0,
        "step_time_p95_s": round(ordered_times[p95_index], 6) if ordered_times else 0.0,
        "loss_first": round(losses[0], 6) if losses else None,
        "loss_last": round(losses[-1], 6) if losses else None,
        "loss_mean": round(statistics.mean(losses), 6) if losses else None,
        "peak_memory_gb": round(torch.cuda.max_memory_allocated(device) / 1024**3, 3),
    }

    cleanup_cuda(model, optimizer, batches)
    return result


def train(args):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this workload.")

    torch.manual_seed(args.seed)
    torch.cuda.reset_peak_memory_stats()

    load_dataset, AutoModelForCausalLM, AutoTokenizer = import_hf()
    texts = load_dataset_texts(load_dataset, args)
    attempts = []

    for model_id in model_ids_to_try(args):
        for batch_size in size_candidates(args.batch_size, 1):
            for block_size in size_candidates(args.block_size, args.min_block_size):
                attempt = {
                    "model_id": model_id,
                    "batch_size": batch_size,
                    "block_size": block_size,
                }
                try:
                    print(
                        f"  Trying model={model_id}, batch={batch_size}, block={block_size}",
                        flush=True,
                    )
                    result = train_attempt(
                        model_id,
                        texts,
                        args,
                        batch_size,
                        block_size,
                        AutoModelForCausalLM,
                        AutoTokenizer,
                    )
                    attempt["status"] = "completed"
                    attempts.append(attempt)
                    return result, attempts
                except torch.cuda.OutOfMemoryError as exc:
                    cleanup_cuda()
                    attempt["status"] = "oom"
                    attempt["error"] = str(exc).splitlines()[0]
                    attempts.append(attempt)
                    print("  OOM - trying smaller training shape.", flush=True)
                except RuntimeError as exc:
                    cleanup_cuda()
                    message = str(exc)
                    if "out of memory" in message.lower():
                        attempt["status"] = "oom"
                        attempt["error"] = message.splitlines()[0]
                        attempts.append(attempt)
                        print("  OOM - trying smaller training shape.", flush=True)
                    else:
                        attempt["status"] = "failed"
                        attempt["error"] = message
                        attempts.append(attempt)
                        raise

    raise RuntimeError("No model/batch/block configuration fit within the available GPU memory.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--model-id", default="hf-internal-testing/tiny-random-LlamaForCausalLM")
    parser.add_argument(
        "--fallback-model-ids",
        nargs="*",
        default=["hf-internal-testing/tiny-random-LlamaForCausalLM"],
    )
    parser.add_argument("--dataset-id", default="wikitext")
    parser.add_argument("--dataset-config", default="wikitext-2-raw-v1")
    parser.add_argument("--dataset-split", default="train")
    parser.add_argument("--text-column", default="text")
    parser.add_argument("--max-samples", type=int, default=128)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--block-size", type=int, default=128)
    parser.add_argument("--min-block-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--output-dir", default="/workspace/results")
    parser.add_argument("--max-vram-gb", type=float, default=12.0)
    args = parser.parse_args()

    run_id = utc_stamp()

    try:
        result, attempts = train(args)
    except Exception as exc:
        payload = skipped_payload(run_id, args, str(exc), attempts=locals().get("attempts", []))
        write_result(payload, args.output_dir)
        print(f"  SKIPPED: {exc}", flush=True)
        return

    payload = {
        "run_id": run_id,
        "workload": "workload7_public_model_training",
        "description": "Public dataset open-model training",
        "status": "completed",
        "benchmark_args": vars(args),
        "gpu_hardware": get_gpu_info(),
        "attempts": attempts,
        "results": [result],
    }
    write_result(payload, args.output_dir)


if __name__ == "__main__":
    main()
