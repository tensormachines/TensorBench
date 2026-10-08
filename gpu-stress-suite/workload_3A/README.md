# Workload 3A — LLaMA-2 13B FP16 memory-pressure characterization

Self-contained. `scp` this directory to the box and run `./run.sh`.

## One command

```bash
HF_TOKEN=hf_xxx ./run.sh                   # batch 1, seq 2048
HF_TOKEN=hf_xxx ./run.sh --seq 4096
HF_TOKEN=hf_xxx ./run.sh --gpus 0          # only GPU 0
```

`HF_TOKEN` must have the `meta-llama/Llama-2-13b-hf` license accepted. Token is
passed by environment only, never written into the image.

## What it does

This is a **failure characterization** test, not a throughput test. Full FP16
13B weights (~26 GB) plus a long KV cache push the GPU toward its memory
limit. By default each visible GPU is exercised **one at a time, sequentially**
with one prompt in flight. The result of interest is the *shape of the response*:

- **graceful** — KV eviction/recompute, latency penalty absorbed, prompts finish
- **hard_oom** — OOM at load or mid-generation

Per-GPU layout flags: `--batch N` (prompts in flight per engine),
`--instances-per-gpu N` (engines on a GPU load one at a time and start together),
`--gpu-mode sequential|parallel`, `--quantization`, `--kv-cache-dtype`,
`--enforce-eager true|false`, `--mps true|false`.

In a full suite run, batch, sequence length and engine layout come from the
`workload_3A` block of the GPU profile in `hardware/gpu/`.

## Outputs (`./results/`)

| File | Contents |
|------|----------|
| `<run>_workload11_metadata.json` | config, env, per-GPU summary, `degradation: graceful\|hard_oom`, stats (mean/variance/stddev/p50/p95/p99) |
| `<run>_workload11_per_prompt.csv` | per prompt: status (OK/OOM/ERROR), TTFT, ITL p50/p95/p99, queue time, scheduler delay, engine cumulative preemptions, engine index |

