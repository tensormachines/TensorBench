# Workload 1C — LLaMA-3 8B INT8 integrated baseline

Self-contained. `scp` this directory to the box and run `./run.sh`.

## One command

```bash
HF_TOKEN=hf_xxx ./run.sh                    # batch 8, seq 512, 7 min
HF_TOKEN=hf_xxx ./run.sh --batch 16 --seq 1024
```

In a full suite run, batch, sequence length and engine layout come from the
`workload_1C` block of the GPU profile in `hardware/gpu/`.

`HF_TOKEN` must have the `meta-llama/Meta-Llama-3-8B-Instruct` license accepted.
The token is passed by environment only and never written into the image.

## What it does

- vLLM 0.5.4 (last reliable Volta sm_70 wheel), one or more **independent**
  engines per visible GPU (replicated, not tensor-parallel — each GPU graded on
  its own). Engines on a GPU load one at a time and start timing together.
- Per-GPU layout flags: `--instances-per-gpu N` (batch is per engine),
  `--gpu-mode parallel|sequential` (sequential runs the full duration on each
  GPU in turn), `--quantization`, `--enforce-eager true|false`, `--mps true|false`.
- Greedy decoding, `enforce_eager`, fixed seed, token-exact 1024-token inputs
  from `frozen_prompts.json` (SHA-256 recorded in the metadata JSON).
- 7-minute run; first 60 s (thermal ramp) is in the CSV but excluded from the
  JSON summary statistics.

## Outputs (`./results/`)

| File | Contents |
|------|----------|
| `<run>_workload8_metadata.json` | config, env, corpus hash, per-GPU + aggregate stats (mean, variance, stddev, p50/p95/p99) |
| `<run>_workload8_throughput.csv` | 1 Hz tokens/sec per GPU, full unsliced run |
| `<run>_workload8_per_prompt.csv` | per prompt: TTFT, ITL p50/p95/p99, output tokens, engine index |


## Reproducibility caveat

Greedy + fixed seed + eager + token-exact inputs remove the controllable
nondeterminism. vLLM continuous batching can still reorder scheduling between
runs; the residual variation under identical inputs is the hardware signal the
suite exists to surface.
