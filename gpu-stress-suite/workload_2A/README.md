# Workload 2A — LLaMA-2 13B GPTQ-INT8 integrated inference

Self-contained. `scp` this directory to the box and run `./run.sh`.

## One command

```bash
HF_TOKEN=hf_xxx ./run.sh                       # batch 4, seq 1024, 15 min
HF_TOKEN=hf_xxx ./run.sh --batch 16 --seq 2048
HF_TOKEN=hf_xxx ./run.sh --gpus 0,1            # subset of GPUs
./run.sh --smoke --gpus 0                      # shakeout: tiny ungated model, no HF_TOKEN
```

`HF_TOKEN` must have the `meta-llama/Llama-2-13b-hf` license accepted. Token
is passed by environment only, never written into the image.

## What it does

Runs LLaMA-2 13B at **GPTQ-INT8** precision (quantized once, in-container,
from Meta's official gated weights — same artifact pipeline as 1C) with 256
generated tokens per prompt and greedy decoding (recipe spec: batch 16,
sequence length 2048).
KV cache is FP8 (`kv_cache_dtype=fp8_e5m2`) so the spec batch fits a 32 GB
V100. Each visible GPU runs an **independent** vLLM instance (replicated,
not tensor-parallel) so the per-GPU health signal is preserved. Run is 15
minutes; first 2 minutes discarded as thermal ramp; final 13 minutes scored.

Per-GPU layout flags: `--instances-per-gpu N` (batch is per engine; engines on
a GPU load one at a time and start timing together), `--quantization`,
`--kv-cache-dtype`, `--enforce-eager true|false`, `--mps true|false`.

In a full suite run, batch, sequence length and engine layout come from the
`workload_2A` block of the GPU profile in `hardware/gpu/`.

The recipe positions 2A as the *primary diagnostic signal*: a genuinely
larger model than 1C at higher batch and longer sequences, pushing HBM
toward saturation. Degradation that hides in 1C's 8 B headroom should
separate cleanly here.

## Outputs (`./results/`)

| File | Contents |
|------|----------|
| `<run>_workload9_metadata.json` | config, env, per-GPU + aggregate stats |
| `<run>_workload9_throughput.csv` | 1 Hz tokens/sec per GPU (full run, unsliced) |
| `<run>_workload9_per_prompt.csv` | one row per completed prompt (TTFT, ITL p50/p95/p99, engine index) |
