# Workload 2B — Llama-3 fine-tuning (HBM write-path stress)

Self-contained. `scp` this directory to the box and run `./run.sh`. This is
the **training** workload — the only one in the recipe that hammers the HBM
write path through gradient computation, optimizer state updates, and
activation-checkpoint recompute.

## One command

```bash
HF_TOKEN=hf_xxx ./run.sh                                # Llama-3.2-3B-Instruct, 200 steps
HF_TOKEN=hf_xxx ./run.sh --base-model meta-llama/Meta-Llama-3-8B-Instruct
HF_TOKEN=hf_xxx ./run.sh --gpus 0,1
./run.sh --smoke --gpus 0                                # shakeout: tiny ungated, no HF_TOKEN
```

`HF_TOKEN` must have the Meta license accepted for the base model used:
- `meta-llama/Llama-3.2-3B-Instruct` (the default)
- `meta-llama/Meta-Llama-3-8B-Instruct`

Token is passed by environment only, never written into the image.

## What it does

**LoRA fine-tuning** on a frozen FP16 Llama-3 base by default. In a full suite
run, the base model and shape come from the `workload_2B` block of the GPU
profile in `hardware/gpu/`:

| GPU profile | Base model | Training | Batch | Seq | grad_accum | Steps |
|---|---|---|---|---|---|---|
| `v100-16gb` | Llama-3.2-3B-Instruct | LoRA | 16 | 256 | 8 | 200 |
| `v100-32gb`, `h100-80gb` | Llama-3-8B-Instruct | LoRA | 16 | 256 | 4 | 60 |
| `h200-141gb` | Llama-3-8B-Instruct | full | 2 | 512 | 1 | 1000 |

Run on its own without options, it uses the `v100-16gb` values.

LoRA adapters target the attention projections (`q_proj/k_proj/v_proj/o_proj`),
rank 16, alpha 32. **Trainable params are cast to fp32** for AdamW numerical
stability (base stays fp16, frozen).

`--training full` (profile key `training`) trains every parameter in fp32
with fused AdamW at lr 1e-5 instead of LoRA adapters. It needs ~16 bytes per
parameter, so only the H200 profile uses it; `lora` is the default.

Each visible GPU runs an **independent** training process (replicated, not
DDP) so each device is graded on its own — same posture as 1C/2A/3A. This
preserves the per-GPU health signal.

Dataset: `tatsu-lab/alpaca` (HuggingFace), fixed slice `[:4800]` cycled
deterministically across the consumed samples. The slice's SHA-256 is
recorded in metadata for reproducibility.

## Outputs (`./results/`)

| File | Contents |
|------|----------|
| `<run>_workload10_metadata.json` | run config, env, per-GPU + aggregate stats, dataset slice SHA |
| `<run>_workload10_per_step.csv` | one row per step: gpu_index, step, step_time_s, loss, grad_norm |

