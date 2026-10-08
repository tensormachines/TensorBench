# GPU Stress Suite

Portable, Docker-based GPU stress framework for prototype validation and burn-in
assessment. Each phase runs in an isolated container while host-side telemetry
(NVML + BMC) is captured for the whole run.

## Assessment Flow

`run_all.sh` runs a fixed 16-phase assessment protocol. Idle/cooldown phases are
observation-only; workload phases run a container and emit a JSON result.

| # | Phase | Workload image | Notes |
|---|-------|----------------|-------|
| 0 | Idle Baseline | - | idle observation |
| 1 | Warmup Tensor Compute | workload1 | warmup |
| 2 | 1A GEMM Compute Baseline | workload1 | `--sequential` |
| 3 | 1B Memory Bandwidth Baseline | workload5 | `--buffer-gb 4.0` |
| 4 | 1C LLM Inference | workload8 | LLaMA-3 8B INT8 |
| 5 | Cooldown 1 | - | idle observation |
| 6 | 2A Inference Workload | workload9 | LLaMA-2 13B INT8 |
| 7 | 2B Training Workload | workload10 | Llama-3 fine-tuning (LoRA or full) |
| 8 | 2C Convolution Compute | workload2 | |
| 9 | 2D FFT Compute | workload4 | |
| 10 | 2E Random Access Memory | workload3 | |
| 11 | Cooldown 2 | - | idle observation |
| 12 | 3A Inference Workload | workload11 | LLaMA-2 13B FP16 |
| 13 | 3B Sustained Power Proxy | workload1 | long GEMM burn |
| 14 | NVLink NCCL Collective Stress | workload6 | `--tensor-mb 512` |
| 15 | Final Idle Snapshot | - | idle observation |

Per-phase runtimes are set by the built-in profile in `run_all.sh` (`PHASE_SEC`),
currently: 1A=240s, 1B=300s, 1C=420s, 2A=900s, 2B=720s, 2C=180s, 2D=180s, 2E=180s,
3A=300s, 3B=1200s, NCCL=300s. Idle/cooldown default to 120s, warmup to 180s.

## Repeated Runs

`loop_run.sh` wraps `run_all.sh` to run the full assessment multiple times back-to-back
(soak/burn-in), with a configurable break between runs. Each run is logged to its own
timestamped `*_run_all_loop<N>.log`, and any extra args are passed straight through to
`run_all.sh`.

```bash
cd /tensormachines/gpu-stress-suite    # all commands below run from here

./loop_run.sh                                         # 1 run, 5-min break (defaults)
./loop_run.sh --runs 5 --break 600 --max-vram-gb 60   # 5 runs, 10-min break, passthrough args
./loop_run.sh --runs 0                                # run forever

# 30-run soak with per-run S3 upload, outer log via tee -i
./loop_run.sh --upload --runs 30 2>&1 | tee -i 26-07-15_17-38_loop_run.log.  # date is <UTC yy-mm-dd_hh-mm>
```

A non-zero exit from `run_all.sh` is logged as a warning and the loop continues, so one
bad run won't abort a long soak.

## Quick Start

Linux shell (Bash 4+). Run from the deployed suite directory on the node,
`/tensormachines/gpu-stress-suite/` (not from the repo checkout):

```bash
cd /tensormachines/gpu-stress-suite
./run_all.sh
```

Common options:

```bash
./run_all.sh --max-vram-gb 8
./run_all.sh --idle-duration 5 --cooldown-duration 5
./run_all.sh --require-multi-gpu
./run_all.sh --skip 6,7          # skip workloads by number (comma-separated)
./run_all.sh --upload            # upload results to S3 after collection (off by default)
./run_all.sh --score             # compute the score (off by default)
./run_all.sh --score --oh 4.00 --rh 0.42 --ce 0.10 --uz 0.5   # score inputs
```

The VRAM cap comes from the GPU profile (its `reserve_fraction` of GPU memory, 80% by
default) unless `--max-vram-gb` is given.

With `--score`, the score (USD per million tokens) is printed at the end of the run and
saved to `score.json` in the assessment dir. Without it, scoring is skipped. It needs three prices in USD, ownership cost per
GPU-hour (`--oh`), reserved capacity cost per GPU-hour (`--rh`) and electricity cost per
kWh (`--ce`). Each option falls back to `SCORE_OH`, `SCORE_RH` or `SCORE_CE` from the
environment or `.env`, then to a default. Oh and Rh default to the prices for the GPU
class in `harness/gpu_pricing.csv` (Oh: 75th percentile rental price, Rh: average);
a GPU with no row there uses the `unknown` row, the average of the known classes,
with a warning. Ce defaults to a built-in value.

The score is printed first at 100% utilization. The run then scores again with the
expected utilization from `--uz` or `SCORE_UZ`, or asks for it at the terminal when
neither is set; an empty answer or no terminal skips the second score. The second score
is calculated from `score.json` (`harness/score_at_uz.py RESULTS_DIR UZ`) and recorded
there as `custom_uz_score`; `cost_per_mtok` stays the 100% score. A warning is printed when Oh
is the default. Scoring is skipped on a dry run.

Node power is not measured yet, so the score adds an estimated non-GPU overhead per GPU.
At peak it is 80% of the platform profile's `design_power_w` minus the GPU power limits,
split equally between the node's GPUs. Half of it is drawn regardless of load and half
scales with GPU power as a fraction of the GPU power limit. Without `design_power_w` the
overhead is 0.

To capture a run log, pipe through `tee -i` (`-i` lets tee survive Ctrl-C so
cleanup output still reaches the log):

```bash
./run_all.sh --upload 2>&1 | tee -i 26-07-15_17-24_run_all.log   # date is <UTC yy-mm-dd_hh-mm>

# SMOKE test
./run_all.sh --upload --skip 1,3,4,5,6,7,8,9,10,11,12,13,14,15 --idle-duration 5 --cooldown-duration 5  2>&1 | tee -i  26-07-15_17-24_run_all.log
```

### Dry run

`--dry-run` goes through every phase of the assessment without running any
workload, to check the setup end to end:

- the hardware check, profile resolution and telemetry loggers run as usual;
- no workload container is started; each workload phase writes a mock result;
- idle and cooldown phases last `DRY_RUN_IDLE_SEC` seconds (default 1);
- results go to `suite_results/assessment_dryrun_<timestamp>/`, and `--upload`
  only simulates the S3 upload (nothing is transferred).

Level 1 (`--dry-run` or `--dry-run 1`) also skips the docker image builds;
level 2 (`--dry-run 2`) builds them, to check that the images build.

```bash
./run_all.sh --dry-run        # fastest check
./run_all.sh --dry-run 2      # also build the workload images
```

## Telemetry

Telemetry is captured by two host-side loggers started/stopped automatically by
`run_all.sh` for the duration of the assessment:

- **NVML logger** (`gpu_logger.py`) — per-GPU driver-level metrics
- **BMC logger** (`bmc_logger.dgx1.py` / `bmc_logger.dgx2.py`) — board/chassis
  sensors via IPMI; the variant is chosen by the matched platform profile

Both live at `/opt/tensormachines/loggers/` and run under the `telemetry_env` interpreter.
They write to their own results directory (separate from the assessment dir below).

Logger lifecycle is orphan-safe:

- `run_all.sh` traps `EXIT`/`INT`/`TERM`/`HUP`, so both loggers are stopped even on
  Ctrl-C or a killed tmux session.
- On startup, any stale logger instance from a previous run is detected by process
  name (`pgrep`) and killed before a fresh one starts.
- Verify anytime with `pgrep -af 'bmc_logger|gpu_logger'` (empty = clean).

NVML fields include GPU temperature, power draw, SM clock, memory clock, GPU/memory
utilization, memory used/total, performance state, and ECC counters where supported.
BMC fields cover board temperatures, per-rail power, voltages, and fan speeds.

## Assessment Outputs

Each run creates a timestamped directory under `suite_results/`:

```text
suite_results/
`-- assessment_<timestamp>/
    |-- manifest.json                # phases, timings, node_id, VRAM cap, phase profile
    |-- phases.tsv                   # phase-by-phase log
    |-- summary.txt                  # human-readable summary
    |-- summary_<timestamp>.txt      # short index of the run
    |-- *_nvml_telemetry.csv         # NVML telemetry for the whole run
    |-- *_bmc_telemetry.csv          # BMC telemetry for the whole run
    |-- *_metadata.txt               # logger metadata (GPU and BMC details)
    |-- *_run_all*.log               # run log, when captured as <date>_run_all.log
    |-- score.json                   # the score and every value it was computed from
    `-- results/                     # workload-level results (JSON and CSV, per phase)
```

`manifest.json` records `node_id`, `run_id`, the phase-duration profile, the VRAM cap,
and every phase's status/timing. Failed workload phases still emit a JSON result with
`"status": "failed"` and the assessment continues.

At the end of a run, artifacts (top-level summary, logger telemetry, run log) are
gathered into the assessment dir by `collect_artifacts`.

## S3 Upload

Uploading is **disabled by default** so experimental runs stay local. Enable per run
with `--upload` (or `UPLOAD_RESULTS=1`); after artifact collection,
`scripts/upload_results.sh` packs the assessment dir into one `.tar.gz` and uploads it
with an HTTP PUT to the public write-only endpoint.
`UPLOAD_URL` overrides the endpoint. `<platform>` and `<node_id>` come from the run's 
`manifest.json`, where `<platform>` is the matched platform profile; `--platform` overrides
it. The random suffix keeps uploads from overwriting each other.

`run_all.sh --contact EMAIL` (only with `--upload`) writes the email address to
`contact.txt` in the assessment dir before it is packed; `upload_results.sh` takes it as
a second argument after the folder. The address must be a valid email address.

Endpoint limits: 200 MB per archive, 50 requests per 5 minutes per IP (then HTTP 403
until the window clears). Uploads are deleted after 30 days. A failed upload logs a warning
and never aborts the run — results remain local for manual retry:

```bash
./scripts/upload_results.sh suite_results/assessment_<timestamp>/
./scripts/upload_results.sh suite_results/assessment_<timestamp>/ --platform dgx2
./scripts/upload_results.sh suite_results/assessment_<timestamp>/ you@example.com
```

## Workload Overview

Synthetic micro-workloads (workloads 1–6) use generated inputs. Model workloads (8–11)
run quantized/half-precision open LLaMA models and are the core of the current protocol.
Workload 7 (tiny public-model training) is still built and callable directly, but is not
part of the default 16-phase flow.

| # | Image | Category | Primary metric |
|-------|----------|----------|----------------|
| workload1 | Tensor (GEMM) Compute | Compute-only | TFLOP/s |
| workload2 | Convolution Compute | Compute-only | TFLOP/s, samples/sec |
| workload3 | Random Access Memory | Memory-only | GB/s |
| workload4 | FFT Compute | Compute-only | TFLOP/s, FFT/s |
| workload5 | Memory Bandwidth | Memory-only | GB/s |
| workload6 | Multi-GPU NCCL Collective | Interconnect | GB/s, latency |
| workload7 | Public-Model Training | Model training | tokens/sec, step time, loss |
| workload8 | LLaMA-3 8B INT8 (1C) | Inference | latency, throughput |
| workload9 | LLaMA-2 13B INT8 (2A) | Inference | latency, throughput |
| workload10 | Llama-3 fine-tuning (2B) | Training | tokens/sec, step time |
| workload11 | LLaMA-2 13B FP16 (3A) | Inference (near-limit) | latency, throughput |

### NCCL (workload6)

Stresses NCCL collectives (all-reduce, all-gather, broadcast). On a single-GPU host it
writes a skipped artifact (`requires >=2 GPUs`). Use `--require-multi-gpu` to enforce
multi-GPU on target systems.

### Model workloads and Hugging Face access

Gated/large models require a valid `HF_TOKEN` in the environment; `run_all.sh` passes it
into the containers. Workload 7's image only installs HF dependencies when built with
`INSTALL_HF_DEPS=1`:

```bash
INSTALL_HF_DEPS=1 ./run_all.sh
```

Model workloads are VRAM-adaptive: they clamp oversized VRAM requests to a safe device
budget and downshift batch/sequence sizes on CUDA OOM.

## Adaptive GPU Sizing

- The VRAM cap is the GPU profile's `reserve_fraction` of GPU memory, 80% by default.
- Workloads clamp overly large requested VRAM budgets to a safe device budget.
- Convolution, random-access, FFT, and memory-bandwidth workloads downshift as needed.

## Hardware Profiles

Workload parameters and logger selection come from `hardware/`, resolved by
`harness/hwdetect.py`, matched against the hardware fingerprint saved during provisioning stages.

```bash
./run_all.sh                                  # run with loaded profile parameters
./run_all.sh --max-vram-gb 8                  # override vram the cap
./run_all.sh --gpu-profile v100-16gb          # force a GPU profile
./run_all.sh --platform-profile dgx1          # force a platform profile
```

## Prerequisites

| Requirement | Notes |
|-------------|-------|
| Docker Engine + NVIDIA Container Toolkit | required for GPU containers |
| CUDA-compatible NVIDIA GPU(s) | multi-GPU needed for NCCL runtime validation |
| Bash 4+ | Linux shell |
| `python3`, `docker`, `nvidia-smi` on host | checked at startup |
| Host loggers at `/opt/tensormachines/loggers/` | `gpu_logger.py`, `bmc_logger.dgx{1,2}.py`, `telemetry_env` |

Verify Docker GPU access:

```bash
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L
```

## Project Structure

```text
gpu-stress-suite/
|-- run_all.sh
|-- loop_run.sh
|-- README.md
|-- SOURCES.md
|-- harness/                         # hwdetect.py (hardware profiles), score.py, gpu_pricing.csv
|-- hardware/                        # GPU and platform profiles
|-- scripts/                         # push_weights.sh, upload_results.sh (S3 transfer)
|-- workload1_tensor_compute/
|-- workload2_conv_compute/
|-- workload3_random_access_memory/
|-- workload4_fft_compute/
|-- workload5_memory_bandwidth/
|-- workload6_multigpu_nccl/
|-- workload7_public_model_training/
|-- workload_1C/                     # LLaMA-3 8B INT8 integrated baseline  (image 8)
|-- workload_2A/                     # LLaMA-2 13B INT8 diagnostic          (image 9)
|-- workload_2B/                     # Llama-3 fine-tuning write-path       (image 10)
`-- workload_3A/                     # LLaMA-2 13B FP16 near-limit boundary (image 11)
```

Image number → directory mapping is defined in `build_all_images()` in `run_all.sh`.

## Individual Workloads

Each `run.sh` can be called directly, e.g.:

```bash
./workload1_tensor_compute/run.sh
./workload5_memory_bandwidth/run.sh --buffer-gb 1.0 --max-vram-gb 2
./workload6_multigpu_nccl/run.sh --duration 60 --tensor-mb 512
INSTALL_HF_DEPS=1 ./workload7_public_model_training/run.sh --duration 60
```

## Scope Notes

- `workload5` measures single-GPU memory bandwidth only.
- `workload6` requires multi-GPU hardware for runtime validation.
- Model workloads need HF dependencies, network/model access, and sufficient VRAM.