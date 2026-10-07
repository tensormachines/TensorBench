# TensorBench

TensorBench is an open-source infrastructure benchmark designed to measure the real-world performance, physical behavior, and true unit economics of AI workloads on accelerators. Unlike traditional peak-specification benchmarks that rely on theoretical ceilings, TensorBench stress-tests hardware across the entire stack from the physics of silicon up to active inference engines. TensorBench evaluates raw compute throughput, power boundaries, and effective cost per million tokens simultaneously. By capturing how factors like operational conditions, power limits, and traffic demand impact actual token yield, TensorBench gives data center operators, MLOps engineers, and infrastructure teams the concrete data they need to optimize their existing GPU fleets and make highly accurate, workload-driven hardware economics decisions.
Everything, from installing drivers to the final score, runs with one command: `./run.sh`.

## Requirements

- A server with a supported GPU and platform (see below).
- Ubuntu 22.04 LTS and sudo access.
- A spare drive of at least 500 GB. Setup formats it and mounts it at
  `/tensormachines` for Docker images and results. **All data on it is lost.**
- Internet access, to install packages, build images and download models.
- A [Hugging Face access token](https://huggingface.co/settings/tokens) from an
  account that has accepted the licenses of these models:
  - [meta-llama/Meta-Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
  - [meta-llama/Llama-2-13b-hf](https://huggingface.co/meta-llama/Llama-2-13b-hf)
  - [meta-llama/Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- Optional: AWS credentials, only needed to upload results to S3.

Setup installs everything else: the NVIDIA driver, Docker, the NVIDIA Container
Toolkit, Python and ipmitool.

## Supported hardware

| GPU | Profile |
|---|---|
| NVIDIA V100 16 GB | `v100-16gb` |
| NVIDIA V100 32 GB | `v100-32gb` |
| NVIDIA H100 80 GB SXM5 | `h100-80gb` |
| NVIDIA H200 141 GB | `h200-141gb` |

| Platform | Profile | BMC access |
|---|---|---|
| NVIDIA DGX-1 | `dgx1` | remote IPMI: set `BMC_HOST`, `BMC_USER` and `BMC_PASS` in `.env` |
| NVIDIA DGX-2 | `dgx2` | local IPMI |
| Supermicro SYS-821GE-TNHR (HGX H100/H200) | `hgx2` | local IPMI |
| DigitalOcean GPU Droplet | `droplet` | none |

Setup detects the hardware and picks the matching profiles from
`gpu-stress-suite/hardware/`. On other hardware it writes template profiles there
and stops; see [Adding hardware](#adding-hardware).

## Setup

```bash
git clone https://github.com/tensormachines/TensorBench.git
cd TensorBench
cp .env.template .env     # optional: fill in tokens and credentials in advance
```

That is all. `run.sh` installs everything else the first time it runs.

## How to run

```bash
./run.sh
```

This one command sets up the server and then runs the benchmark. Setup asks for
sudo once and always asks before formatting the drive or rebooting. It also asks
for:

- the Hugging Face token, unless it is in `.env`;
- AWS credentials (press Enter at each prompt if you won't upload results);
- the score inputs, all optional (see [Expected results](#expected-results)).

Answers are saved to `.env`. When setup needs you to act, it prints what to do and
exits. Do it, then run `./run.sh` again; it continues where it stopped:

| Exit code | Do this |
|---|---|
| 10 | Reboot. |
| 11 | Log out and back in, or run `exec newgrp docker`. |
| 12 | Follow the printed instruction (pick the drive, fix the token, ...). |
| 13 | Fill in the hardware profile templates it wrote. |

Completed setup steps are recorded, so later runs go straight to the benchmark.
The benchmark takes about 1.5 hours, plus image builds and model downloads the
first time. It goes through these phases:

| Phase | Workload | Minutes |
|---|---|---|
| Idle baseline | | 2 |
| Warmup, matrix multiplication | 1 | 3 |
| 1A Matrix multiplication | 1 | 4 |
| 1B Memory bandwidth | 5 | 5 |
| 1C LLM inference, Llama 3 8B INT8 (used for the score) | 8 | 7 |
| Cooldown | | 2 |
| 2A LLM inference, Llama 2 13B INT8 | 9 | 15 |
| 2B LLM fine-tuning | 10 | 12 |
| 2C Convolution | 2 | 3 |
| 2D FFT | 4 | 3 |
| 2E Random memory access | 3 | 3 |
| Cooldown | | 2 |
| 3A LLM inference, Llama 2 13B FP16 | 11 | 5 |
| 3B Sustained matrix multiplication | 1 | 20 |
| NCCL collectives across GPUs | 6 | 5 |
| Final idle | | 2 |

Useful options (`./run.sh --help` lists all):

| Option | Effect |
|---|---|
| `--skip 2,4` | Skip workloads by number. |
| `--dry-run` | Check the setup without running workloads; results are mock data. |
| `--upload` | Upload the results to S3 when the run ends. |
| `--only provision`, `--only benchmark` | Run only the setup or only the benchmark. |

To keep a log of the run: `./run.sh 2>&1 | tee -i run.log`.

## Expected results

The run ends by printing the score:

```
  Inputs : Oh=4.6844 Rh=0.4215 Ce=0.0903 USD, Uz=1.0
  WARNING: defaults used for Oh, Rh, Ce, Uz.

============================================================
  SCORE (USD per million tokens): 0.546340768281673
============================================================
```

The score is the estimated cost of generating one million tokens in the 1C LLM
inference phase. Lower is better. It combines the measured throughput and power
with four inputs:

| Input | Meaning | Default |
|---|---|---|
| `SCORE_OH` | ownership or rental cost per GPU per hour, USD | average for the GPU class |
| `SCORE_RH` | reserved power capacity cost per GPU per hour, USD | average for the GPU class |
| `SCORE_CE` | electricity cost per kWh, USD | US industrial average |
| `SCORE_UZ` | utilization, greater than 0 and at most 1 | 1.0 |

Set them during setup or in `.env`. Unset inputs use the defaults, with the
warning shown above. Power for the rest of the node is estimated from the
platform's design power. If phase 1C does not run, there is no score.

Everything from the run is saved in
`/tensormachines/gpu-stress-suite/suite_results/assessment_<timestamp>/`:

| File | Contents |
|---|---|
| `summary.txt` | overview of the run |
| `manifest.json`, `phases.tsv` | timing and status of every phase |
| `results/` | each workload's results |
| `*_nvml_telemetry.csv`, `*_bmc_telemetry.csv` | GPU and BMC telemetry for the whole run |
| `score.json` | the score and every value it was computed from |

A workload that fails is recorded as failed, and the run continues with the next
phase.

## Contributing

Contributions are welcome as GitHub issues and pull requests. The most useful one
is support for new hardware.

### Adding hardware

Each server is described by two profiles in `gpu-stress-suite/hardware/`:

- a **GPU profile** (`gpu/*.json`) says which GPUs it applies to and how large each
  workload runs on them;
- a **platform profile** (`platform/*.json`) says which servers it applies to and
  how to read their BMC.

A profile applies when every condition under its `match` key holds. If several
apply, the one with the most conditions wins.

1. **Run `./run.sh` on the new hardware.** When no profile applies, it writes a
   template to `gpu-stress-suite/hardware/gpu/` or `.../platform/`, with the
   detected values already filled in, and exits with code 13.
2. **Replace every `CHANGEME` in the template:**

   | Key | Profile | What to put there |
   |---|---|---|
   | `workloads.<workload>.<parameter>` | GPU | How large each workload runs on this GPU. Start from the existing profile with the closest memory size and scale from there. Each parameter is passed to its workload as an option. |
   | `match.requires_sensors` | platform | Two or three sensor names that only this server's BMC reports, taken from `platform.sensors` in `/opt/tensormachines/hw_fingerprint.json`. |
   | `bmc.logger` | platform | `bmc_logger.dgx1.py` for a BMC reached over the network (credentials in `.env`), `bmc_logger.dgx2.py` for local IPMI. |

3. **Fill in the score data:**
   - In the platform profile, add `design_power_w`: the vendor's maximum system
     power in watts. The score uses it to estimate the power drawn by the rest of
     the server. Leave it out for virtual machines, whose host is shared.
   - For a new GPU class, add its average Oh and Rh prices to
     `gpu-stress-suite/harness/gpu_pricing.csv`. Without them, the score needs
     `SCORE_OH` and `SCORE_RH` to be set.
4. **Run `./run.sh` again.** Finished profiles are copied to the server
   automatically. Check `phases.tsv` in the results: a workload that failed,
   for example by running out of GPU memory, needs smaller sizes.
5. **Open a pull request** with the new profiles and the `summary.txt` and
   `score.json` from a complete run.