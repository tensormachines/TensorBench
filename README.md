# TensorBench

TensorBench is an open-source infrastructure benchmark designed to measure the
real-world performance, physical behavior, and true unit economics of AI workloads
on accelerators. Unlike traditional peak-specification benchmarks that rely on
theoretical ceilings, TensorBench stress-tests hardware across the entire stack from
the physics of silicon up to active inference engines. TensorBench evaluates raw
compute throughput, power boundaries, and effective cost per million tokens
simultaneously.

By capturing how factors like operational conditions, power limits, and traffic
demand impact actual token yield, TensorBench gives data center operators, MLOps
engineers, and infrastructure teams the concrete data they need to optimize their
existing GPU fleets and make highly accurate, workload-driven hardware economics
decisions.

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
| NVIDIA HGX-2 | `hgx2` | local IPMI |
| DigitalOcean Droplet | `droplet` | none |

Setup detects the hardware and picks the matching profiles from
`gpu-stress-suite/hardware/`. 

### Unsupported hardware
Setup writes template profiles for unsupported hardware at `gpu-stress-suite/hardware/` and stops. In this case, you need to complete the profiles and rerun. See [Adding hardware](#adding-hardware).

## Setup

```bash
git clone https://github.com/tensormachines/TensorBench
cd TensorBench
cp .env.template .env     # optional: fill in tokens and credentials in advance
```
If you don't fill `.env`, you will be prompted for the values during setup.

`run.sh` installs everything else the first time it runs.

## How to run

```bash
./run.sh
```

This one command sets up the server and then runs the benchmark. Setup asks for
sudo once and always asks before formatting the drive or rebooting. It also asks
for:

- The Hugging Face token, unless it is in `.env`;
- The ownership cost per GPU per hour (Oh) for the score
  - Press Enter to use the default for your GPU class (see [Expected results](#expected-results)).

Answers are saved to `.env`.

When setup needs you to act, it prints what to do and
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
| `--upload` | Upload the results to TensorMachines when the run ends. |
| `--only provision`, `--only benchmark` | Run only the setup or only the benchmark. |

To keep a log of the run: `./run.sh 2>&1 | tee -i run.log`.

## Expected results

The run ends by printing the score, first for GPUs that are busy all the time (100% utilization).

Here is an example output (numbers are hypothetical):
```
  Inputs : Oh=5.685 Rh=0.4215 Ce=0.1000 USD

============================================================
  SCORE at 100% utilization (USD per million tokens): 0.5466669121552199
============================================================
  This is the cost if the GPUs are busy 100% of the time.

  Your expected utilization (greater than 0, at most 1), or Enter to skip: 0.6

============================================================
  SCORE at 60% utilization (USD per million tokens): 0.9096377453490845
============================================================

  Score details: /tensormachines/gpu-stress-suite/suite_results/assessment_<timestamp>/score.json
```

The score is the estimated cost of generating one million tokens in the 1C LLM
inference phase. Lower is better. It combines the measured throughput and power
with these inputs:

| Input | Meaning | Default |
|---|---|---|
| `SCORE_OH` | ownership or rental cost per GPU per hour, USD | 75th percentile price for the GPU class |
| `SCORE_RH` | reserved power capacity cost per GPU per hour, USD | average for the GPU class |
| `SCORE_CE` | electricity cost per kWh, USD | US industrial average |
| `SCORE_UZ` | expected utilization, greater than 0 and at most 1 | asked at the end of the run |

Setup asks only for Oh. The others use their defaults unless set in `.env`.
Utilization is the share of time the GPUs do useful work. After the 100% score,
the run asks for your expected utilization and prints a second score for it. If
`SCORE_UZ` is set, the run uses it without asking. Power for the rest of the node
is estimated from the platform's design power. If phase 1C does not run, there is
no score.

`score.json` holds both scores and every value they were computed from, for your
own calculations.

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

## Get a benchmark comparison

Upload your run and we'll send you a comparison of your node against other
systems of the same platform, as data comes in.

```bash
./run.sh --upload --contact you@company.com
```

`--contact` is optional. Without it the upload is anonymous and we can't send you
the comparison. Your email is only used to send the comparison. No account or
credentials needed. If the upload fails, the run still completes and results stay
on your server.

Already ran without `--upload`?

```bash
./gpu-stress-suite/scripts/upload_results.sh \
  /tensormachines/gpu-stress-suite/suite_results/assessment_<timestamp>/ you@company.com
```

The upload contains the whole assessment folder: logs, telemetry, the score and the
server's short hostname. Add `--dryrun` to see what would be sent without sending it.

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
      - For a new GPU class, add its Oh and Rh prices to
        `gpu-stress-suite/harness/gpu_pricing.csv`. Without them, the score uses
        the `unknown` row, the average of the known classes, and warns about it.
4. **Run `./run.sh` again.** Finished profiles are copied to the server
   automatically. Check `phases.tsv` in the results: a workload that failed,
   for example by running out of GPU memory, needs smaller sizes.
5. **Open a pull request** with the new profiles and the `summary.txt` and
   `score.json` from a complete run.

### Adding Models

The benchmark currently uses Llama 8B for its cost-per-token results. If you want to test another model, fork the repository and adapt the model configuration and benchmark code to your needs. The project is intended to serve as a framework for that work; adding a model is not yet a configuration-only step.
We plan to add more models over time. If you’ve adapted the benchmark for another model, contributions are welcome.

## Scoring methodology

The score is calculated for each GPU, and the reported score is the median across
GPUs:

```
Cost per million tokens = ( (Eh − Ei) × Ce / Th  +  (Ce × Ei + Oh + Rh) / (Uz × Th) ) × 10^6
```

| Symbol | Meaning |
|---|---|
| Th | tokens per hour: all tokens generated in the scored part of phase 1C (after its warm-up), divided by its length |
| Eh | power under load, in kWh per hour: the median of 10-second rolling averages of GPU power during the scored part of 1C, plus the node overhead below |
| Ei | idle power, in kWh per hour: the median GPU power in the idle phases at the start and end of the run, plus the node overhead below |
| Oh, Rh, Ce, Uz | the inputs described in [Expected results](#expected-results) |

The first term is the cost of the electricity used above idle. The second is the
cost of everything paid for whether the GPU is busy or not: idle electricity,
ownership and reserved power capacity. Dividing it by Uz spreads those costs over
only the share of time the GPU does useful work.

For further analysis of score calculation refer to `score.py` and `score.json`

### Node power estimate

GPU power is measured, but power for the rest of the server (CPUs, memory,
networking, fans) is estimated from the platform profile's design power:

```
Rest-of-node peak per GPU  R = (0.8 × design power − sum of GPU power limits) / GPU count
Node overhead per GPU         = R × (0.5 + 0.5 × GPU power / GPU power limit)
```

| Symbol | Meaning |
|---|---|
| design power | the server's maximum power from its datasheet (`design_power_w` in the platform profile) |
| 0.8 | share of the design power assumed to be drawn at peak |
| GPU power limit | each GPU's power limit, read from NVML |
| 0.5 | share of the rest-of-node power assumed to be drawn regardless of load; the other half scales with GPU load |

The overhead is added to the GPU's power under load for Eh and to its idle power
for Ei. A platform without a design power, such as a cloud VM, gets no overhead.

## Sources

The default prices come from public data collected in October 2026. They are
there so the score works out of the box; set your own prices for a real decision.
The per-class values are in `gpu-stress-suite/harness/gpu_pricing.csv`.

**Oh (ownership cost per GPU per hour):** the 75th percentile of every published hourly
price for the GPU class (a middle ground for on-demand, spot, reserved and committed pricing configurations) from these providers' rate cards:
[AWS](https://aws.amazon.com/ec2/capacityblocks/pricing/),
[CoreWeave](https://www.coreweave.com/pricing),
[Crusoe](https://www.crusoe.ai/cloud/pricing),
[GMI Cloud](https://www.gmicloud.ai/en/pricing),
[Google Cloud](https://cloud.google.com/products/compute/gpus-pricing),
[Hyperstack](https://www.hyperstack.cloud/gpu-pricing),
[Lambda](https://lambda.ai/pricing),
[Nebius](https://docs.nebius.com/compute/resources/pricing),
[Oracle](https://www.oracle.com/cloud/price-list/),
[Paperspace/DigitalOcean](https://docs.digitalocean.com/products/paperspace/pricing/),
[Runpod](https://www.runpod.io/pricing) and
[Verda](https://verda.com/pricing).

**Rh (reserved power capacity cost per GPU per hour):** the cost of reserving a
reference system's rated power in a colocation data center, per GPU:

```
Rh = rated system power in kW × 1.10 × rent per kW per month / GPUs / 730
```

| Part | Meaning |
|---|---|
| rated system power | the maximum power of the NVIDIA platform, from manufacturer documentation |
| × 1.10 | 10% headroom |
| rent per kW per month | wholesale colocation asking rent from [CBRE North America Data Center Trends H1 2026](https://www.cbre.com/insights/books/north-america-data-center-trends-h1-2026/northern-virginia-data-center-market), excluding electricity |
| GPUs | GPUs in the rack |
| 730 | average hours in a month |

The default Rh is the average of this value over Dallas-Ft. Worth, Northern
Virginia and Silicon Valley, at the midpoint of each market's low and high rent, and the
rack layouts in the data (for example, racks of one, two and four DGX servers, or
NVL72 racks at nominal and peak power). For example, one DGX H200 (10.2 kW, 8 GPUs) in
Dallas-Ft. Worth at $160 per kW per month gives 10.2 × 1.10 × 160 / 8 / 730 =
$0.30 per GPU per hour.

**Ce (electricity cost per kWh):** $0.10, the average of 2026 year-to-date
industrial electricity prices across 17 US data center markets, rounded. Each
market uses its state's price from the U.S. Energy Information Administration
([Electric Power Monthly, Table 5.6.B](https://www.eia.gov/electricity/monthly/epm_table_grapher.php?t=table_5_06_b)).
The markets are the ones in the CBRE report used for Rh; New York Tri-State uses
the average of New York, New Jersey and Connecticut.

**Unknown GPUs:** the average Oh and Rh of the known classes.
