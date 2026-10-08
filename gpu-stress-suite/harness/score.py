#!/usr/bin/env python3
"""
score.py - benchmark cost per million tokens for the GPU stress suite.

Reads an assessment results folder, prints the GPU class score at full
utilization (median per-GPU cost per million tokens, Uz = 1) on stdout and writes
it to RESULTS_DIR/score.json with every value it is computed from: inputs, source
files, workload 1C parameters, idle phases and the per-GPU formula components.
score_at_uz.py rescores that score.json at another utilization.

  Cost/Mtok = ((Eh - Ei) * Ce / Th + (Ce * Ei + Oh + Rh) / (Uz * Th)) * 10^6

  Th  tokens per hour over the 1C scored window
  Eh  median rolling-window GPU power under load plus the node overhead, kWh per hour
  Ei  median GPU power over the pre- and post-run idle phases plus the node
      overhead, kWh per hour
  Uz  utilization, a fraction greater than 0 and at most 1

Usage:
  score.py RESULTS_DIR --ce PRICE [--oh PRICE] [--rh PRICE] [--np WATTS] [--window SECONDS]

  --oh  ownership cost per GPU per hour (default from gpu_pricing.csv)
  --rh  reserved capacity cost per GPU per hour (default from gpu_pricing.csv)
  --ce  electricity cost per kWh
  --np  node design power in watts; without it the node overhead is 0
"""

import argparse
import bisect
import csv
import json
import re
import statistics
import sys
from datetime import datetime, timezone
from pathlib import Path

NVML_TS_FORMAT = "%Y-%m-%d %H:%M:%S.%f"
THROUGHPUT_TS_FORMAT = "%Y-%m-%d %H:%M:%S"
PHASE_TS_FORMAT = "%y-%m-%d_%H-%M-%S"

NVML_COLUMNS = ("energy_consumption", "power_gpu_reported", "power_limit", "utilization_mem")

# Default Oh (75th percentile) and Rh (average) per GPU class, USD per GPU per hour.
PRICING_FILE = Path(__file__).with_name("gpu_pricing.csv")
# Pricing row for GPUs that match no class: the average of the known classes.
UNKNOWN_CLASS = "unknown"

# Utilization the score is calculated at.
UZ = 1.0

# Share of the node design power assumed to be drawn at peak.
NODE_PEAK_FRACTION = 0.8
# Share of the non-GPU node power assumed to be drawn regardless of load.
NODE_STATIC_FRACTION = 0.5

def _epoch(text, fmt):
    """Parse a UTC timestamp string to epoch seconds."""
    return datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp()

def _iso(epoch):
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="milliseconds")

def _find_one(directory, pattern):
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        sys.exit(f"ERROR: expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]

class Integral:
    """Running integral of a sampled quantity, interpolated linearly in time.

    Window means are taken as the change in the integral over the window, which
    gives a time-weighted mean. NVML samples are unevenly spaced, so a plain mean
    of the samples in a window would over-weight closely spaced samples and ignore
    the time between the window edges and the nearest samples.
    """

    def __init__(self, times, totals):
        if len(times) < 2:
            sys.exit("ERROR: NVML trace has fewer than two samples")
        self.times = times
        self.totals = totals

    @classmethod
    def of_samples(cls, times, values):
        """Trapezoid integral of point samples.

        Each gap between samples contributes the mean of its two end values times
        its length, which is the standard way to integrate unevenly spaced samples.
        """
        totals = [0.0]
        for i in range(1, len(times)):
            totals.append(totals[-1] + (values[i - 1] + values[i]) / 2 * (times[i] - times[i - 1]))
        return cls(times, totals)

    def at(self, t):
        """Integral at time t, interpolated between the two nearest samples.

        Window edges rarely fall on a sample, so the integral at the edge is
        estimated rather than taken from the nearest sample.
        """
        if not self.times[0] <= t <= self.times[-1]:
            sys.exit(f"ERROR: {_iso(t)} is outside the NVML trace "
                     f"({_iso(self.times[0])} - {_iso(self.times[-1])})")
        i = bisect.bisect_left(self.times, t)
        if self.times[i] == t:
            return self.totals[i]
        t0, t1 = self.times[i - 1], self.times[i]
        v0, v1 = self.totals[i - 1], self.totals[i]
        return v0 + (v1 - v0) * (t - t0) / (t1 - t0)

    def mean(self, start, end):
        return (self.at(end) - self.at(start)) / (end - start)

def load_nvml(path):
    """Return {gpu: {column: (times, values)}}, skipping empty cells."""
    series = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            gpu = series.setdefault(int(row["gpu_id"]), {c: ([], []) for c in NVML_COLUMNS})
            t = _epoch(row["timestamp"], NVML_TS_FORMAT)
            for column in NVML_COLUMNS:
                if row[column]:
                    gpu[column][0].append(t)
                    gpu[column][1].append(float(row[column]))
    return series

def load_throughput(path):
    """Return {gpu: (t0, [tokens per 1 s bucket, ordered by elapsed_s])}."""
    rows = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            rows.setdefault(int(row["gpu_index"]), []).append(
                (int(row["elapsed_s"]), _epoch(row["timestamp"], THROUGHPUT_TS_FORMAT),
                 int(row["tokens_per_sec"])))
    out = {}
    for gpu, gpu_rows in rows.items():
        gpu_rows.sort()
        elapsed, ts, _ = gpu_rows[0]
        # Timestamps are t0 + elapsed_s truncated to the second; take the midpoint.
        t0 = ts - elapsed + 0.5
        out[gpu] = (t0, [tokens for _, _, tokens in gpu_rows])
    return out

def idle_phases(manifest):
    """Completed idle phases before the first or after the last workload phase."""
    phases = manifest["phases"]
    workloads = [i for i, p in enumerate(phases) if p["phase_kind"] == "workload"]
    if not workloads:
        sys.exit("ERROR: manifest has no workload phases")
    return [{"label": p["phase_label"],
             "start": _epoch(p["start_time"], PHASE_TS_FORMAT),
             "end": _epoch(p["end_time"], PHASE_TS_FORMAT)}
            for i, p in enumerate(phases)
            if p["phase_kind"] == "idle" and p["status"] == "completed"
            and (i < workloads[0] or i > workloads[-1])]

def idle_power_w(times, watts, phases):
    """Median and count of the power samples inside the idle phases."""
    samples = [w for t, w in zip(times, watts)
               if any(p["start"] <= t <= p["end"] for p in phases)]
    if not samples:
        sys.exit("ERROR: no NVML power samples in the pre- or post-run idle phases")
    return statistics.median(samples), len(samples)

def utilization(pc_w, pmax_w, pi_w, um_pct):
    """Return (Uz, source) for one window."""
    # Selection compares Pc / Pmax, while the power branch subtracts idle power,
    # so the selected value can be lower than the one not selected.
    if um_pct / 100 > pc_w / pmax_w:
        return um_pct / 100, "memory"
    return (pc_w - pi_w) / (pmax_w - pi_w), "power"

def rolling_windows(start, scored_s, window_s, pi_w, energy, power_limit, util_mem):
    """Telemetry means over windows of window_s seconds, stepped by 1 s."""
    windows = []
    for i in range(scored_s - window_s + 1):
        w_start = start + i
        w_end = w_start + window_s
        pc_w = energy.mean(w_start, w_end)
        pmax_w = power_limit.mean(w_start, w_end)
        # utilization_mem is the percent of time device memory was being read or
        # written, not the share of peak bandwidth used, so light but steady
        # traffic reads as high utilization.
        um_pct = util_mem.mean(w_start, w_end)
        # uz, source = utilization(pc_w, pmax_w, pi_w, um_pct)
        windows.append({
            "start": _iso(w_start),
            "end": _iso(w_end),
            "pc_w": pc_w,
            "pmax_w": pmax_w,
            "um_pct": um_pct,
            # "uz": uz,
            # "uz_source": source,
        })
    return windows

def node_rest_peak_w(design_w, power_limits_w):
    """Estimated non-GPU node power per GPU at peak, in watts.

    Node power is not measured, so the rest of the node (CPUs, memory, fans and
    so on) is taken as the node's peak draw minus the GPU power limits, split
    equally between the node's GPUs.
    """
    total_w = NODE_PEAK_FRACTION * design_w - sum(power_limits_w)
    if total_w < 0:
        sys.exit(f"ERROR: {NODE_PEAK_FRACTION:.0%} of the node design power ({design_w} W) "
                 f"is below the GPU power limits ({sum(power_limits_w)} W)")
    return total_w / len(power_limits_w)

def node_overhead_w(rest_peak_w, gpu_w, limit_w):
    """Non-GPU node power per GPU: a static part plus a part scaled by GPU load."""
    return rest_peak_w * (NODE_STATIC_FRACTION + (1 - NODE_STATIC_FRACTION) * gpu_w / limit_w)

def default_prices(gpu_name):
    """Return (gpu_class, oh, rh) from the pricing file for a GPU name.

    A class matches when its name appears in the GPU name as whole words, so H200
    does not match GH200. The longest matching class wins; with no match, the
    unknown row is used. Missing prices are None.
    """
    with open(PRICING_FILE, newline="") as f:
        rows = list(csv.DictReader(f))
    matches = [r for r in rows if r["gpu_class"] != UNKNOWN_CLASS
               and re.search(rf"\b{re.escape(r['gpu_class'])}\b", gpu_name, re.IGNORECASE)]
    if not matches:
        matches = [r for r in rows if r["gpu_class"] == UNKNOWN_CLASS]
    if not matches:
        return None, None, None
    row = max(matches, key=lambda r: len(r["gpu_class"]))
    price = lambda col: float(row[col]) if row[col] else None
    return row["gpu_class"], price("oh_usd_per_gpu_hour"), price("rh_usd_per_gpu_hour")

def cost_terms(eh, ei, th, uz, ce, oh, rh):
    """Return the (energy, fixed) cost terms in USD per million tokens."""
    return (eh - ei) * ce / th * 1e6, (ce * ei + oh + rh) / (uz * th) * 1e6

def score_gpu(gpu, t0, buckets, series, discard_s, window_s, phases, rest_peak_w, args):
    scored_s = len(buckets) - discard_s
    if scored_s < window_s:
        sys.exit(f"ERROR: GPU {gpu} scored window is shorter than {window_s} s")
    start = t0 + discard_s

    scored_tokens = sum(buckets[discard_s:])
    th = scored_tokens / scored_s * 3600
    pi_w, idle_samples = idle_power_w(*series["power_gpu_reported"], phases)
    # The energy counter is already an integral kept by the driver, so it also
    # covers power changes between samples that integrating power readings misses.
    windows = rolling_windows(start, scored_s, window_s, pi_w,
                              Integral(*series["energy_consumption"]),
                              Integral.of_samples(*series["power_limit"]),
                              Integral.of_samples(*series["utilization_mem"]))
    pc_w = statistics.median(w["pc_w"] for w in windows)
    limit_w = statistics.median(series["power_limit"][1])
    eh_gpu = pc_w / 1000
    ei_gpu = pi_w / 1000
    eh = (pc_w + node_overhead_w(rest_peak_w, pc_w, limit_w)) / 1000
    ei = (pi_w + node_overhead_w(rest_peak_w, pi_w, limit_w)) / 1000
    # uz = statistics.median(w["uz"] for w in windows)
    if th <= 0:
        sys.exit(f"ERROR: GPU {gpu} has non-positive Th ({th})")
    energy, fixed = cost_terms(eh, ei, th, UZ, args.ce, args.oh, args.rh)

    return {
        "gpu_index": gpu,
        "workload_start": _iso(t0),
        "scored_window": {"start": _iso(start), "end": _iso(start + scored_s),
                          "seconds": scored_s},
        "scored_tokens": scored_tokens,
        "th_tokens_per_h": th,
        "eh_gpu_kwh_per_h": eh_gpu,
        "ei_gpu_kwh_per_h": ei_gpu,
        "eh_kwh_per_h": eh,
        "ei_kwh_per_h": ei,
        "pi_w": pi_w,
        "idle_samples": idle_samples,
        "power_limit_w": limit_w,
        "energy_cost_per_mtok": energy,
        "fixed_cost_per_mtok": fixed,
        "cost_per_mtok": energy + fixed,
    }

def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("results_dir", type=Path)
    parser.add_argument("--oh", type=float,
                        help="ownership cost per GPU per hour (default from gpu_pricing.csv)")
    parser.add_argument("--rh", type=float,
                        help="reserved capacity cost per GPU per hour (default from gpu_pricing.csv)")
    parser.add_argument("--ce", type=float, required=True, help="electricity cost per kWh")
    parser.add_argument("--np", type=float,
                        help="node design power in watts; without it the node overhead is 0")
    parser.add_argument("--window", type=int, default=10, help="rolling window in seconds (default 10)")
    args = parser.parse_args()

    results_dir = args.results_dir
    sources = {
        "manifest": _find_one(results_dir, "manifest.json"),
        "workload_1c_metadata": _find_one(results_dir / "results", "*_workload8_metadata.json"),
        "workload_1c_throughput": _find_one(results_dir / "results", "*_workload8_throughput.csv"),
        "nvml_telemetry": _find_one(results_dir, "*_nvml_telemetry.csv"),
    }
    manifest = json.loads(sources["manifest"].read_text())
    metadata = json.loads(sources["workload_1c_metadata"].read_text())
    throughput = load_throughput(sources["workload_1c_throughput"])
    nvml = load_nvml(sources["nvml_telemetry"])
    discard_s = int(metadata["spec"]["discarded_ramp_s"])
    phases = idle_phases(manifest)

    gpu_classes = {g["gpu_name"] for g in metadata["per_gpu_stats"]}
    if len(gpu_classes) != 1:
        sys.exit(f"ERROR: expected one GPU class, found {sorted(gpu_classes)}")
    gpu_name = next(iter(gpu_classes))

    # Fill unset Oh and Rh with the GPU class defaults.
    pricing_class, applied = None, []
    if args.oh is None or args.rh is None:
        pricing_class, oh, rh = default_prices(gpu_name)
        for name, value in (("oh", oh), ("rh", rh)):
            if getattr(args, name) is not None:
                continue
            if value is None:
                sys.exit(f"ERROR: no default {name.capitalize()} for {gpu_name} in "
                         f"{PRICING_FILE.name}; pass --{name}")
            setattr(args, name, value)
            applied.append(name)

    # Power limits of every GPU in the node, including any not used by 1C.
    power_limits_w = [statistics.median(nvml[g]["power_limit"][1]) for g in sorted(nvml)]
    rest_peak_w = 0.0
    if args.np is not None:
        rest_peak_w = node_rest_peak_w(args.np, power_limits_w)

    gpus = []
    for gpu in sorted(throughput):
        if gpu not in nvml:
            sys.exit(f"ERROR: GPU {gpu} has throughput results but no NVML samples")
        t0, buckets = throughput[gpu]
        gpus.append(score_gpu(gpu, t0, buckets, nvml[gpu], discard_s, args.window, phases,
                              rest_peak_w, args))
    score = statistics.median(g["cost_per_mtok"] for g in gpus)

    with open(results_dir / "score.json", "w") as f:
        json.dump({
            "results_dir": str(results_dir),
            "gpu_class": gpu_name,
            "cost_per_mtok": score,
            "aggregation": "median of per-GPU cost_per_mtok",
            "formula": "((Eh - Ei) * Ce / Th + (Ce * Ei + Oh + Rh) / (Uz * Th)) * 10^6",
            "uz": UZ,
            "inputs": {
                "oh": args.oh,
                "rh": args.rh,
                "ce": args.ce,
                "node_design_power_w": args.np,
                "window_s": args.window,
            },
            "pricing_defaults": {
                "file": PRICING_FILE.name,
                "gpu_class": pricing_class,
                "applied": applied,
            } if applied else None,
            "node_estimate": {
                "peak_fraction": NODE_PEAK_FRACTION,
                "static_fraction": NODE_STATIC_FRACTION,
                "gpu_count": len(power_limits_w),
                "gpu_power_limit_w": sum(power_limits_w),
                "rest_peak_w_per_gpu": rest_peak_w,
            },
            "sources": {k: str(p.relative_to(results_dir)) for k, p in sources.items()},
            "workload_1c": {k: metadata.get(k) for k in
                            ("workload", "workload_num", "run_id", "node_id", "smoke",
                             "corpus_sha256", "spec", "env")},
            "idle_phases": [{"label": p["label"], "start": _iso(p["start"]), "end": _iso(p["end"])}
                            for p in phases],
            "gpus": gpus,
        }, f, indent=2)
        f.write("\n")
    return score

if __name__ == "__main__":
    print(main())
