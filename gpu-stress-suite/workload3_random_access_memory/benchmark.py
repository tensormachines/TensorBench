"""
Workload 3: Random Access Memory Stress
Category  : Memory-only
Source    : PyTorch CUDA indexing kernels
Purpose   : Stress GPU memory with low-arithmetic-intensity random gathers.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta
import torch.multiprocessing as mp
from pathlib import Path

import torch

if 'TZ' in os.environ:
    time.tzset()

# DUTY_CYCLE = .95
TIMEZONE = 0  # UTC for standardization

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


def resolve_effective_vram_cap_gb(device_id: int, args, reserve_fraction: float = 0.8) -> float:
    if args.max_vram_gb <= 0:
        return 0.0
    total_gb = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    return round(min(args.max_vram_gb, total_gb * reserve_fraction), 3)


def resolve_working_set_gb(device_id: int, args) -> float:
    total_gb = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    allowed_gb = resolve_effective_vram_cap_gb(device_id, args) if args.max_vram_gb > 0 else total_gb
    return round(max(0.125, min(args.working_set_gb, allowed_gb * 0.45)), 3)


def build_buffers(device, working_set_gb: float, index_fraction: float):
    source_elements = int(working_set_gb * 1024**3 / 4)
    index_count = max(1024, int(source_elements * index_fraction))
    source = torch.randn(source_elements, device=device, dtype=torch.float32)
    indices = torch.randint(0, source_elements, (index_count,), device=device, dtype=torch.long)
    return source, indices


def benchmark_device(device_id: int, args) -> dict:
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device_id)
    working_set_gb = resolve_working_set_gb(device_id, args)

    if working_set_gb < args.working_set_gb:
        print(
            f"  [GPU {device_id}] Requested working set {args.working_set_gb:.2f} GB exceeds the safe budget. "
            f"Auto-reducing to {working_set_gb:.3f} GB.",
            flush=True,
        )

    while True:
        try:
            source, indices = build_buffers(device, working_set_gb, args.index_fraction)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            next_working_set_gb = round(working_set_gb / 2, 3)
            if next_working_set_gb < 0.125:
                raise RuntimeError("Random-access memory stress could not fit even a 128 MB working set.")
            print(
                f"  [GPU {device_id}] Working set {working_set_gb:.3f} GB OOM - retrying with {next_working_set_gb:.3f} GB.",
                flush=True,
            )
            working_set_gb = next_working_set_gb

    gather_bytes = indices.numel() * source.element_size() * 2 + indices.numel() * indices.element_size()
    print(
        f"  [GPU {device_id}] Warming up ({args.warmup_iters} iters, working set={working_set_gb:.3f} GB, "
        f"indices={indices.numel():,}) ...",
        flush=True,
    )
    for _ in range(args.warmup_iters):
        gathered = torch.index_select(source, 0, indices)
        _ = gathered.sum()
    torch.cuda.synchronize(device)

    print(f"  [GPU {device_id}] Running benchmark for {args.duration}s ...", flush=True)
    iter_times = []
    deadline = time.perf_counter() + args.duration
    loop_start = time.perf_counter() 
    per_second_gbps = []
    _bucket = []
    _current_second = 0
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        gathered = torch.index_select(source, 0, indices)
        _ = gathered.sum()
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        iter_times.append(elapsed)
        _elapsed_s = int(time.perf_counter() - loop_start)
        if _elapsed_s > _current_second:
            if _bucket: 
                per_second_gbps.append(round(sum(_bucket) / len(_bucket), 2))
            _bucket = []
            _current_second = _elapsed_s
        _bucket.append(gather_bytes / elapsed / 1e9)
        
    if _bucket:
          per_second_gbps.append(round(sum(_bucket) / len(_bucket), 2))

    total_iters = len(iter_times)
    gbps = [gather_bytes / t / 1e9 for t in iter_times]
    trim = max(1, total_iters // 10)

    result = {
        "gpu_index": device_id,
        "gpu_name": props.name,
        "working_set_gb": working_set_gb,
        "requested_working_set_gb": args.working_set_gb,
        "index_fraction": args.index_fraction,
        "index_count": int(indices.numel()),
        "requested_vram_cap_gb": args.max_vram_gb if args.max_vram_gb > 0 else "unlimited",
        "vram_cap_gb": resolve_effective_vram_cap_gb(device_id, args) if args.max_vram_gb > 0 else "unlimited",
        "total_iterations": total_iters,
        "peak_GBps": round(max(gbps), 2),
        "sustained_GBps": round(sum(sorted(gbps)[trim:-trim]) / max(1, total_iters - 2 * trim), 2),
        "bytes_per_iter": int(gather_bytes),
        "per_second_gbps": per_second_gbps,
    }

    print(
        f"  [GPU {device_id}] Done - Peak: {result['peak_GBps']:.1f} GB/s | "
        f"Sustained: {result['sustained_GBps']:.1f} GB/s"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Random Access Memory Stress Benchmark")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--working-set-gb", type=float, default=1.5)
    parser.add_argument("--index-fraction", type=float, default=0.25)
    parser.add_argument("--warmup-iters", type=int, default=20)
    parser.add_argument("--output-dir", type=str, default="/workspace/results")
    parser.add_argument("--max-vram-gb", type=float, default=12.0)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("No CUDA devices found. Run with --gpus all.")

    n_gpus = torch.cuda.device_count()
    if args.max_vram_gb > 0:
        for gpu_id in range(n_gpus):
            total_gb = torch.cuda.get_device_properties(gpu_id).total_memory / 1024**3
            effective_cap_gb = resolve_effective_vram_cap_gb(gpu_id, args)
            if effective_cap_gb < args.max_vram_gb - 0.05:
                print(
                    f"  [GPU {gpu_id}] Requested VRAM cap {args.max_vram_gb:.1f} GB exceeds "
                    f"the safe budget for this GPU. Auto-clamping to {effective_cap_gb:.1f} GB.",
                    flush=True,
                )
            torch.cuda.set_per_process_memory_fraction(min(effective_cap_gb / total_gb, 1.0), device=gpu_id)

    print(f"\n{'='*60}")
    print("  Random Access Memory Stress - Workload 3")
    print(f"  Working set : {args.working_set_gb:.2f} GB requested")
    print(f"  Duration    : {args.duration}s per GPU")
    print(f"  VRAM cap    : requested {args.max_vram_gb:.1f} GB per GPU (auto-clamped if needed)")
    print(f"  GPUs        : {n_gpus} detected")
    print(f"{'='*60}\n")

    gpu_info = get_gpu_info()
    run_start = datetime.now(timezone(timedelta(hours=TIMEZONE)))
    run_id = run_start.strftime("%y-%m-%d_%H-%M")
    
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_gpus) as pool:
        results = pool.starmap(benchmark_device, [(gpu_id, args) for gpu_id in range(n_gpus)])
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{run_id}_workload3_gbps.csv")
    with open(csv_path, "w") as f:
        f.write("timestamp,elapsed_s,gpu_index,gbps\n")
        for r in results:
            for elapsed_s, gbps in enumerate(r["per_second_gbps"], start=1):
                ts = (run_start + timedelta(seconds=elapsed_s)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"{ts},{elapsed_s},{r['gpu_index']},{gbps}\n")
    print(f"  Throughput CSV saved -> {csv_path}\n")
    
    for r in results:
        r.pop("per_second_gbps", None)
        
    report = {
        "run_id": run_id,
        "workload": "workload3_random_access_memory",
        "description": "Synthetic random-access memory stress - memory-only workload",
        "cuda_version": torch.version.cuda or "unknown",
        "torch_version": torch.__version__,
        "benchmark_args": vars(args),
        "gpu_hardware": gpu_info,
        "results": results,
        "summary": {
            "total_gpus": n_gpus,
            "mean_sustained_GBps": round(sum(r["sustained_GBps"] for r in results) / n_gpus, 2),
        },  
    }   
    
    out_path = os.path.join(args.output_dir, f"{run_id}_workload3_metadata.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
