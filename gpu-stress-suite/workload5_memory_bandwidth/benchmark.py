"""
Workload 5: Memory Bandwidth Stress (Single-GPU HBM)
Category  : Memory-bound
Source    : PyTorch (https://github.com/pytorch/pytorch) - official Meta source
Purpose   : Measures HBM read/write/copy bandwidth in GB/s.

This version auto-reduces the test buffer on smaller GPUs or tighter VRAM caps
because the benchmark allocates two device buffers.
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


def resolve_effective_buffer_gb(device_id: int, args) -> float:
    """
    The workload needs two large buffers. Use at most ~35% of the allowed VRAM
    per buffer so the combined allocation plus overhead still fits.
    """
    total_memory_gb = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    allowed_gb = resolve_effective_vram_cap_gb(device_id, args) if args.max_vram_gb > 0 else total_memory_gb
    return round(max(0.125, min(args.buffer_gb, allowed_gb * 0.35)), 3)


def measure_bandwidth(device: torch.device, buf_gb: float, duration: int, warmup_iters: int):
    n_elements = int(buf_gb * 1024**3 / 4)  # FP32, 4 bytes each
    a = torch.ones(n_elements, device=device, dtype=torch.float32)
    b = torch.empty(n_elements, device=device, dtype=torch.float32)

    bytes_per_op = {
        "read": a.numel() * a.element_size(),
        "write": b.numel() * b.element_size(),
        "copy": a.numel() * a.element_size() * 2,
    }

    results = {}

    per_second_bw = {}
    for mode in ("read", "write", "copy"):
        for _ in range(warmup_iters):
            if mode == "read":
                _ = a.sum()
            elif mode == "write":
                b.fill_(1.0)
            else:
                b.copy_(a)
        torch.cuda.synchronize(device)

        bw_list = []
        deadline = time.perf_counter() + duration
        loop_start = time.perf_counter()
        _bucket = [] 
        _current_second = 0

        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            if mode == "read":
                _ = a.sum()
            elif mode == "write":
                b.fill_(1.0)
            else:
                b.copy_(a)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            bw_list.append(bytes_per_op[mode] / elapsed / 1e9)
            _elapsed_s = int(time.perf_counter() - loop_start)
            if _elapsed_s > _current_second:
                if _bucket: 
                    per_second_bw.setdefault(mode, []).append(round(sum(_bucket) / len(_bucket), 2))
                _bucket = []
                _current_second = _elapsed_s
            _bucket.append(bytes_per_op[mode] / elapsed / 1e9)
            
        if _bucket:
            per_second_bw.setdefault(mode, []).append(round(sum(_bucket) / len(_bucket), 2))

        trim = max(1, len(bw_list) // 10)
        results[mode] = {
            "peak_GBps": round(max(bw_list), 2),
            "sustained_GBps": round(
                sum(sorted(bw_list)[trim:-trim]) / max(1, len(bw_list) - 2 * trim), 2
            ),
            "iterations": len(bw_list),
        }
        print(
            f"    {mode.upper():5s}: "
            f"Peak {results[mode]['peak_GBps']:.1f} GB/s  |  "
            f"Sustained {results[mode]['sustained_GBps']:.1f} GB/s"
        )

    del a, b
    torch.cuda.empty_cache()
    return results, per_second_bw


def benchmark_device(device_id: int, args) -> dict:
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    props = torch.cuda.get_device_properties(device_id)
    effective_buffer_gb = resolve_effective_buffer_gb(device_id, args)

    if effective_buffer_gb < args.buffer_gb:
        print(
            f"  [GPU {device_id}] Requested buffer {args.buffer_gb:.1f} GB exceeds the "
            f"safe budget for this GPU/VRAM cap. Auto-reducing to {effective_buffer_gb:.3f} GB.",
            flush=True,
        )

    while True:
        try:
            print(
                f"  [GPU {device_id}] {props.name}  - Buffer: {effective_buffer_gb:.3f} GB  "
                f"|  Duration: {args.duration}s/mode",
                flush=True,
            )
            bw, per_second_bw = measure_bandwidth(device, effective_buffer_gb, args.duration, args.warmup_iters)
            break
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            next_buffer_gb = round(effective_buffer_gb / 2, 3)
            if next_buffer_gb < 0.125:
                raise RuntimeError(
                    "Memory bandwidth workload could not fit even a 128 MB buffer on this GPU."
                )
            print(
                f"  [GPU {device_id}] Buffer {effective_buffer_gb:.3f} GB OOM - retrying with "
                f"{next_buffer_gb:.3f} GB.",
                flush=True,
            )
            effective_buffer_gb = next_buffer_gb

    return {
        "gpu_index": device_id,
        "gpu_name": props.name,
        "buffer_gb": effective_buffer_gb,
        "requested_buffer_gb": args.buffer_gb,
        "requested_vram_cap_gb": args.max_vram_gb if args.max_vram_gb > 0 else "unlimited",
        "vram_cap_gb": resolve_effective_vram_cap_gb(device_id, args) if args.max_vram_gb > 0 else "unlimited",
        "bandwidth": bw,
        "note": (
            "Single-GPU HBM bandwidth mode. "
            "On DGX multi-GPU: extend with NCCL AllReduce for NVLink/NVSwitch bandwidth."
        ),
        "per_second_bw": per_second_bw,
    }


def main():
    parser = argparse.ArgumentParser(
        description="HBM Memory Bandwidth Benchmark (Single-GPU)"
    )
    parser.add_argument(
        "--duration", type=int, default=30, help="Duration per mode per GPU in seconds (default: 30)"
    )
    parser.add_argument(
        "--buffer-gb", type=float, default=4.0, help="Requested buffer size in GB (default: 4.0)"
    )
    parser.add_argument(
        "--warmup-iters", type=int, default=10, help="Warm-up iterations per mode (default: 10)"
    )
    parser.add_argument(
        "--max-vram-gb",
        type=float,
        default=12.0,
        help="Max VRAM per GPU in GB (default: 12.0). 0 = unlimited.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="/workspace/results",
        help="Directory for JSON result logs",
    )
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
            torch.cuda.set_per_process_memory_fraction(
                min(effective_cap_gb / total_gb, 1.0), device=gpu_id
            )

    print(f"\n{'='*60}")
    print("  Memory Bandwidth Stress - Workload 5")
    print(f"  Buffer    : {args.buffer_gb:.1f} GB requested per test")
    print(f"  Duration  : {args.duration}s per mode (read / write / copy)")
    print(f"  VRAM cap  : requested {args.max_vram_gb:.1f} GB per GPU (auto-clamped if needed)")
    print(f"  GPUs      : {n_gpus} detected")
    print(f"{'='*60}\n")

    gpu_info = get_gpu_info()
    run_start = datetime.now(timezone(timedelta(hours=TIMEZONE)))
    run_id = run_start.strftime("%y-%m-%d_%H-%M")
    
    ctx = mp.get_context("spawn")
    with ctx.Pool(processes=n_gpus) as pool:
        results = pool.starmap(benchmark_device, [(gpu_id, args) for gpu_id in range(n_gpus)])
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{run_id}_workload5_bandwidth.csv")
    with open(csv_path, "w") as f:
        f.write("timestamp,elapsed_s,gpu_index,mode,gbps\n")
        for r in results:
            for mode, values in r["per_second_bw"].items():
                for elapsed_s, gbps in enumerate(values, start=1):
                    ts = (run_start + timedelta(seconds=elapsed_s)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                    f.write(f"{ts},{elapsed_s},{r['gpu_index']},{mode},{gbps}\n")
    print(f"  Throughput CSV saved -> {csv_path}\n")
    
    for r in results:
        r.pop("per_second_bw", None)
        
    report = {
        "run_id": run_id,
        "workload": "workload5_memory_bandwidth",
        "description": "HBM bandwidth stress - read/write/copy GB/s per GPU",
        "cuda_version": torch.version.cuda or "unknown",
        "torch_version": torch.__version__,
        "benchmark_args": vars(args),
        "gpu_hardware": gpu_info,
        "results": results,
        "summary": {
            "total_gpus": n_gpus,
            "mean_read_GBps": round(sum(r["bandwidth"]["read"]["sustained_GBps"] for r in results) / n_gpus, 2),
            "mean_write_GBps": round(sum(r["bandwidth"]["write"]["sustained_GBps"] for r in results) / n_gpus, 2),
            "mean_copy_GBps": round(sum(r["bandwidth"]["copy"]["sustained_GBps"] for r in results) / n_gpus, 2),
        },  
    }   
    
    out_path = os.path.join(args.output_dir, f"{run_id}_workload5_metadata.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
