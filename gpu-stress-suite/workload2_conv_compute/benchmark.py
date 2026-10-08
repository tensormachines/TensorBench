"""
Workload 2: Convolution Compute Stress
Category  : Compute-only
Source    : PyTorch CUDA convolution kernels
Purpose   : Stress GPU compute units with synthetic FP16 Conv2D operations.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta
import torch.multiprocessing as mp
from pathlib import Path

import torch
import torch.nn.functional as F

if 'TZ' in os.environ:
    time.tzset()

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


def conv2d_flops(batch_size: int, in_channels: int, out_channels: int, image_size: int, kernel_size: int) -> int:
    output_size = image_size
    return (
        2
        * batch_size
        * output_size
        * output_size
        * out_channels
        * in_channels
        * kernel_size
        * kernel_size
    )


def run_batch_sweep(device, weight, args):
    for batch_size in sorted(args.batch_sizes, reverse=True):
        try:
            inputs = torch.randn(
                batch_size,
                args.in_channels,
                args.image_size,
                args.image_size,
                device=device,
                dtype=torch.float16,
            )
            with torch.no_grad(), torch.amp.autocast("cuda"):
                _ = F.conv2d(inputs, weight, stride=1, padding=args.kernel_size // 2)
            torch.cuda.synchronize(device)
            del inputs
            torch.cuda.empty_cache()
            print(f"  Batch size {batch_size} fits - using for benchmark.", flush=True)
            return batch_size
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            print(f"  Batch size {batch_size} OOM - trying smaller.", flush=True)
    raise RuntimeError("No configured batch size fits within the VRAM budget.")


def benchmark_device(device_id: int, args) -> dict:
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)

    weight = torch.randn(
        args.out_channels,
        args.in_channels,
        args.kernel_size,
        args.kernel_size,
        device=device,
        dtype=torch.float16,
    )

    print(f"  [GPU {device_id}] Finding largest fitting batch size ...", flush=True)
    batch_size = run_batch_sweep(device, weight, args)
    flops_per_iter = conv2d_flops(
        batch_size, args.in_channels, args.out_channels, args.image_size, args.kernel_size
    )

    inputs = torch.randn(
        batch_size,
        args.in_channels,
        args.image_size,
        args.image_size,
        device=device,
        dtype=torch.float16,
    )

    print(f"  [GPU {device_id}] Warming up ({args.warmup_iters} iters, batch={batch_size}) ...", flush=True)
    with torch.no_grad(), torch.amp.autocast("cuda"):
        for _ in range(args.warmup_iters):
            _ = F.conv2d(inputs, weight, stride=1, padding=args.kernel_size // 2)
    torch.cuda.synchronize(device)

    print(f"  [GPU {device_id}] Running benchmark for {args.duration}s ...", flush=True)
    iter_times = []
    deadline = time.perf_counter() + args.duration
    loop_start = time.perf_counter() 
    per_second_samples = []
    per_second_tflops = []
    _bucket_samples = []
    _bucket_tflops = []
    _current_second = 0

    with torch.no_grad(), torch.amp.autocast("cuda"):
        while time.perf_counter() < deadline:
            t0 = time.perf_counter()
            _ = F.conv2d(inputs, weight, stride=1, padding=args.kernel_size // 2)
            torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - t0
            iter_times.append(elapsed)
            _elapsed_s = int(time.perf_counter() - loop_start)
            if _elapsed_s > _current_second:
                if _bucket_samples:
                    per_second_samples.append(round(sum(_bucket_samples) / len(_bucket_samples), 2))
                    per_second_tflops.append(round(sum(_bucket_tflops) / len(_bucket_tflops), 4))
                _bucket_samples = []
                _bucket_tflops = []
                _current_second = _elapsed_s
            _bucket_samples.append(batch_size / elapsed)
            _bucket_tflops.append(flops_per_iter / elapsed / 1e12)
            
        if _bucket_samples:
          per_second_samples.append(round(sum(_bucket_samples) / len(_bucket_samples), 2))
          per_second_tflops.append(round(sum(_bucket_tflops) / len(_bucket_tflops), 4))

    total_iters = len(iter_times)
    samples_per_iter = batch_size
    throughputs = [samples_per_iter / t for t in iter_times]
    tflops_list = [flops_per_iter / t / 1e12 for t in iter_times]
    trim = max(1, total_iters // 10)

    props = torch.cuda.get_device_properties(device_id)
    result = {
        "gpu_index": device_id,
        "gpu_name": props.name,
        "precision": "fp16",
        "batch_size": batch_size,
        "image_size": args.image_size,
        "in_channels": args.in_channels,
        "out_channels": args.out_channels,
        "kernel_size": args.kernel_size,
        "requested_vram_cap_gb": args.max_vram_gb if args.max_vram_gb > 0 else "unlimited",
        "vram_cap_gb": resolve_effective_vram_cap_gb(device_id, args) if args.max_vram_gb > 0 else "unlimited",
        "total_iterations": total_iters,
        "peak_samples_sec": round(max(throughputs), 2),
        "sustained_samples_sec": round(sum(sorted(throughputs)[trim:-trim]) / max(1, total_iters - 2 * trim), 2),
        "peak_tflops": round(max(tflops_list), 4),
        "sustained_tflops": round(sum(sorted(tflops_list)[trim:-trim]) / max(1, total_iters - 2 * trim), 4),
        "flops_per_iter": flops_per_iter,
        "per_second_samples": per_second_samples,
        "per_second_tflops": per_second_tflops,
    }

    print(
        f"  [GPU {device_id}] Done - "
        f"Peak: {result['peak_samples_sec']:.0f} samples/sec | "
        f"Sustained: {result['sustained_samples_sec']:.0f} samples/sec | "
        f"TFLOP/s: {result['sustained_tflops']:.2f}"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Convolution Compute Stress Benchmark")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[64, 32, 16])
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--in-channels", type=int, default=64)
    parser.add_argument("--out-channels", type=int, default=128)
    parser.add_argument("--kernel-size", type=int, default=3)
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
    print("  Convolution Compute Stress - Workload 2")
    print("  Precision : FP16")
    print(f"  Batch     : auto-select from {sorted(args.batch_sizes, reverse=True)}")
    print(f"  Image     : {args.in_channels}x{args.image_size}x{args.image_size} -> {args.out_channels} channels")
    print(f"  Duration  : {args.duration}s per GPU")
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
    csv_path = os.path.join(args.output_dir, f"{run_id}_workload2_throughput.csv")
    with open(csv_path, "w") as f:
        f.write("timestamp,elapsed_s,gpu_index,samples_sec,tflops\n")
        for r in results:
            for elapsed_s, (s, t) in enumerate(zip(r["per_second_samples"], r["per_second_tflops"]), start=1):
                ts = (run_start + timedelta(seconds=elapsed_s)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"{ts},{elapsed_s},{r['gpu_index']},{s},{t}\n")
    print(f"  Throughput CSV saved -> {csv_path}\n")

    for r in results:
        r.pop("per_second_samples", None)
        r.pop("per_second_tflops", None)

    report = {
        "run_id": run_id,
        "workload": "workload2_convolution_compute",
        "description": "Synthetic Conv2D compute stress - compute-only workload",
        "cuda_version": torch.version.cuda or "unknown",
        "torch_version": torch.__version__,
        "benchmark_args": vars(args),
        "gpu_hardware": gpu_info,
        "results": results,
        "summary": {
            "total_gpus": n_gpus,
            "mean_sustained_samples_sec": round(sum(r["sustained_samples_sec"] for r in results) / n_gpus, 2),
            "mean_sustained_tflops": round(sum(r["sustained_tflops"] for r in results) / n_gpus, 4),
        },
    }

    out_path = os.path.join(args.output_dir, f"{run_id}_workload2_metadata.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved -> {out_path}\n")

if __name__ == "__main__":
    main()
