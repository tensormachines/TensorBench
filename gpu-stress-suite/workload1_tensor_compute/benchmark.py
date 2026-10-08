"""
Workload 1: Tensor Compute Stress (Synthetic GEMM Benchmark)
Category  : Compute-only
Source    : PyTorch CUDA batched matmul
Purpose   : Measure sustained FP16/BF16 GEMM throughput without loading any model.
"""

import argparse
import json
import os
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
import torch.multiprocessing as mp

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
                "sm_count": props.multi_processor_count,
                "cuda_capability": f"{props.major}.{props.minor}",
            }
        )
    return info


def resolve_effective_vram_cap_gb(device_id: int, args, reserve_fraction: float = 0.8) -> float:
    if args.max_vram_gb <= 0:
        return 0.0
    total_gb = torch.cuda.get_device_properties(device_id).total_memory / 1024**3
    return round(min(args.max_vram_gb, total_gb * reserve_fraction), 3)


def candidate_values(start: int, minimum: int):
    values = []
    current = start
    while current >= minimum:
        if current not in values:
            values.append(current)
        if current == minimum:
            break
        current = max(minimum, current // 2)
    return values


def flops_per_matmul(m: int, k: int, n: int, batch: int = 1) -> int:
    return 2 * batch * m * k * n


def pick_runtime_shape(device, args, dtype):
    for matrix_size in candidate_values(args.matrix_size, 512):
        for batch_size in candidate_values(args.batch_size, 1):
            try:
                a = torch.randn(batch_size, matrix_size, matrix_size, device=device, dtype=dtype)
                b = torch.randn(batch_size, matrix_size, matrix_size, device=device, dtype=dtype)
                _ = torch.bmm(a, b)
                torch.cuda.synchronize(device)
                del a, b
                torch.cuda.empty_cache()
                print(
                    f"  Selected runtime shape batch={batch_size}, matrix={matrix_size}.",
                    flush=True,
                )
                return batch_size, matrix_size
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(
                    f"  batch={batch_size}, matrix={matrix_size} OOM - trying smaller shape.",
                    flush=True,
                )
    raise RuntimeError("No GEMM shape fits within the VRAM budget.")


def benchmark_device(device_id: int, args) -> dict:
    device = torch.device(f"cuda:{device_id}")
    torch.cuda.set_device(device)
    os.nice(10)
    dtype = torch.float16 if args.precision == "fp16" else torch.bfloat16

    print(f"  [GPU {device_id}] Selecting a fitting matrix shape ...", flush=True)
    batch_size, matrix_size = pick_runtime_shape(device, args, dtype)
    flops_per_iter = flops_per_matmul(matrix_size, matrix_size, matrix_size, batch_size)

    a = torch.randn(batch_size, matrix_size, matrix_size, device=device, dtype=dtype)
    b = torch.randn(batch_size, matrix_size, matrix_size, device=device, dtype=dtype)

    print(
        f"  [GPU {device_id}] Warming up ({args.warmup_iters} iters, batch={batch_size}, matrix={matrix_size}) ...",
        flush=True,
    )
    for _ in range(args.warmup_iters):
        _ = torch.bmm(a, b)
    torch.cuda.synchronize(device)

    print(f"  [GPU {device_id}] Running benchmark for {args.duration}s ...", flush=True)
    iter_flops = []
    deadline = time.perf_counter() + args.duration
    
    loop_start = time.perf_counter()
    per_second_tflops = []
    _bucket = []
    _current_second = 0
    
    while time.perf_counter() < deadline:
        t0 = time.perf_counter()
        _ = torch.bmm(a, b)
        torch.cuda.synchronize(device)
        elapsed = time.perf_counter() - t0
        iter_flops.append(flops_per_iter / (time.perf_counter() - t0))
        _elapsed_s = int(time.perf_counter() - loop_start)
        if _elapsed_s > _current_second:
            if _bucket:
                per_second_tflops.append(round(sum(_bucket) / len(_bucket) / 1e12, 4))
            _bucket = []
            _current_second = _elapsed_s
        _bucket.append(flops_per_iter / elapsed)
        # time.sleep(elapsed * (1 - DUTY_CYCLE) / DUTY_CYCLE)

    if _bucket:
        per_second_tflops.append(round(sum(_bucket) / len(_bucket) / 1e12, 4))

    total_iters = len(iter_flops)
    trim = max(1, total_iters // 10)
    sustained = sorted(iter_flops)[trim:-trim] if total_iters > 2 * trim else iter_flops
    props = torch.cuda.get_device_properties(device_id)

    result = {
        "gpu_index": device_id,
        "gpu_name": props.name,
        "precision": args.precision,
        "requested_batch_size": args.batch_size,
        "requested_matrix_size": args.matrix_size,
        "batch_size": batch_size,
        "matrix_size": matrix_size,
        "requested_vram_cap_gb": args.max_vram_gb if args.max_vram_gb > 0 else "unlimited",
        "vram_cap_gb": resolve_effective_vram_cap_gb(device_id, args) if args.max_vram_gb > 0 else "unlimited",
        "total_iterations": total_iters,
        "peak_tflops": round(max(iter_flops) / 1e12, 4),
        "avg_tflops": round((sum(iter_flops) / total_iters) / 1e12, 4),
        "sustained_tflops": round((sum(sustained) / len(sustained)) / 1e12, 4),
        "flops_per_iter": flops_per_iter,
        "per_second_tflops": per_second_tflops,
    }

    print(
        f"  [GPU {device_id}] Done - Peak: {result['peak_tflops']:.2f} TFLOP/s | "
        f"Avg: {result['avg_tflops']:.2f} TFLOP/s | Sustained: {result['sustained_tflops']:.2f} TFLOP/s"
    )
    return result


def main():
    parser = argparse.ArgumentParser(description="Synthetic GEMM Compute Stress Benchmark")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--sequential", action="store_true")
    parser.add_argument("--matrix-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "bf16"])
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
    print("  Tensor Compute Stress - Workload 1")
    print(f"  Precision : {args.precision.upper()}")
    print(f"  Matrix    : requested batch={args.batch_size}, matrix={args.matrix_size}")
    print(f"  Duration  : {args.duration}s per GPU")
    print(f"  VRAM cap  : requested {args.max_vram_gb:.1f} GB per GPU (auto-clamped if needed)")
    print(f"  GPUs      : {n_gpus} detected")
    print(f"{'='*60}\n")

    gpu_info = get_gpu_info()
    run_start = datetime.now(timezone(timedelta(hours=TIMEZONE)))  # PDT
    # run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_id = run_start.strftime("%y-%m-%d_%H-%M")
    
    # ctx = mp.get_context("spawn")
    # with ctx.Pool(processes=n_gpus) as pool:    
    #     results = pool.starmap(benchmark_device, [(gpu_id, args) for gpu_id in range(n_gpus)])
    # results = [benchmark_device(gpu_id, args) for gpu_id in range(n_gpus)]
    if args.sequential:
      results = [benchmark_device(gpu_id, args) for gpu_id in range(n_gpus)]
    else:
        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=n_gpus) as pool:
            results = pool.starmap(benchmark_device, [(gpu_id, args) for gpu_id in range(n_gpus)])


    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{run_id}_workload1_tflops.csv")
    with open(csv_path, "w") as f:
        f.write("timestamp,elapsed_s,gpu_index,tflops\n")
        for r in results:
            for elapsed_s, tflops in enumerate(r["per_second_tflops"], start=1):
                ts = (run_start + timedelta(seconds=elapsed_s)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"{ts},{elapsed_s},{r['gpu_index']},{tflops}\n")
    print(f"  Throughput CSV saved -> {csv_path}\n")
    
    for r in results:
        r.pop("per_second_tflops", None)
    
    report = {
        "run_id": run_id,
        "workload": "workload1_tensor_compute",
        "description": "Synthetic GEMM stress - compute-only workload",
        "cuda_version": torch.version.cuda or "unknown",
        "torch_version": torch.__version__,
        "benchmark_args": vars(args),
        "gpu_hardware": gpu_info,
        "results": results,
        "summary": {
            "total_gpus": n_gpus,
            "mean_peak_tflops": round(sum(r["peak_tflops"] for r in results) / n_gpus, 4),
            "mean_avg_tflops": round(sum(r["avg_tflops"] for r in results) / n_gpus, 4),
            "mean_sustained_tflops": round(sum(r["sustained_tflops"] for r in results) / n_gpus, 4),
        },
    }

    print(f"\n{'='*60}")
    print("  RESULTS SUMMARY")
    print(f"{'='*60}")
    for r in results:
        print(
            f"  GPU {r['gpu_index']} ({r['gpu_name']})  "
            f"batch={r['batch_size']} matrix={r['matrix_size']}"
        )
        print(f"    Peak      : {r['peak_tflops']:.2f} TFLOP/s")
        print(f"    Average   : {r['avg_tflops']:.2f} TFLOP/s")
        print(f"    Sustained : {r['sustained_tflops']:.2f} TFLOP/s")
    print(f"{'='*60}\n")

    out_path = os.path.join(args.output_dir, f"{run_id}_workload1_metadata.json")
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"  Results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
