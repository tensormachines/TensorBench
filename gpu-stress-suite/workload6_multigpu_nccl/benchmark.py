"""
Workload 6: Multi-GPU NCCL Collective Stress
Category  : Multi-GPU interconnect / collective communication
Purpose   : Stress NCCL all-reduce, all-gather, and broadcast paths.

On single-GPU machines this workload writes a skipped result artifact and exits
successfully unless --require-multi-gpu is passed.
"""

import argparse
import json
import os
import socket
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

if 'TZ' in os.environ:
    time.tzset()
      
# DUTY_CYCLE = .95
TIMEZONE = 0  # UTC for standardization


# def utc_stamp():
#     return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


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


def write_result(payload, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    run_id = payload["run_id"]
    target = output_path / f"{run_id}_workload6_metadata.json"
    target.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"  Results saved -> {target}", flush=True)


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def percentile(values, q):
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round((len(ordered) - 1) * q))))
    return ordered[idx]


def run_collective(op_name, tensor, world_size, duration, warmup_iters):
    gather_buffers = None
    if op_name == "all_gather":
        gather_buffers = [torch.empty_like(tensor) for _ in range(world_size)]

    def one_op():
        if op_name == "all_reduce":
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        elif op_name == "all_gather":
            dist.all_gather(gather_buffers, tensor)
        elif op_name == "broadcast":
            dist.broadcast(tensor, src=0)
        else:
            raise ValueError(f"Unknown operation: {op_name}")

    for _ in range(warmup_iters):
        one_op()
    torch.cuda.synchronize()
    dist.barrier()

    tensor_bytes = tensor.numel() * tensor.element_size()
    if op_name == "all_reduce":
        bytes_per_gpu = tensor_bytes * 2 * (world_size - 1) / world_size
    elif op_name == "all_gather":
        bytes_per_gpu = tensor_bytes * (world_size - 1)
    else:
        bytes_per_gpu = tensor_bytes
        
    latencies = []
    deadline = time.perf_counter() + duration
    loop_start = time.perf_counter() 
    per_second_gbps = []
    _bucket = []
    _current_second = 0
    
    while time.perf_counter() < deadline:
        start = time.perf_counter()
        one_op()
        torch.cuda.synchronize()
        dist.barrier()
        elapsed = time.perf_counter() - start
        latencies.append(elapsed)
        _elapsed_s = int(time.perf_counter() - loop_start)
        if _elapsed_s > _current_second:
            if _bucket: 
                per_second_gbps.append(round(sum(_bucket) / len(_bucket), 3))
            _bucket = []
            _current_second = _elapsed_s
        _bucket.append((bytes_per_gpu / elapsed) / 1e9)
        
    if _bucket:
        per_second_gbps.append(round(sum(_bucket) / len(_bucket), 3))



    gbps = [(bytes_per_gpu / seconds) / 1e9 for seconds in latencies if seconds > 0]
    return {
        "operation": op_name,
        "iterations": len(latencies),
        "tensor_mb": round(tensor_bytes / 1024**2, 3),
        "latency_ms_avg": round((sum(latencies) / max(len(latencies), 1)) * 1000, 4),
        "latency_ms_p50": round(percentile(latencies, 0.50) * 1000, 4),
        "latency_ms_p95": round(percentile(latencies, 0.95) * 1000, 4),
        "sustained_gbps_per_gpu": round(sum(gbps) / max(len(gbps), 1), 3),
        "peak_gbps_per_gpu": round(max(gbps) if gbps else 0.0, 3),
        "per_second_gbps": per_second_gbps,
    }


def worker(rank, world_size, args, port, result_queue):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(port)
    torch.cuda.set_device(rank)
    dist.init_process_group("nccl", rank=rank, world_size=world_size)

    elements = int(args.tensor_mb * 1024**2 / 4)
    tensor = torch.ones(elements, device=f"cuda:{rank}", dtype=torch.float32)

    rank_results = []
    for op_name in args.operations:
        metrics = run_collective(op_name, tensor, world_size, args.duration, args.warmup_iters)
        if rank == 0:
            rank_results.append(metrics)

    dist.barrier()
    dist.destroy_process_group()

    if rank == 0:
        result_queue.put(rank_results)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--tensor-mb", type=int, default=512)
    parser.add_argument("--warmup-iters", type=int, default=10)
    parser.add_argument("--operations", nargs="+", default=["all_reduce", "all_gather", "broadcast"])
    parser.add_argument("--output-dir", default="/workspace/results")
    parser.add_argument("--max-vram-gb", type=float, default=12.0)
    parser.add_argument("--require-multi-gpu", action="store_true")
    parser.add_argument("--env", nargs="*", default=[], metavar="KEY=VALUE",
                        help="NCCL env vars")
    args = parser.parse_args()

    for pair in args.env:
        key, _, value = pair.partition("=")
        os.environ[key] = value

    run_start = datetime.now(timezone(timedelta(hours=TIMEZONE)))
    run_id = run_start.strftime("%y-%m-%d_%H-%M")
    gpu_count = torch.cuda.device_count()
    gpu_info = get_gpu_info()

    if gpu_count < 2:
        payload = {
            "run_id": run_id,
            "workload": "workload6_multigpu_nccl",
            "description": "Multi-GPU NCCL collective stress",
            "status": "failed" if args.require_multi_gpu else "skipped",
            "skip_reason": f"requires >=2 GPUs; detected {gpu_count}",
            "benchmark_args": vars(args),
            "gpu_hardware": gpu_info,
            "results": [],
        }
        write_result(payload, args.output_dir)
        if args.require_multi_gpu:
            raise SystemExit(2)
        print(f"  SKIPPED: requires >=2 GPUs; detected {gpu_count}", flush=True)
        return

    print("=" * 60)
    print("  Multi-GPU NCCL Collective Stress - Workload 6")
    print(f"  GPUs      : {gpu_count} detected")
    print(f"  Tensor    : {args.tensor_mb} MB")
    print(f"  Duration  : {args.duration}s per operation")
    print("=" * 60)

    _ctx = mp.get_context("spawn")
    result_queue = _ctx.SimpleQueue()
    port = find_free_port()
    mp.spawn(worker, args=(gpu_count, args, port, result_queue), nprocs=gpu_count, join=True)
    results = result_queue.get()
    
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    csv_path = os.path.join(args.output_dir, f"{run_id}_workload6_nccl.csv")
    with open(csv_path, "w") as f:
        f.write("timestamp,elapsed_s,operation,gbps_per_gpu\n")
        for r in results:
            for elapsed_s, gbps in enumerate(r["per_second_gbps"], start=1):
                ts = (run_start + timedelta(seconds=elapsed_s)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                f.write(f"{ts},{elapsed_s},{r['operation']},{gbps}\n")
    print(f"  Throughput CSV saved -> {csv_path}\n")
    
    for r in results:
        r.pop("per_second_gbps", None)

    payload = {
        "run_id": run_id,
        "workload": "workload6_multigpu_nccl",
        "description": "Multi-GPU NCCL collective stress",
        "status": "completed",
        "benchmark_args": vars(args),
        "gpu_hardware": gpu_info,
        "results": results,
    }
    out_path = os.path.join(args.output_dir, f"{run_id}_workload6_metadata.json")
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"  Results saved -> {out_path}\n")


if __name__ == "__main__":
    main()
