# =============================================================================
# Workload 6: Multi-GPU NCCL Collective Stress
# Base  : nvcr.io/nvidia/pytorch:24.09-py3
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace

COPY benchmark.py /workspace/benchmark.py

RUN mkdir -p /workspace/results

ENTRYPOINT ["python", "/workspace/benchmark.py"]
# CMD ["--duration", "60", "--tensor-mb", "512", "--max-vram-gb", "12"]
