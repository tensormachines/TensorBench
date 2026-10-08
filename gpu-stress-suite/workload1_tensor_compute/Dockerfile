# =============================================================================
# Workload 1: Tensor Compute Stress
# Base  : nvcr.io/nvidia/pytorch:24.09-py3  (ships with CUDA 12.6)
# Source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.09-py3

# No host packages installed — all deps are in the NGC base image.
# The base image already contains: PyTorch, CUDA 12.6, cuBLAS, cuDNN.

WORKDIR /workspace

COPY benchmark.py /workspace/benchmark.py

# Results directory — mount a host volume here to persist logs
RUN mkdir -p /workspace/results

# Default: 60s benchmark, FP16, 8192x8192 matrices
ENTRYPOINT ["python", "/workspace/benchmark.py"]
# CMD ["--duration", "60", "--precision", "fp16", "--matrix-size", "8192", "--batch-size", "4", "--max-vram-gb", "12"]
