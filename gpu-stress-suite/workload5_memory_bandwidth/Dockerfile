# =============================================================================
# Workload 5: Memory Bandwidth Stress (Interconnect)
# Base  : nvcr.io/nvidia/pytorch:24.09-py3  (CUDA 12.6)
# Source: PyTorch — official Meta source (https://github.com/pytorch/pytorch)
# Note  : Single-GPU mode measures HBM bandwidth (read / write / copy GB/s).
#         On DGX multi-GPU: extend with NCCL AllReduce for NVLink/NVSwitch.
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace
COPY benchmark.py /workspace/benchmark.py
RUN mkdir -p /workspace/results

ENTRYPOINT ["python", "/workspace/benchmark.py"]
# CMD ["--duration", "30", "--buffer-gb", "4.0", "--max-vram-gb", "12"]
