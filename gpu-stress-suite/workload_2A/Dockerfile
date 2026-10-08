# =============================================================================
# Workload 2A — LLaMA-2 13B GPTQ-INT8 integrated inference (8 GPUs)
# Base : nvcr.io/nvidia/pytorch:24.09-py3  (CUDA 12.6)
# Source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#
# vLLM 0.5.4 is the last release with a working Volta sm_70 wheel; it pins
# torch==2.4.0+cu121 (ABI-compatible with the CUDA 12.6 runtime).
#
# METHOD DEVIATION (forced by stack constraints; documented in README):
#   The recipe specifies AWQ quantization. autoawq versions split into two
#   incompatible groups vs our pinned stack:
#     - autoawq <= 0.2.6   pins torch<2.4 (clashes with vLLM 0.5.4's torch==2.4)
#     - autoawq >= 0.2.9   pins transformers>=4.45 (clashes with vLLM 0.5.4's
#                           era-pinned transformers 4.43.4)
#   There is no version that satisfies both pins simultaneously. So 2A uses
#   AutoGPTQ-INT8 instead — same INT8 precision, same diagnostic intent,
#   already V100-validated end-to-end in 1C. Different method (GPTQ vs AWQ),
#   identical memory profile.
# =============================================================================
FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace

# All six NGC + vLLM 0.5.4 fixes from M2 ([[m2-ngc-vllm-build-fixes]]),
# applied preemptively here so the build is right on the first try:
#  - optimum is intentionally not pinned (it caps transformers<4.43.0,
#    irreconcilable with vLLM 0.5.4's transformers>=4.43.2)
#  - era-matched peft for any auto-gptq-style integration; harmless here
#  - PIP_CONSTRAINT cleared (NGC's hash-pinned constraint file conflicts
#    with vLLM 0.5.4's older dep stack -> "PACKAGES DO NOT MATCH THE HASHES")
ENV PIP_CONSTRAINT=
RUN pip install --no-cache-dir \
        vllm==0.5.4 \
        transformers==4.43.4 \
        accelerate==0.33.0 \
        peft==0.11.1 \
        numpy==1.26.4 \
 && pip install --no-cache-dir --no-build-isolation auto-gptq==0.7.1

# The NGC base ships CUDA extensions (transformer_engine, flash_attn, apex,
# cudf, torch-tensorrt) precompiled against ITS torch build. vLLM 0.5.4
# installs upstream PyPI torch==2.4.0, a different C++ ABI, so those NGC .so
# files either fail with `undefined symbol` when transformers/accelerate
# import them, or emit pip dependency-resolver warnings every build. We use
# none of them. Remove for a clean and working image. (Same as 1C/3A.)
RUN pip uninstall -y transformer_engine transformer-engine flash_attn flash-attn apex \
        cudf cudf-cu12 torch-tensorrt 2>/dev/null || true

# Vendored first-party `pyairports` shim. vLLM 0.5.4 -> outlines (every ver
# in its allowed >=0.0.43,<0.1 range) imports `pyairports` at module load;
# the only distribution on the pinned index is the structurally broken
# pyairports==0.0.1 (ships no module), so `import vllm` dies without this.
# 2A never invokes guided decoding's airport type.
COPY vendor/pyairports /usr/local/lib/python3.10/dist-packages/pyairports

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY frozen_prompts.json /workspace/frozen_prompts.json
COPY quantize.py         /workspace/quantize.py
COPY benchmark.py        /workspace/benchmark.py
COPY entrypoint.sh       /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh && mkdir -p /workspace/results /workspace/model_int8

# entrypoint: quantize-once-and-cache (deterministic GPTQ-INT8 from official
# Meta weights — same artifact pipeline as 1C) -> benchmark. All args pass
# through to benchmark.py.
ENTRYPOINT ["/workspace/entrypoint.sh"]
# CMD ["--max-vram-gb", "32"]
