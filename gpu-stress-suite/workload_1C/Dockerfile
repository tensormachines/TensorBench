# =============================================================================
# Workload 1C - LLaMA-3 8B INT8 (GPTQ) integrated inference baseline
# Base : nvcr.io/nvidia/pytorch:24.09-py3  (CUDA 12.6)
# Source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#
# vLLM 0.5.4 is the last release with a working Volta sm_70 wheel. Installing
# it pins torch==2.4.0+cu121, which is ABI-compatible with the CUDA 12.6
# runtime in the base image. Pins are exact for reproducibility.
# =============================================================================
FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace

# Order matters: vllm pins torch==2.4.0; install it first so the rest resolve
# against that torch. auto-gptq builds its CUDA extension against the same
# torch (the NGC image ships the CUDA toolkit, so nvcc is available).
#
# NOTE: `optimum` is intentionally NOT pinned here. optimum<=1.21.x caps
# transformers<4.43.0, which is irreconcilable with vllm 0.5.4's
# transformers>=4.43.2 requirement. quantize.py uses the native AutoGPTQ
# API (AutoGPTQForCausalLM), not optimum's GPTQ path, and auto-gptq==0.7.1
# does not hard-require optimum — so it is simply omitted.
#
# NGC images set PIP_CONSTRAINT=/etc/pip/constraint.txt (hash-pinned to the
# exact package set NVIDIA validated). vLLM 0.5.4 needs an older dependency
# stack than that constraint allows, so pip enters hash-checking mode and
# fails with "PACKAGES DO NOT MATCH THE HASHES". We clear PIP_CONSTRAINT for
# our install; reproducibility is preserved by our own exact == pins.
#
# peft is pinned: auto-gptq==0.7.1 does
# `from peft.peft_model import PEFT_TYPE_TO_MODEL_MAPPING`, a symbol removed in
# newer peft. Unpinned, the resolver pulls peft 0.19.x and auto_gptq fails to
# import. 0.11.1 is the newest release that still exposes the symbol and is
# era-compatible with transformers 4.43.4.
ENV PIP_CONSTRAINT=
RUN pip install --no-cache-dir \
        vllm==0.5.4 \
        transformers==4.43.4 \
        accelerate==0.33.0 \
        peft==0.11.1 \
        numpy==1.26.4 \
 && pip install --no-cache-dir --no-build-isolation auto-gptq==0.7.1

# The NGC base ships CUDA extensions (transformer_engine, flash_attn, apex)
# precompiled against ITS torch build. vLLM 0.5.4 installs the upstream PyPI
# torch==2.4.0, a different C++ ABI, so those NGC .so files fail with
# `undefined symbol` the moment transformers/accelerate opportunistically
# import them. We don't use any of them (vLLM ships its own attention; the
# quantize calibration runs eager), so remove them. Without this, `import
# vllm` / `from auto_gptq import ...` crash.
RUN pip uninstall -y transformer_engine transformer-engine flash_attn flash-attn apex \
        cudf cudf-cu12 torch-tensorrt 2>/dev/null || true

# Vendored first-party `pyairports` shim. vLLM 0.5.4 -> outlines (every
# version in its allowed >=0.0.43,<0.1 range) imports `pyairports` at module
# load; the only distribution on the pinned index is the structurally broken
# pyairports==0.0.1 (ships no module), so `import vllm` dies. We never use
# guided decoding's airport type. See vendor/pyairports/airports.py.
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

# entrypoint: quantize-once (cached) -> benchmark. All args pass through to
# benchmark.py.
ENTRYPOINT ["/workspace/entrypoint.sh"]
