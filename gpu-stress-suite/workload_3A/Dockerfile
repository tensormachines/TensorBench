# =============================================================================
# Workload 3A — LLaMA-2 13B FP16 memory-pressure characterization
# Base : nvcr.io/nvidia/pytorch:24.09-py3  (CUDA 12.6).
#
# vLLM 0.5.4 is the last release with a working Volta sm_70 wheel; it pins
# torch==2.4.0+cu121 (ABI-compatible with the CUDA 12.6 runtime). No
# quantization toolchain is needed here — 3A runs full FP16.
# =============================================================================
FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace

# NGC images set PIP_CONSTRAINT=/etc/pip/constraint.txt (hash-pinned to the
# package set NVIDIA validated). vLLM 0.5.4 needs an older dependency stack
# than that constraint allows, so pip enters hash-checking mode and fails
# with "PACKAGES DO NOT MATCH THE HASHES". We clear PIP_CONSTRAINT for our
# install; reproducibility is preserved by our own exact == pins.
ENV PIP_CONSTRAINT=
RUN pip install --no-cache-dir \
        vllm==0.5.4 \
        transformers==4.43.4 \
        numpy==1.26.4

# The NGC base ships CUDA extensions (transformer_engine, flash_attn, apex)
# precompiled against ITS torch build. vLLM 0.5.4 installs upstream PyPI
# torch==2.4.0 (different C++ ABI), so those NGC .so files fail with
# `undefined symbol` the moment transformers/vLLM import them. 3A uses vLLM's
# own attention and never needs them. Without this, `import vllm` crashes.
RUN pip uninstall -y transformer_engine transformer-engine flash_attn flash-attn apex \
        cudf cudf-cu12 torch-tensorrt 2>/dev/null || true

# Vendored first-party `pyairports` shim. vLLM 0.5.4 -> outlines imports
# `pyairports` at module load; the only distribution on the pinned index is
# the structurally broken pyairports==0.0.1, so `import vllm` dies. Guided
# decoding's airport type is unused here. See vendor/pyairports/airports.py.
COPY vendor/pyairports /usr/local/lib/python3.10/dist-packages/pyairports

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1

COPY frozen_prompts.json /workspace/frozen_prompts.json
COPY benchmark.py        /workspace/benchmark.py
COPY entrypoint.sh       /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh && mkdir -p /workspace/results

ENTRYPOINT ["/workspace/entrypoint.sh"]
# CMD ["--max-vram-gb", "32"]
