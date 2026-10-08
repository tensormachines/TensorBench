# =============================================================================
# Workload 2B — LLaMA-3 8B INT8 + LoRA fine-tuning (8 GPUs, replicated)
# Base : nvcr.io/nvidia/pytorch:24.09-py3  (CUDA 12.6) — same family as the
#        other workloads.
# Source: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch
#
# This is the TRAINING workload, not inference. Stack:
#   - HF transformers loads the frozen FP16 base directly. We do NOT quantize
#     it: the recipe asked for an INT8 base + LoRA, but auto-gptq's
#     PEFT integration is structurally broken against peft 0.11.x (the
#     auto-gptq 0.7.1 GPTQLoraModel does not compose with peft 0.11's
#     internal API -> RecursionError). bitsandbytes-INT8 has unreliable
#     Volta sm_70 support. Falling back to FP16 base preserves 2B's
#     diagnostic intent: 2B exists to measure the HBM **write path**
#     (gradient + optimizer + activation-checkpoint churn), and the base
#     is read-only/frozen regardless of its precision, so the write-path
#     signal is identical. Memory budget on a 32 GB V100 stays well within
#     bounds (~16 GB FP16 base + ~3 GB activations w/ grad checkpointing +
#     tiny LoRA/AdamW = ~20 GB). Documented in README + run metadata.
#   - peft attaches LoRA adapters (FP16, trainable) on top of the frozen
#     base. Standard `get_peft_model` path, no integration helpers.
#   - datasets pulls the frozen Alpaca slice.
#
# We do NOT install vLLM here (training only -> no inference engine
# needed). That also means no `outlines` -> no broken `pyairports`, so the
# vendored shim from 1C/2A/3A is not required in 2B.
# =============================================================================
FROM nvcr.io/nvidia/pytorch:24.09-py3

WORKDIR /workspace

# Same NGC + dependency lessons as M2 ([[m2-ngc-vllm-build-fixes]]):
#  - PIP_CONSTRAINT cleared (NGC hash-pinning conflicts with the auto-gptq +
#    transformers 4.43 dep stack).
#  - era-matched peft (auto-gptq 0.7.1 imports `PEFT_TYPE_TO_MODEL_MAPPING`
#    from peft.peft_model — removed in peft >= 0.12.x).
#  - optimum intentionally absent (caps transformers<4.43.0, conflicts).
ENV PIP_CONSTRAINT=
RUN pip install --no-cache-dir \
        transformers==4.43.4 \
        accelerate==0.33.0 \
        peft==0.11.1 \
        datasets==2.20.0 \
        numpy==1.26.4

# Same NGC-extension purge as 1C/2A/3A: these are precompiled against NGC's
# torch and break against the PyPI torch our installs pull. accelerate
# imports transformer_engine opportunistically, so this is necessary even
# without vLLM.
RUN pip uninstall -y transformer_engine transformer-engine flash_attn flash-attn apex \
        cudf cudf-cu12 torch-tensorrt 2>/dev/null || true

ENV HF_HUB_DISABLE_TELEMETRY=1 \
    DO_NOT_TRACK=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    CUBLAS_WORKSPACE_CONFIG=:4096:8

COPY frozen_prompts.json /workspace/frozen_prompts.json
COPY train.py            /workspace/train.py
COPY entrypoint.sh       /workspace/entrypoint.sh
RUN chmod +x /workspace/entrypoint.sh && mkdir -p /workspace/results

# entrypoint: verify HF_TOKEN then train. All args pass through to train.py.
ENTRYPOINT ["/workspace/entrypoint.sh"]
# CMD ["--max-vram-gb", "32"]
