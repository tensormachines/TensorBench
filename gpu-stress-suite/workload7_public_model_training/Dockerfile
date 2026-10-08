# =============================================================================
# Workload 7: Public Dataset Open-Model Training
# Base  : nvcr.io/nvidia/pytorch:24.09-py3
# =============================================================================

FROM nvcr.io/nvidia/pytorch:24.09-py3

ARG INSTALL_HF_DEPS=0

RUN if [ "$INSTALL_HF_DEPS" = "1" ]; then \
        python -m pip install --no-cache-dir \
            transformers==4.46.3 \
            datasets==2.21.0 \
            accelerate==0.34.2 \
            sentencepiece==0.2.0 \
            pyarrow==16.1.0; \
    fi

WORKDIR /workspace

COPY benchmark.py /workspace/benchmark.py

RUN mkdir -p /workspace/results /workspace/hf_cache

ENV HF_HOME=/workspace/hf_cache

ENTRYPOINT ["python", "/workspace/benchmark.py"]
# CMD ["--duration", "60", "--model-id", "hf-internal-testing/tiny-random-LlamaForCausalLM", "--dataset-id", "wikitext", "--dataset-config", "wikitext-2-raw-v1", "--max-steps", "20", "--max-vram-gb", "12"]
