# Trusted Source Manifest

This document lists the external software sources used in the GPU Stress Suite.

## Docker Base Image

| Source | Tag | Verification |
|---|---|---|
| NVIDIA NGC PyTorch | `nvcr.io/nvidia/pytorch:24.09-py3` | SHA256: `0603cdfa7c20c77a2cb3e87e5112ecbd68f8af2e45945aae4779ccc513c45446` |

Why trusted:
- official NVIDIA vendor container
- ships with CUDA 12.6 and NVIDIA-tested PyTorch stack
- pinned image tag and digest for reproducibility

## Runtime Libraries

These workloads use libraries already present inside the pinned NGC image:

| Library | Version in image | Used in |
|---|---|---|
| PyTorch | 2.5.0a0+b465a5843b.nv24.09 | W1, W2, W3, W4, W5 |
| CUDA Toolkit | 12.6 | All |
| cuBLAS | 12.6 | W1 |
| cuDNN | 9.x | W2 |
| cuFFT | 12.6 stack via CUDA image components | W4 |
| NCCL | NGC image default | W6 |

No extra pip packages are required for workloads 1-6.

Workload 7 can optionally install these runtime dependencies when built with `INSTALL_HF_DEPS=1`:

| Library | Used for |
|---|---|
| transformers `4.46.3` | Open model loading and causal-LM training |
| datasets `2.21.0` | Public dataset loading |
| accelerate `0.34.2` | Hugging Face runtime support |
| sentencepiece `0.2.0` | Tokenizer support for LLaMA-family models |
| pyarrow `16.1.0` | Dataset table backend compatible with the NGC image |

## Models and Datasets

Optional for Workload 7.

Workloads 1-6 use synthetic inputs only:
- random matrices
- random image-shaped tensors
- random-access index tensors
- synthetic FFT inputs
- synthetic memory buffers
- synthetic NCCL collective tensors

Workload 7 supports public Hugging Face datasets and open model IDs. The default configured public dataset is:

- `wikitext`, config `wikitext-2-raw-v1`

The full assessment currently runs three small public/open model-training presets:

- `hf-internal-testing/tiny-random-LlamaForCausalLM`
- `sshleifer/tiny-gpt2`
- `hf-internal-testing/tiny-random-OPTForCausalLM`

The direct workload default is `hf-internal-testing/tiny-random-LlamaForCausalLM`. Larger model IDs, such as TinyLlama or licensed LLaMA-family checkpoints, can be supplied at runtime when the target environment has the required network access, model permissions, and sufficient GPU memory.

## Verification Commands

```bash
# Verify the pinned image digest
docker inspect nvcr.io/nvidia/pytorch:24.09-py3 --format='{{index .RepoDigests 0}}'

# Check GPU visibility inside Docker
docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu22.04 nvidia-smi -L

# Inspect installed torch version in a workload image
docker run --rm --entrypoint python gpu-stress-workload1 -c "import torch; print(torch.__version__)"
```

*Last updated: May 2026*
