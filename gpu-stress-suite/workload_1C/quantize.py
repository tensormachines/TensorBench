"""
Workload 1C — reproducible INT8 quantization step.

There is no canonical first-party GPTQ-INT8 LLaMA-3-8B repository, and the
trusted-source policy forbids pulling community re-quantizations. So the INT8
model is produced *here*, deterministically, from Meta's official gated weights:

    source weights : meta-llama/Meta-Llama-3-8B-Instruct  (HF_TOKEN, gated)
    method         : AutoGPTQ, 8-bit, group_size 128, symmetric, act-order off
    calibration    : the frozen prompt corpus (frozen_prompts.json) only —
                     no external dataset, no extra network egress
    determinism    : all RNG seeded; desc_act=False removes the only
                     data-order-sensitive step

The resulting directory is what vLLM loads at run time (quant_method=gptq,
which has a real sm_70 kernel path on Volta — unlike W8A8 INT8). Quantization
runs once; the output is cached on a host-mounted volume so subsequent runs
skip it. Re-running with the same inputs yields the same weights.
"""

import argparse
import hashlib
import json
import os
import random
from pathlib import Path

import numpy as np
import torch


CALIB_SAMPLES = 128
CALIB_SEQ_LEN = 512
SEED = 0


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _load_corpus(path: Path) -> list[str]:
    with open(path, "rb") as f:
        raw = f.read()
    sha = hashlib.sha256(raw).hexdigest()
    texts = json.loads(raw)["base_texts"]
    print(f"  [quantize] corpus {path.name} sha256={sha} ({len(texts)} base texts)", flush=True)
    return texts


def _build_calibration(tokenizer, texts: list[str], device: str) -> list[dict]:
    """Deterministic calibration set: tile the corpus to a fixed number of
    fixed-length token windows. No sampling, no shuffling.

    Tensors are placed on `device` (cuda when available). AutoGPTQ moves
    model layers to CUDA during quantize(), and transformers 4.43.4's
    LlamaRotaryEmbedding builds `position_ids` on the inputs' device — if
    inputs live on CPU, the rotary bmm against the (CUDA) inv_freq throws
    `Expected all tensors to be on the same device`. Keeping calibration on
    cuda from the start avoids that mismatch.
    """
    joined = []
    for t in texts:
        joined.extend(tokenizer(t, add_special_tokens=False)["input_ids"])
    if len(joined) < CALIB_SEQ_LEN:
        joined = (joined * (CALIB_SEQ_LEN // max(1, len(joined)) + 1))
    examples = []
    for i in range(CALIB_SAMPLES):
        start = (i * CALIB_SEQ_LEN) % (len(joined) - CALIB_SEQ_LEN + 1)
        window = joined[start:start + CALIB_SEQ_LEN]
        ids = torch.tensor([window], dtype=torch.long, device=device)
        examples.append({"input_ids": ids, "attention_mask": torch.ones_like(ids)})
    return examples


def main() -> None:
    ap = argparse.ArgumentParser(description="Deterministic AutoGPTQ-INT8 quantizer for 1C")
    ap.add_argument("--base-model", default="meta-llama/Meta-Llama-3-8B-Instruct")
    ap.add_argument("--revision", default=None,
                    help="Pin the source repo to a commit SHA. Recommended; "
                         "the resolved revision is written to the output metadata.")
    ap.add_argument("--out-dir", default="/workspace/model_int8")
    ap.add_argument("--corpus", default="/workspace/frozen_prompts.json")
    args = ap.parse_args()

    out = Path(args.out_dir)
    done_marker = out / "QUANT_OK.json"
    if done_marker.exists():
        print(f"  [quantize] cached INT8 model present at {out} — skipping.", flush=True)
        return

    from transformers import AutoTokenizer
    from auto_gptq import AutoGPTQForCausalLM, BaseQuantizeConfig

    _seed_everything(SEED)

    tok = AutoTokenizer.from_pretrained(args.base_model, revision=args.revision, use_fast=True)
    corpus = _load_corpus(Path(args.corpus))
    calib_device = "cuda" if torch.cuda.is_available() else "cpu"
    if calib_device == "cpu":
        print("  [quantize] WARNING no CUDA visible — calibration will run on CPU "
              "(slow, and not the production path).", flush=True)
    calib = _build_calibration(tok, corpus, calib_device)

    qcfg = BaseQuantizeConfig(
        bits=8,
        group_size=128,
        desc_act=False,   # act-order on would make calibration order matter
        sym=True,
        damp_percent=0.01,
    )
    print(f"  [quantize] loading {args.base_model} (revision={args.revision}) ...", flush=True)
    model = AutoGPTQForCausalLM.from_pretrained(
        args.base_model, qcfg, revision=args.revision, torch_dtype=torch.float16
    )
    # transformers 4.43+ moved Llama's rotary_emb from each decoder layer up
    # to the top LlamaModel. auto-gptq 0.7.1's quantize() hook-moves only
    # the *decoder layers* to CUDA one at a time; the top-level rotary_emb
    # and embed_tokens stay on CPU, so the first calibration forward builds
    # position_ids on CPU and tries to bmm them against inv_freq (CUDA
    # inside a moved layer), producing:
    #   RuntimeError: Expected all tensors to be on the same device,
    #   but found at least two devices, cpu and cuda:0
    #
    # The fix is to move ONLY the two top-level submodules that need to
    # match the layer-device (embed_tokens, rotary_emb), and leave decoder
    # layers on CPU so auto-gptq can shuttle them one at a time as it
    # quantizes. Earlier this code did `model.to("cuda")` which moves the
    # whole model — that worked for 8B (~16 GB) but OOMs at pack time on
    # 13B (~26 GB FP16 + ~13 GB packed weights coexist briefly > 32 GB).
    # The targeted move below works for both sizes.
    if torch.cuda.is_available():
        print("  [quantize] moving embed_tokens + rotary_emb to cuda before quantize ...", flush=True)
        llama_model = model.model.model    # AutoGPTQ -> HF CausalLM -> LlamaModel
        llama_model.embed_tokens = llama_model.embed_tokens.to("cuda")
        llama_model.rotary_emb = llama_model.rotary_emb.to("cuda")
    print(f"  [quantize] running GPTQ 8-bit on {CALIB_SAMPLES}x{CALIB_SEQ_LEN} calibration windows ...",
          flush=True)
    model.quantize(calib)

    out.mkdir(parents=True, exist_ok=True)
    model.save_quantized(str(out), use_safetensors=True)
    tok.save_pretrained(str(out))

    resolved_rev = args.revision
    try:
        resolved_rev = model.config._commit_hash or args.revision
    except Exception:
        pass

    meta = {
        "base_model": args.base_model,
        "requested_revision": args.revision,
        "resolved_revision": resolved_rev,
        "method": "autogptq",
        "bits": 8,
        "group_size": 128,
        "sym": True,
        "desc_act": False,
        "damp_percent": 0.01,
        "calib_samples": CALIB_SAMPLES,
        "calib_seq_len": CALIB_SEQ_LEN,
        "calib_source": "frozen_prompts.json",
        "seed": SEED,
    }
    with open(done_marker, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  [quantize] done -> {out}", flush=True)


if __name__ == "__main__":
    main()
