# PROJECT.md — TriChronos-0.1B

## Overview
TriChronos-0.1B is a from-scratch time-series forecasting model (~100M params) built on 1.58-bit ternary quantization (BitNet-style `BitLinear` layers). The goal is to train, evaluate, and publish an open-source ternary-weight forecasting model under a hard compute budget.

- **Objective:** Train a 100M-parameter time-series forecasting model using 1.58-bit ternary quantization.
- **Compute budget:** $18.00 total
- **Hardware:** Hugging Face Spaces, A100-80GB ($2.50/hr) → ~7.2 hour runway
- **Dataset:** Hugging Face `Salesforce/lotsa_data` (LOTSA), streamed — not downloaded (~1TB, too large to persist)
- **Eval dataset:** Monash Time Series Forecasting archive (zero-shot MASE)
- **Release target:** v0.1.0 on the Hugging Face Hub, open-sourced

## Architecture
- **Type:** Encoder-only Transformer over patched time series
- **Size:** ~100.8M params — `d_model=768`, `14 layers`, `12 heads` (hand-verified param count)
- **Patch embedding:** Standard FP16 linear projection (not quantized)
- **Encoder blocks:** BitLinear layers replace dense layers inside each block
  - Alternates **temporal self-attention** (within-sequence) with **group attention across the batch axis** (multivariate/cross-series correlation)
- **Output head:** FP16 residual projection → 21 quantiles (probabilistic forecast distribution)

### BitLinear (core quantization mechanism)
- Absmean ternary quantization: scale FP16 weights by mean absolute value, round to `{-1, 0, 1}`
- Straight-Through Estimator (STE) so gradients pass through the non-differentiable rounding op
- Activations quantized per-token; weights stay ternary during forward pass
- Everything outside BitLinear (patch embed, output head) stays FP16

## Data pipeline (Bronze/Silver/Gold, streaming)
1. **Bronze (raw stream):** `datasets.load_dataset('Salesforce/lotsa_data', streaming=True)` — never persisted to disk
2. **Silver (normalization):** running/standard scaler + `asinh` transform to handle NaNs and stabilize scale
3. **Gold (patching):** chunk scaled series into non-overlapping 8-timestep patches for the embedding layer

## Training
- Optimizer: AdamW, cosine LR schedule with warmup
- Precision: BF16 autocast for activations; BitLinear compresses weights to ternary during forward pass
- **Hard wall-clock cutoff: 7h10m** — script checkpoints and exits automatically to stay inside the $18 budget
- Known gap: the cutoff stops the training loop but does **not** call the Spaces API to pause the Space itself — pausing is currently manual (or needs `huggingface_hub.pause_space` wired in)

## Evaluation
- Zero-shot MASE against the Monash Time Series Forecasting benchmark
- Designed to run on a free CPU tier (no GPU required for eval)

## Publishing
- Packages architecture code + trained weights + model card
- Pushes to the Hugging Face Hub, tagged `v0.1.0`
- Repo intended to be fully open-source / instantiable by others

## Repo structure
```
bitlinear.py     # Absmean ternary quantization + STE, per-token activation quantization
data_pipeline.py # Bronze → Silver → Gold streaming pipeline
model.py         # Encoder architecture (attention + BitLinear blocks + quantile head)
train.py         # AdamW + cosine LR, BF16 autocast, 7h10m hard cutoff
evaluate.py      # Zero-shot MASE vs. Monash
publish.py       # Package + push to Hugging Face Hub (v0.1.0)
Dockerfile       # PyTorch + CUDA 12 base image
requirements.txt
README.md
```

## Status
- All six milestones scaffolded: BitLinear core, data pipeline, model assembly, training loop, eval script, publish script.
- Param count hand-verified at ~100.8M for the chosen config.
- Code syntax-checked (`py_compile`); **not yet run end-to-end** — `torch` wasn't available in the dev sandbox, so no live forward pass has been executed.

## Open items / risks before spending A100 hours
1. **Space auto-pause is unimplemented.** The training script self-terminates at 7h10m, but nothing pauses the underlying HF Space — need to either watch logs live or wire up `huggingface_hub.pause_space`.
2. **Group attention assumption.** Cross-batch attention only captures real multivariate correlation if the series within a batch are actually related — sanity-check batch composition/sampling strategy against this assumption before trusting the mechanism.
3. **No live smoke test yet.** Need a quick local/CPU or short-GPU forward+backward pass to confirm shapes and gradient flow before committing full A100 budget.

## Next steps
- [ ] Install `torch` + deps locally or in a throwaway environment; run a smoke test (single batch forward/backward)
- [ ] Wire up `huggingface_hub.pause_space` (or equivalent) into the 7h10m cutoff in `train.py`
- [ ] Validate batch sampling strategy for group attention's cross-series assumption
- [ ] Provision the actual A100 Space and kick off the real training run
- [ ] Run `evaluate.py` against Monash once a checkpoint exists
- [ ] Run `publish.py` to tag and push `v0.1.0`