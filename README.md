# TriChronos-0.1B

**TriChronos-0.1B** is an open-source, ~100M-parameter encoder-only Transformer
for probabilistic time-series forecasting, trained with **1.58-bit ternary weight
quantisation** (BitNet-style `BitLinear` layers). It outputs **21 quantile
predictions** over a 24-step forecast horizon.

---

## Architecture

| Property | Value |
|---|---|
| Parameters | ~100.8 M |
| `d_model` | 768 |
| Layers | 14 |
| Attention heads | 12 |
| Patch size | 8 timesteps |
| Weight precision | 1.58-bit ternary `{-1, 0, +1}` |
| Activation precision | 8-bit per-token |
| Training precision | BF16 autocast |
| Output | 21 quantile forecasts (τ = 0.025 … 0.975) |

### BitLinear (core quantisation mechanism)

- **Absmean ternary quantisation**: scale FP16 weights by their mean absolute
  value, round to `{-1, 0, +1}`
- **Straight-Through Estimator (STE)**: gradients pass through the rounding op
  unchanged — enabling end-to-end backprop
- **Per-token activation quantisation**: activations quantised to 8-bit
  independently per token
- Everything **outside** BitLinear (patch embedding, output head) stays FP16

### Encoder blocks

Each of the 14 blocks contains:
1. **Temporal self-attention** — standard MHA across the patch sequence
   (all Q/K/V/O projections use `BitLinear`)
2. **Group attention** — cross-series MHA across the batch axis, capturing
   multivariate/cross-dataset correlations
3. **BitLinear FFN** with GELU activation

---

## Repository structure

```
bitlinear.py      # Absmean ternary quantisation + STE + per-token activation quant
data_pipeline.py  # Bronze → Silver → Gold streaming pipeline from LOTSA
model.py          # Encoder architecture + BitLinear blocks + quantile head
train.py          # AdamW + cosine LR, BF16 autocast, 7 h 10 m hard cutoff
evaluate.py       # Zero-shot MASE vs. Monash benchmark (CPU-friendly)
publish.py        # Package + push to Hugging Face Hub (v0.1.0)
Dockerfile        # PyTorch + CUDA 12 base image
requirements.txt
README.md
```

---

## Quick start

### 1. Install dependencies

```bash
pip install torch>=2.3.0 --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

### 2. Smoke test (CPU, no data download)

```bash
python bitlinear.py     # tests quantisation + STE
python data_pipeline.py # tests normalisation + patching on synthetic data
python model.py         # verifies ~100.8 M param count + forward/backward pass
```

### 3. Train on A100 (HF Space)

```bash
# Set environment variables in the HF Space secrets:
#   SPACE_ID = "username/space-name"
#   HF_TOKEN = "hf_..."

python train.py --batch-size 32
# The script hard-exits and pauses the Space after 7 h 10 m.
# To resume: python train.py --resume
```

### 4. Evaluate (CPU-friendly)

```bash
python evaluate.py --checkpoint checkpoints/model_state.pt
```

### 5. Publish to Hugging Face Hub

```bash
HF_TOKEN=hf_... python publish.py \
    --repo-id your-username/trichronos-0.1b \
    --checkpoint checkpoints/model_state.pt
```

### 6. Docker

```bash
# Build
docker build -t trichronos .

# Train (GPU)
docker run --gpus all \
    -e SPACE_ID=username/space-name \
    -e HF_TOKEN=hf_... \
    -v $(pwd)/checkpoints:/app/checkpoints \
    trichronos

# Evaluate (CPU)
docker run \
    -v $(pwd)/checkpoints:/app/checkpoints \
    trichronos python evaluate.py
```

---

## Data pipeline

Training streams from [Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data)
(LOTSA, ~1 TB) — **nothing is persisted to disk**.

| Stage | What happens |
|---|---|
| Bronze | `datasets.load_dataset(…, streaming=True)` |
| Silver | Running z-score + `asinh` transform; NaN-safe |
| Gold | Non-overlapping 8-step patches → `(patches, target)` tensors |

Samples within each batch are **sorted by dataset subset** so that the group
attention mechanism operates on genuinely related series.

---

## Training details

| Setting | Value |
|---|---|
| Compute budget | $18.00 (A100-80 GB @ $2.50/hr ≈ 7.2 h) |
| Hard cutoff | 7 h 10 m wall-clock; auto-pauses HF Space |
| Optimizer | AdamW (`β₁=0.9, β₂=0.95, wd=0.01`) |
| LR schedule | Cosine decay with 2 000-step linear warmup |
| Peak LR | 3 × 10⁻⁴ |
| Batch size | 32 |
| Loss | Pinball (quantile) loss, 21 quantiles |
| Gradient clipping | 1.0 |

---

## Evaluation — Monash MASE (zero-shot)

Results will be published after the first full A100 training run.

Datasets evaluated: M1, M3, M4 (monthly/quarterly/yearly), Tourism, Electricity Hourly, Traffic Hourly, Weather.

---

## Open items / known gaps

1. **Group attention assumption** — cross-batch attention only captures real
   multivariate correlation if the series in a batch are related. The data
   pipeline sorts by subset name; validate this before trusting group-attention
   signal.
2. **No live end-to-end run yet** — `torch` was unavailable in the dev
   environment. Run `python model.py` as a smoke test before the A100 run.

---

## License

Apache 2.0 — see [LICENSE](LICENSE).

## Citation

```bibtex
@misc{trichronos2024,
  title  = {TriChronos-0.1B: Ternary-Quantised Time-Series Forecasting},
  year   = {2024},
  url    = {https://github.com/your-username/trichronos}
}
```
