"""
publish.py — TriChronos-0.1B
Package trained weights + model card and push to Hugging Face Hub as v0.1.0.

Usage
-----
  HF_TOKEN=hf_... python publish.py \
      --repo-id your-username/trichronos-0.1b \
      --checkpoint checkpoints/model_state.pt
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from pathlib import Path

import torch

from model import TriChronos, QUANTILE_LEVELS
from data_pipeline import PATCH_SIZE, FORECAST_HORIZON


# ---------------------------------------------------------------------------
# Model card template
# ---------------------------------------------------------------------------

MODEL_CARD = """\
---
language: en
license: apache-2.0
tags:
  - time-series
  - forecasting
  - quantization
  - bitnet
  - ternary
library_name: pytorch
pipeline_tag: time-series-forecasting
---

# TriChronos-0.1B

**TriChronos-0.1B** is a ~100M-parameter encoder-only Transformer for
probabilistic time-series forecasting, trained with 1.58-bit ternary weight
quantisation (BitNet-style).

## Architecture

| Property | Value |
|---|---|
| Parameters | ~100.8 M |
| d_model | 768 |
| Layers | 14 |
| Heads | 12 |
| Patch size | 8 timesteps |
| Weight precision | 1.58-bit ternary (`{-1, 0, +1}`) |
| Activation precision | 8-bit per-token |
| Training precision | BF16 autocast |
| Output | 21 quantiles (τ = 0.025, 0.05, ..., 0.975) |

## Usage

```python
import torch
from model import TriChronos

model = TriChronos()
model.load_state_dict(torch.load("model_state.pt", map_location="cpu"))
model.eval()

# patches: (batch, n_patches, patch_size=8)
patches = torch.randn(1, 64, 8)
with torch.no_grad():
    quantiles = model(patches)  # (1, 24, 21)
```

## Training

Trained on [Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data)
using an A100-80 GB GPU with a $18 compute budget.

- Optimizer: AdamW, cosine LR, warmup 2 000 steps
- Loss: Pinball / quantile loss (21 quantiles)
- Precision: BF16 autocast; BitLinear compresses weights to ternary in-forward

## Evaluation

Zero-shot MASE on the Monash Time Series Forecasting benchmark.
Results are reported in the repository README.

## Citation

If you use this model, please cite:

```bibtex
@misc{trichronos2024,
  title  = {TriChronos-0.1B: Ternary-Quantised Time-Series Forecasting},
  year   = {2024},
  url    = {https://huggingface.co/{repo_id}}
}
```

## License

Apache 2.0
"""

# ---------------------------------------------------------------------------
# Config to save alongside weights
# ---------------------------------------------------------------------------

MODEL_CONFIG = {
    "model_type": "trichronos",
    "patch_size": PATCH_SIZE,
    "d_model": 768,
    "n_layers": 14,
    "n_heads": 12,
    "ffn_dim": 3072,
    "horizon": FORECAST_HORIZON,
    "dropout": 0.0,                 # eval mode — dropout off
    "n_quantiles": len(QUANTILE_LEVELS),
    "quantile_levels": QUANTILE_LEVELS,
}


# ---------------------------------------------------------------------------
# Publish
# ---------------------------------------------------------------------------

def publish(args: argparse.Namespace):
    from huggingface_hub import HfApi, create_repo

    token = args.token or os.environ.get("HF_TOKEN", "")
    if not token:
        print("ERROR: Hugging Face token required. Set HF_TOKEN or pass --token.", file=sys.stderr)
        sys.exit(1)

    api = HfApi(token=token)
    repo_id = args.repo_id

    # ---- Create repo if it doesn't exist ----
    print(f"Creating / verifying repo: {repo_id} …", flush=True)
    create_repo(repo_id, token=token, exist_ok=True, repo_type="model", private=False)

    # ---- Build a temporary staging directory ----
    with tempfile.TemporaryDirectory() as tmpdir:
        staging = Path(tmpdir)

        # 1. Model weights
        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"ERROR: checkpoint not found at {checkpoint_path}", file=sys.stderr)
            sys.exit(1)

        # Instantiate model + load weights (so we can verify & re-save cleanly)
        print("Loading checkpoint …", flush=True)
        model = TriChronos(patch_size=PATCH_SIZE, horizon=FORECAST_HORIZON)
        state_dict = torch.load(checkpoint_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        n_params = model.count_params()
        print(f"  Parameters: {n_params:,}", flush=True)

        # Save weights to staging
        torch.save(model.state_dict(), staging / "model_state.pt")

        # 2. Config
        (staging / "config.json").write_text(json.dumps(MODEL_CONFIG, indent=2))

        # 3. Architecture source files
        for src_file in ["bitlinear.py", "data_pipeline.py", "model.py", "evaluate.py"]:
            src = Path(src_file)
            if src.exists():
                shutil.copy(src, staging / src.name)

        # 4. Model card
        card = MODEL_CARD.replace("{repo_id}", repo_id)
        (staging / "README.md").write_text(card)

        # ---- Upload all files ----
        print(f"\nUploading to {repo_id} …", flush=True)
        api.upload_folder(
            folder_path=str(staging),
            repo_id=repo_id,
            repo_type="model",
            commit_message="v0.1.0 — TriChronos-0.1B initial release",
        )

        # ---- Tag as v0.1.0 ----
        try:
            api.create_tag(repo_id=repo_id, tag="v0.1.0", repo_type="model")
            print("Tagged as v0.1.0", flush=True)
        except Exception as exc:
            print(f"Tagging failed (tag may already exist): {exc}", flush=True)

    print(f"\n✅  Published: https://huggingface.co/{repo_id}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Publish TriChronos-0.1B to Hugging Face Hub")
    p.add_argument("--repo-id", required=True, help="HF Hub repo id, e.g. username/trichronos-0.1b")
    p.add_argument(
        "--checkpoint",
        default="checkpoints/model_state.pt",
        help="Path to model_state.pt",
    )
    p.add_argument("--token", default="", help="HF write token (or set HF_TOKEN env var)")
    return p.parse_args()


if __name__ == "__main__":
    publish(parse_args())
