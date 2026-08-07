"""
evaluate.py — TriChronos-0.1B
Zero-shot MASE evaluation against the Monash Time Series Forecasting benchmark.

Designed to run on a free CPU tier — no GPU required.

Metrics
-------
MASE = MAE(forecast, actual) / naïve_seasonal_MAE
     where naïve_seasonal_MAE uses the last known value from the context
     as the forecast (seasonal period = 1, i.e. random-walk naive baseline).

For probabilistic forecasts we evaluate the **median quantile** (τ=0.50).

Usage
-----
  python evaluate.py --checkpoint checkpoints/model_state.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader

from data_pipeline import (
    _silver_normalize,
    _gold_patch,
    PATCH_SIZE,
    FORECAST_HORIZON,
)
from model import TriChronos, QUANTILE_LEVELS


# ---------------------------------------------------------------------------
# Monash datasets to evaluate on
# ---------------------------------------------------------------------------
# Each entry: (hf_dataset_name, subset_name_or_None, seasonal_period)
MONASH_DATASETS: List[Tuple[str, Optional[str], int]] = [
    ("monash_tsf_data", "m1_monthly",       12),
    ("monash_tsf_data", "m1_quarterly",      4),
    ("monash_tsf_data", "m1_yearly",         1),
    ("monash_tsf_data", "m3_monthly",       12),
    ("monash_tsf_data", "m3_quarterly",      4),
    ("monash_tsf_data", "m3_yearly",         1),
    ("monash_tsf_data", "m4_monthly",       12),
    ("monash_tsf_data", "m4_quarterly",      4),
    ("monash_tsf_data", "m4_yearly",         1),
    ("monash_tsf_data", "tourism_monthly",  12),
    ("monash_tsf_data", "tourism_quarterly", 4),
    ("monash_tsf_data", "tourism_yearly",    1),
    ("monash_tsf_data", "electricity_hourly", 24),
    ("monash_tsf_data", "traffic_hourly",    24),
    ("monash_tsf_data", "weather",            1),
]

# Median is quantile index for τ=0.50
MEDIAN_IDX: int = QUANTILE_LEVELS.index(0.5)


# ---------------------------------------------------------------------------
# Naive seasonal baseline
# ---------------------------------------------------------------------------

def naive_mae(history: np.ndarray, horizon: int, period: int) -> float:
    """
    MAE of the seasonal-naive forecast on the history (in-sample).

    For period=1 (random walk): forecast[t] = history[t-1].
    """
    if len(history) <= period:
        return float(np.abs(history).mean()) or 1.0
    errors = np.abs(history[period:] - history[:-period])
    return float(errors.mean()) or 1.0


# ---------------------------------------------------------------------------
# Single-dataset evaluation
# ---------------------------------------------------------------------------

def evaluate_dataset(
    model: TriChronos,
    ds_name: str,
    subset: Optional[str],
    period: int,
    device: torch.device,
    max_series: int = 200,
) -> Tuple[float, int]:
    """
    Run zero-shot evaluation on one Monash dataset.

    Returns
    -------
    mean_mase : float  — mean MASE across all evaluated series
    n_series  : int    — number of series evaluated
    """
    from datasets import load_dataset

    try:
        ds_kwargs = dict(
            path=ds_name,
            split="test",
            trust_remote_code=True,
        )
        if subset:
            ds_kwargs["name"] = subset
        dataset = load_dataset(**ds_kwargs)
    except Exception as exc:
        print(f"  [SKIP] {subset or ds_name}: {exc}")
        return float("nan"), 0

    mase_scores: List[float] = []
    model.eval()

    with torch.no_grad():
        for i, example in enumerate(dataset):
            if i >= max_series:
                break

            values = example.get("target", None)
            if values is None or len(values) < PATCH_SIZE * 2 + FORECAST_HORIZON:
                continue
            values = np.asarray(values, dtype=np.float32)

            # Normalise
            normed, mean_, std_ = _silver_normalize(values)

            # Use the last window as context; forecast FORECAST_HORIZON steps
            # (if the series is shorter than FORECAST_HORIZON, skip it)
            if len(normed) < FORECAST_HORIZON * 2:
                continue

            # Context = all but last FORECAST_HORIZON steps
            context = normed[: -FORECAST_HORIZON]
            actual_normed = normed[-FORECAST_HORIZON:]

            # Build patches from context
            n_full = len(context) // PATCH_SIZE
            if n_full == 0:
                continue
            aligned = context[-n_full * PATCH_SIZE:]
            patches = aligned.reshape(n_full, PATCH_SIZE)

            # Pad/truncate to max_patches=64
            max_patches = 64
            if patches.shape[0] > max_patches:
                patches = patches[-max_patches:]
            elif patches.shape[0] < max_patches:
                pad = np.zeros((max_patches - patches.shape[0], PATCH_SIZE), dtype=np.float32)
                patches = np.concatenate([pad, patches], axis=0)

            patches_t = torch.from_numpy(patches).unsqueeze(0).to(device)  # (1, 64, 8)

            # Model forward → (1, horizon, n_quantiles)
            preds = model(patches_t)
            # Median prediction
            median_normed = preds[0, :FORECAST_HORIZON, MEDIAN_IDX].cpu().numpy()

            # Compute MAE in normalised space
            mae = float(np.abs(median_normed - actual_normed).mean())

            # Naive baseline MAE (on normalised context)
            n_mae = naive_mae(context, FORECAST_HORIZON, period)

            mase = mae / max(n_mae, 1e-8)
            mase_scores.append(mase)

    if not mase_scores:
        return float("nan"), 0

    return float(np.mean(mase_scores)), len(mase_scores)


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------

def evaluate(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}\n", flush=True)

    # Load model
    model = TriChronos(patch_size=PATCH_SIZE, horizon=FORECAST_HORIZON).to(device)
    if args.checkpoint and Path(args.checkpoint).exists():
        state = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(state)
        print(f"Loaded checkpoint: {args.checkpoint}\n", flush=True)
    else:
        print("⚠️  No checkpoint loaded — evaluating with random weights (sanity check only).\n")

    model.eval()

    # Per-dataset results
    results: Dict[str, Dict] = {}
    all_mase: List[float] = []

    header = f"{'Dataset':<35} {'MASE':>8} {'N':>6}"
    print(header)
    print("-" * len(header))

    for ds_name, subset, period in MONASH_DATASETS:
        label = subset or ds_name
        mase, n = evaluate_dataset(model, ds_name, subset, period, device, max_series=args.max_series)
        results[label] = {"mase": mase, "n": n}

        if not np.isnan(mase):
            all_mase.append(mase)
            print(f"{label:<35} {mase:>8.4f} {n:>6}")
        else:
            print(f"{label:<35} {'N/A':>8} {n:>6}")

    print("-" * len(header))
    if all_mase:
        agg_mase = float(np.mean(all_mase))
        print(f"{'AGGREGATE (mean MASE)':<35} {agg_mase:>8.4f}")
    else:
        print("No datasets were successfully evaluated.")

    return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate TriChronos-0.1B on Monash")
    p.add_argument(
        "--checkpoint",
        type=str,
        default="checkpoints/model_state.pt",
        help="Path to model checkpoint",
    )
    p.add_argument(
        "--max-series",
        type=int,
        default=200,
        help="Max series per dataset (limits CPU eval time)",
    )
    return p.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
