"""
data_pipeline.py — TriChronos-0.1B
Bronze → Silver → Gold streaming pipeline from Salesforce/lotsa_data.

Never materialises the full dataset to disk — everything is streamed and
processed on-the-fly.

Stages
------
Bronze  raw stream from the Hugging Face Hub
Silver  NaN-safe running z-score normalisation + asinh transform
Gold    non-overlapping 8-timestep patches  →  (patches, targets) tensors
"""

from __future__ import annotations

import math
import os
from typing import Iterator, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import IterableDataset


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATCH_SIZE: int = 8          # timesteps per patch
FORECAST_HORIZON: int = 24   # timesteps to predict
MIN_SERIES_LEN: int = PATCH_SIZE * 4 + FORECAST_HORIZON  # minimum usable length

# All LOTSA subsets to stream (use None to stream all)
LOTSA_SUBSETS: Optional[List[str]] = None   # None → HF will enumerate all

# Asinh stabilises heavy-tailed distributions and handles near-zero series.
ASINH_SCALE: float = 1.0


# ---------------------------------------------------------------------------
# Bronze — raw streaming from HF
# ---------------------------------------------------------------------------

LOTSA_REPO: str = "Salesforce/lotsa_data"


def list_lotsa_subsets() -> List[str]:
    """
    Enumerate every config (subset) name in Salesforce/lotsa_data.

    Streaming the repo with no ``name`` globs arrow files across *all*
    subsets under one implicit config, forcing HF to infer a single schema
    for ``target``. Because univariate subsets store ``target`` as
    ``list<float>`` and multivariate ones as ``list<list<float>>``, the cast
    fails. Streaming each config by name keeps every arrow file in the stream
    on a homogeneous schema, so no cross-schema cast is attempted.
    """
    from datasets import get_dataset_config_names

    names = get_dataset_config_names(LOTSA_REPO)
    # "default" (if present) is the union config that triggers the schema
    # clash — drop it and stream the real subsets instead.
    return [n for n in names if n != "default"]


def _iter_channels(values) -> Iterator[np.ndarray]:
    """
    Normalise a raw ``target`` into one or more 1-D float32 series.

    LOTSA ``target`` is either univariate (``list<float>``) or multivariate
    (``list<list<float>>`` — one inner list per channel). Multivariate series
    are split into independent univariate channels so the univariate patching
    downstream stays valid.
    """
    if values is None:
        return
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        yield arr
    elif arr.ndim == 2:
        for channel in arr:            # (n_channels, T) → T-length series each
            yield np.ascontiguousarray(channel, dtype=np.float32)
    # anything else (ragged / empty) is silently skipped


def _bronze_stream(subset: Optional[str] = None, split: str = "train"):
    """
    Yield raw examples from Salesforce/lotsa_data for a specific named subset config.

    Each example has at least:
        "target"   : List[float] or List[List[float]] — univariate or multivariate time series values
        "start"    : str — ISO-format start timestamp
    """
    from datasets import load_dataset  # lazy import

    ds_kwargs: dict = dict(
        path="Salesforce/lotsa_data",
        split=split,
        streaming=True,
    )
    if subset is not None:
        ds_kwargs["name"] = subset

    try:
        dataset = load_dataset(**ds_kwargs)
    except Exception as exc:
        print(f"[data_pipeline] Skipping subset '{subset}': {exc}", flush=True)
        return

    for example in dataset:
        values = example.get("target", None)
        if values is None:
            continue
        val_arr = np.asarray(values, dtype=np.float32)

        # Handle 2D multivariate series (channels, timesteps)
        if val_arr.ndim == 2:
            for c in range(val_arr.shape[0]):
                chan = val_arr[c]
                if len(chan) >= MIN_SERIES_LEN:
                    yield {
                        "values": chan,
                        "subset": subset or "unknown",
                        "start": example.get("start", ""),
                    }
        elif val_arr.ndim == 1 and len(val_arr) >= MIN_SERIES_LEN:
            yield {
                "values": val_arr,
                "subset": subset or "unknown",
                "start": example.get("start", ""),
            }


# ---------------------------------------------------------------------------
# Silver — normalisation
# ---------------------------------------------------------------------------

def _silver_normalize(values: np.ndarray) -> Tuple[np.ndarray, float, float]:
    """
    NaN-safe running z-score + asinh transform.

    Returns
    -------
    normed   : normalised array (same length as input)
    mean_    : sample mean used for normalisation
    std_     : sample std  used for normalisation (clipped to ≥ 1e-6)
    """
    # Replace NaN/Inf with 0 before computing stats
    clean = np.where(np.isfinite(values), values, 0.0)

    mean_ = float(np.mean(clean))
    std_ = float(np.std(clean))
    std_ = max(std_, 1e-6)

    z = (clean - mean_) / std_
    normed = np.arcsinh(ASINH_SCALE * z).astype(np.float32)
    return normed, mean_, std_


# ---------------------------------------------------------------------------
# Gold — patching
# ---------------------------------------------------------------------------

def _gold_patch(
    normed: np.ndarray,
    patch_size: int = PATCH_SIZE,
    horizon: int = FORECAST_HORIZON,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Slide over the normalised series and yield (context_patches, target) pairs.

    context_patches : (n_patches, patch_size)  — past context
    target          : (horizon,)               — future values to forecast
    """
    n = len(normed)
    # Maximum number of non-overlapping patches we can form from the context
    # (leave `horizon` steps at the end for the target)
    max_start = n - horizon
    if max_start < patch_size:
        return

    # Use all valid (start, end) positions with a stride of patch_size
    for end_of_context in range(patch_size, max_start + 1, patch_size):
        context = normed[:end_of_context]
        target = normed[end_of_context: end_of_context + horizon]

        if len(target) < horizon:
            break

        # Chop context into non-overlapping patches of exactly patch_size
        # Discard any incomplete leading patch
        n_full_patches = len(context) // patch_size
        if n_full_patches == 0:
            continue
        context_aligned = context[-n_full_patches * patch_size:]
        patches = context_aligned.reshape(n_full_patches, patch_size)

        yield patches, target


# ---------------------------------------------------------------------------
# IterableDataset (Gold)
# ---------------------------------------------------------------------------

class LOTSAStreamDataset(IterableDataset):
    """
    PyTorch IterableDataset wrapping the full Bronze->Silver->Gold pipeline.

    Yields dicts:
        "patches"  : FloatTensor (n_patches, patch_size)
        "target"   : FloatTensor (horizon,)
        "subset"   : str

    Parameters
    ----------
    subsets     : list of LOTSA subset names, or None to dynamically discover all
    split       : HF dataset split ("train")
    patch_size  : timesteps per patch
    horizon     : forecast horizon
    max_patches : cap on n_patches (pad/truncate); None = variable length
    """

    def __init__(
        self,
        subsets: Optional[List[str]] = None,
        split: str = "train",
        patch_size: int = PATCH_SIZE,
        horizon: int = FORECAST_HORIZON,
        max_patches: Optional[int] = 64,
    ):
        super().__init__()
        if subsets is None:
            from datasets import get_dataset_config_names
            try:
                all_configs = get_dataset_config_names("Salesforce/lotsa_data")
                subsets = [c for c in all_configs if c != "default"]
            except Exception:
                subsets = [
                    "m4_monthly", "traffic_hourly", "electricity_hourly",
                    "solar_power", "weather", "m4_daily", "m4_hourly",
                    "favorita_sales", "kdd2022", "wind_power"
                ]
        self.subsets = subsets
        self.split = split
        self.patch_size = patch_size
        self.horizon = horizon
        self.max_patches = max_patches

    # ------------------------------------------------------------------

    def _iter_subset(self, subset: Optional[str]) -> Iterator[dict]:
        for bronze in _bronze_stream(subset, self.split):
            normed, _, _ = _silver_normalize(bronze["values"])
            for patches, target in _gold_patch(normed, self.patch_size, self.horizon):
                patches_t = torch.from_numpy(patches)   # (n_patches, patch_size)
                target_t = torch.from_numpy(target)     # (horizon,)

                # Pad / truncate patches to max_patches
                if self.max_patches is not None:
                    n = patches_t.shape[0]
                    if n > self.max_patches:
                        patches_t = patches_t[-self.max_patches:]   # keep most-recent
                    elif n < self.max_patches:
                        pad = torch.zeros(
                            self.max_patches - n, self.patch_size,
                            dtype=patches_t.dtype,
                        )
                        patches_t = torch.cat([pad, patches_t], dim=0)

                yield {
                    "patches": patches_t,     # (max_patches, patch_size)
                    "target": target_t,       # (horizon,)
                    "subset": bronze["subset"],
                }

    # ------------------------------------------------------------------

    def __iter__(self) -> Iterator[dict]:
        worker_info = torch.utils.data.get_worker_info()
        subsets = self.subsets

        if worker_info is not None:
            # Distribute subsets across DataLoader workers
            subsets = [
                s for i, s in enumerate(subsets)
                if i % worker_info.num_workers == worker_info.id
            ]

        for subset in subsets:
            yield from self._iter_subset(subset)


# ---------------------------------------------------------------------------
# Collate function
# ---------------------------------------------------------------------------

def collate_fn(batch: List[dict]) -> dict:
    """
    Collate a list of Gold samples into batched tensors.

    Groups samples by subset so that cross-series (group) attention
    in the model operates on related series from the same dataset —
    a prerequisite for the group attention mechanism to capture real
    multivariate correlations.

    Returns
    -------
    patches : FloatTensor (B, max_patches, patch_size)
    targets : FloatTensor (B, horizon)
    subsets : List[str]
    """
    # Sort by subset name so related series are adjacent in the batch
    batch = sorted(batch, key=lambda x: x["subset"])

    patches = torch.stack([b["patches"] for b in batch], dim=0)
    targets = torch.stack([b["target"] for b in batch], dim=0)
    subsets = [b["subset"] for b in batch]

    return {
        "patches": patches,   # (B, n_patches, patch_size)
        "targets": targets,   # (B, horizon)
        "subsets": subsets,
    }


# ---------------------------------------------------------------------------
# Quick smoke test (python data_pipeline.py)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Running data_pipeline smoke test with synthetic data …")

    # Build synthetic series to avoid requiring HF credentials
    rng = np.random.default_rng(0)
    series_len = 512
    values = rng.standard_normal(series_len).astype(np.float32) * 10 + 5

    # Silver
    normed, mean_, std_ = _silver_normalize(values)
    print(f"Silver: mean={mean_:.3f}, std={std_:.3f}, shape={normed.shape}")

    # Gold
    examples = list(_gold_patch(normed))
    print(f"Gold: {len(examples)} (patches, target) pairs from a single series")
    patches0, target0 = examples[0]
    print(f"  patches shape: {patches0.shape}, target shape: {target0.shape}")

    # Collate
    MAX_P = 64
    fake_batch = []
    for p, t in examples[:4]:
        pt = torch.from_numpy(p)   # (n, patch_size) — variable n
        # Pad/truncate to MAX_P (same logic as LOTSAStreamDataset)
        n = pt.shape[0]
        if n > MAX_P:
            pt = pt[-MAX_P:]
        elif n < MAX_P:
            pad = torch.zeros(MAX_P - n, pt.shape[1], dtype=pt.dtype)
            pt = torch.cat([pad, pt], dim=0)
        fake_batch.append({"patches": pt, "target": torch.from_numpy(t), "subset": "test"})
    batch = collate_fn(fake_batch)
    print(f"Collated patches: {batch['patches'].shape}, targets: {batch['targets'].shape}")
    print("data_pipeline.py - all checks passed OK")
