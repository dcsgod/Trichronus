"""
train.py — TriChronos-0.1B
AdamW + cosine LR schedule + BF16 autocast training loop.

Hard wall-clock cutoff: 7h10m (25,800 seconds)
  → saves checkpoint, pauses the HF Space, then exits cleanly.

Usage
-----
  python train.py [--resume] [--steps N] [--batch-size B]

Environment variables (set by HF Spaces)
-----------------------------------------
  SPACE_ID        : "username/space-name"   (required for pause_space)
  HF_TOKEN        : Hugging Face write token
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data_pipeline import LOTSAStreamDataset, collate_fn, FORECAST_HORIZON, PATCH_SIZE
from model import TriChronos, QUANTILE_LEVELS


# ---------------------------------------------------------------------------
# Hyper-parameters
# ---------------------------------------------------------------------------

WALL_CLOCK_LIMIT: float = 7 * 3600 + 10 * 60   # 7 h 10 m in seconds
# Use /data/checkpoints when running in an HF Space (bucket mounted at /data)
# Fall back to local ./checkpoints for local dev runs.
_IN_SPACE = bool(os.environ.get("CHECKPOINT_BUCKET_ID"))
CHECKPOINT_DIR: Path = Path("/data/checkpoints") if _IN_SPACE else Path("checkpoints")
LOG_EVERY: int = 100                             # log every N steps
SAVE_EVERY: int = 1_000                          # checkpoint every N steps

DEFAULT_LR: float = 3e-4
DEFAULT_WEIGHT_DECAY: float = 1e-2
DEFAULT_WARMUP_STEPS: int = 2_000
DEFAULT_MAX_STEPS: int = 500_000
DEFAULT_BATCH_SIZE: int = 32
DEFAULT_GRAD_CLIP: float = 1.0


# ---------------------------------------------------------------------------
# Quantile (pinball) loss
# ---------------------------------------------------------------------------

def quantile_loss(
    preds: torch.Tensor,      # (B, horizon, n_quantiles)
    targets: torch.Tensor,    # (B, horizon)
    quantile_levels: list,
) -> torch.Tensor:
    """
    Pinball / quantile loss averaged over all quantiles, horizons, and batch.

    L(q, y, ŷ) = q·max(y-ŷ, 0) + (1-q)·max(ŷ-y, 0)
    """
    tau = torch.tensor(quantile_levels, dtype=preds.dtype, device=preds.device)
    # targets: (B, horizon) → (B, horizon, 1) for broadcasting
    y = targets.unsqueeze(-1)
    errors = y - preds                           # (B, horizon, n_quantiles)
    loss = torch.max(tau * errors, (tau - 1.0) * errors)
    return loss.mean()


# ---------------------------------------------------------------------------
# Learning rate schedule
# ---------------------------------------------------------------------------

def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float) -> float:
    """Linear warmup then cosine decay to 10% of peak LR."""
    import math
    if step < warmup_steps:
        return max_lr * step / max_steps
    if step >= max_steps:
        return max_lr * 0.1
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return max_lr * 0.1 + 0.5 * (max_lr - max_lr * 0.1) * (1 + math.cos(math.pi * progress))


# ---------------------------------------------------------------------------
# Checkpoint helpers
# ---------------------------------------------------------------------------

def save_checkpoint(
    model: TriChronos,
    optimizer: torch.optim.Optimizer,
    step: int,
    loss: float,
    directory: Path,
):
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), directory / "model_state.pt")
    torch.save(optimizer.state_dict(), directory / "optimizer_state.pt")
    (directory / "step.txt").write_text(f"{step}\n")
    (directory / "loss.txt").write_text(f"{loss:.6f}\n")
    print(f"[step {step}] Checkpoint saved to {directory}", flush=True)


def load_checkpoint(
    model: TriChronos,
    optimizer: torch.optim.Optimizer,
    directory: Path,
) -> int:
    """Load checkpoint; return the step to resume from (0 if none found)."""
    if not (directory / "model_state.pt").exists():
        print("No checkpoint found — starting from scratch.", flush=True)
        return 0
    model.load_state_dict(torch.load(directory / "model_state.pt", map_location="cpu"))
    optimizer.load_state_dict(torch.load(directory / "optimizer_state.pt", map_location="cpu"))
    step = int((directory / "step.txt").read_text().strip())
    print(f"Resumed from checkpoint at step {step}", flush=True)
    return step


# ---------------------------------------------------------------------------
# HF Space auto-pause
# ---------------------------------------------------------------------------

def pause_hf_space():
    """
    Pause the Hugging Face Space this script is running in.
    No-op if SPACE_ID is not set (e.g., local runs).
    """
    space_id = os.environ.get("SPACE_ID", "")
    hf_token = os.environ.get("HF_TOKEN", "")
    if not space_id:
        print("[pause_hf_space] SPACE_ID not set — skipping Space pause.", flush=True)
        return
    try:
        from huggingface_hub import pause_space
        pause_space(space_id, token=hf_token or None)
        print(f"[pause_hf_space] Space '{space_id}' paused successfully.", flush=True)
    except Exception as exc:
        print(f"[pause_hf_space] Failed to pause space: {exc}", flush=True)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train(args: argparse.Namespace):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}", flush=True)

    # ---- Model ----
    model = TriChronos(
        patch_size=PATCH_SIZE,
        horizon=FORECAST_HORIZON,
    ).to(device)
    print(f"Parameters: {model.count_params():,}", flush=True)

    # ---- Optimizer ----
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=DEFAULT_LR,
        weight_decay=DEFAULT_WEIGHT_DECAY,
        betas=(0.9, 0.95),
    )

    # ---- Resume ----
    start_step = 0
    if args.resume:
        start_step = load_checkpoint(model, optimizer, CHECKPOINT_DIR)

    # ---- Dataset ----
    dataset = LOTSAStreamDataset(
        subsets=None,           # stream all LOTSA subsets
        split="train",
        patch_size=PATCH_SIZE,
        horizon=FORECAST_HORIZON,
        max_patches=64,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=(device.type == "cuda"),
    )

    # ---- AMP scaler (for BF16 we don't need GradScaler, but keep for FP16 compat) ----
    use_bf16 = device.type == "cuda" and torch.cuda.is_bf16_supported()
    amp_dtype = torch.bfloat16 if use_bf16 else torch.float16
    scaler = torch.cuda.amp.GradScaler(enabled=(amp_dtype == torch.float16))

    # ---- Training state ----
    step = start_step
    max_steps = args.steps
    t_start = time.time()
    running_loss = 0.0
    best_loss = float("inf")

    # ---- SIGTERM handler (HF Spaces sends SIGTERM on preemption) ----
    def _graceful_exit(signum, frame):
        print(f"\n[SIGTERM] Saving checkpoint at step {step} …", flush=True)
        save_checkpoint(model, optimizer, step, running_loss, CHECKPOINT_DIR)
        pause_hf_space()
        sys.exit(0)

    signal.signal(signal.SIGTERM, _graceful_exit)

    # ---- Main loop ----
    model.train()
    print("Training started …", flush=True)

    for batch in loader:
        if step >= max_steps:
            print(f"Reached max_steps={max_steps}. Stopping.", flush=True)
            break

        # --- Wall-clock cutoff ---
        elapsed = time.time() - t_start
        if elapsed >= WALL_CLOCK_LIMIT:
            print(
                f"\n⏰  Wall-clock limit reached ({elapsed/3600:.2f} h). "
                "Saving checkpoint and pausing Space …",
                flush=True,
            )
            save_checkpoint(model, optimizer, step, running_loss, CHECKPOINT_DIR)
            pause_hf_space()
            sys.exit(0)

        # --- LR update ---
        lr = get_lr(step, DEFAULT_WARMUP_STEPS, max_steps, DEFAULT_LR)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # --- Forward pass ---
        patches = batch["patches"].to(device, non_blocking=True)    # (B, n_patches, patch_size)
        targets = batch["targets"].to(device, non_blocking=True)    # (B, horizon)

        optimizer.zero_grad(set_to_none=True)

        with torch.autocast(device_type=device.type, dtype=amp_dtype):
            preds = model(patches)                                  # (B, horizon, n_quantiles)
            loss = quantile_loss(preds, targets, model.quantile_levels)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), DEFAULT_GRAD_CLIP)
        scaler.step(optimizer)
        scaler.update()

        running_loss = loss.item()
        step += 1

        # --- Logging ---
        if step % LOG_EVERY == 0:
            elapsed_h = (time.time() - t_start) / 3600
            budget_pct = elapsed / WALL_CLOCK_LIMIT * 100
            print(
                f"step={step:7d}  loss={running_loss:.4f}  lr={lr:.2e}"
                f"  elapsed={elapsed_h:.2f}h  budget={budget_pct:.1f}%",
                flush=True,
            )

        # --- Periodic checkpoint ---
        if step % SAVE_EVERY == 0:
            if running_loss < best_loss:
                best_loss = running_loss
            save_checkpoint(model, optimizer, step, running_loss, CHECKPOINT_DIR)

    # ---- End of training ----
    save_checkpoint(model, optimizer, step, running_loss, CHECKPOINT_DIR)
    print(f"Training complete at step {step}. Best loss: {best_loss:.4f}", flush=True)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train TriChronos-0.1B")
    p.add_argument("--resume", action="store_true", help="Resume from checkpoint")
    p.add_argument("--steps", type=int, default=DEFAULT_MAX_STEPS, help="Max training steps")
    p.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="Batch size")
    return p.parse_args()


if __name__ == "__main__":
    train(parse_args())
