---
title: TriChronos Eval
emoji: 📊
colorFrom: teal
colorTo: cyan
sdk: gradio
sdk_version: 5.34.2
python_version: "3.11"
app_file: app.py
pinned: false
hardware: cpu-upgrade
---

# TriChronos-0.1B — Evaluation Space

Upload a TriChronos checkpoint and run zero-shot MASE evaluation against the
[Monash Time Series Forecasting](https://forecastingdata.org/) benchmark.

Runs fully on CPU — no GPU required.

## Usage

1. Upload a `model_state.pt` checkpoint (from the training Space)
2. Set the max series per dataset (default 50 for fast eval)
3. Click **Run Evaluation**
4. Watch MASE results stream in per dataset

## What is MASE?

Mean Absolute Scaled Error vs. a seasonal-naive baseline.
Lower is better. MASE < 1.0 means the model beats the naive forecast.
