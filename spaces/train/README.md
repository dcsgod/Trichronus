---
title: TriChronos Train
emoji: 🧠
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 5.34.2
python_version: "3.11"
app_file: app.py
pinned: true
hardware: a100-large
storage: small
---

# TriChronos-0.1B — Training Space

This Space trains TriChronos-0.1B (100M-param ternary time-series forecasting model)
on [Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data).

## Setup checklist (one-time, manual steps on HF)

1. **Settings → Hardware** → upgrade to `Nvidia A100 - large` (~$2.50/hr)
2. **Settings → Persistent storage** → enable (so checkpoints survive restarts)
3. **Settings → Variables and Secrets** → add:
   - `HF_TOKEN` = your HF write token
   - `SPACE_ID` = `iravikr/trichronos-train`
4. Restart the Space — training will start automatically when the app loads.

## What the UI shows

- **Live training log** — streams stdout from `train.py`
- **Loss tracker** — latest checkpoint loss from `checkpoints/loss.txt`
- **Budget meter** — elapsed wall-clock time vs. 7h10m hard limit
- **Start / Stop** buttons
