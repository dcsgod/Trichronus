"""
spaces/eval/app.py — TriChronos Evaluation UI
Gradio app that:
  • Accepts a model_state.pt checkpoint (file upload or HF model repo ID)
  • Runs evaluate.py logic inline (no subprocess) for clean streaming
  • Streams MASE results row-by-row as each Monash dataset completes
  • Shows aggregate MASE at the end

Runs on CPU (free tier). No GPU required.
"""

from __future__ import annotations

import io
import os
import sys
import tempfile
from pathlib import Path
from typing import Generator

import gradio as gr
import numpy as np
import torch

# ---------------------------------------------------------------------------
# The evaluate.py logic is inlined here so we don't need subprocess.
# We import from the project source files which are copied into the Space.
# ---------------------------------------------------------------------------

def _load_model(checkpoint_path: str) -> "TriChronos":
    from model import TriChronos
    from data_pipeline import PATCH_SIZE, FORECAST_HORIZON

    model = TriChronos(patch_size=PATCH_SIZE, horizon=FORECAST_HORIZON)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def _load_model_from_hub(repo_id: str) -> "TriChronos":
    from huggingface_hub import hf_hub_download
    ckpt_path = hf_hub_download(repo_id=repo_id, filename="model_state.pt")
    return _load_model(ckpt_path)


# ---------------------------------------------------------------------------
# Monash datasets — single source of truth in evaluate.py
# ---------------------------------------------------------------------------

from evaluate import MONASH_DATASETS


def _run_eval(
    model,
    max_series: int,
) -> Generator[tuple[list[list], str], None, None]:
    """
    Generator: yields (rows, status) after each dataset finishes.
    rows = list of [dataset, MASE, N] for the results table.
    """
    from evaluate import evaluate_dataset
    from data_pipeline import PATCH_SIZE, FORECAST_HORIZON

    device = torch.device("cpu")
    rows: list[list] = []
    all_mase: list[float] = []

    for ds_name, subset, period in MONASH_DATASETS:
        label = subset or ds_name
        yield rows, f"⏳ Evaluating **{label}** …"

        mase, n = evaluate_dataset(model, ds_name, subset, period, device, max_series)

        if not np.isnan(mase):
            all_mase.append(mase)
            rows.append([label, f"{mase:.4f}", str(n)])
        else:
            rows.append([label, "N/A (skipped)", "0"])

        yield rows, f"✅ {label}: MASE={mase:.4f}" if not np.isnan(mase) else f"⚠️ {label}: skipped"

    # Final aggregate
    if all_mase:
        agg = float(np.mean(all_mase))
        rows.append(["**AGGREGATE**", f"**{agg:.4f}**", f"**{len(all_mase)} datasets**"])
        yield rows, f"✅ Done! Aggregate MASE = **{agg:.4f}** across {len(all_mase)} datasets."
    else:
        yield rows, "⚠️ No datasets were successfully evaluated."


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
#results-table table { font-size: 14px; }
#results-table tr:last-child { font-weight: bold; background: #1a2744; }
"""

with gr.Blocks(
    title="TriChronos Evaluation",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.teal,
        neutral_hue=gr.themes.colors.slate,
    ),
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown("""
# 📊 TriChronos-0.1B — Evaluation
**Zero-shot MASE against the Monash Time Series Forecasting benchmark**  
Runs on CPU — no GPU needed.
""")

    with gr.Tab("Upload Checkpoint"):
        ckpt_upload = gr.File(
            label="Upload model_state.pt",
            file_types=[".pt", ".pth"],
        )
        upload_eval_btn = gr.Button("▶ Run Evaluation (uploaded checkpoint)", variant="primary")

    with gr.Tab("Load from HF Hub"):
        hub_repo = gr.Textbox(
            label="HF Model Repo ID",
            placeholder="iravikr/trichronos-0.1b",
            value="iravikr/trichronos-0.1b",
        )
        hub_eval_btn = gr.Button("▶ Run Evaluation (from Hub)", variant="primary")

    with gr.Tab("Latest training checkpoint (/data)"):
        gr.Markdown(
            "Evaluates the most recent `model_state.pt` synced by the training "
            "Space to the mounted checkpoint bucket (`/data`). No upload needed."
        )
        data_eval_btn = gr.Button("▶ Run Evaluation (/data checkpoint)", variant="primary")

    max_series = gr.Slider(
        minimum=10,
        maximum=500,
        value=50,
        step=10,
        label="Max series per dataset",
        info="Lower = faster eval. Monash M4-monthly has 48,000 series; 50 gives a quick estimate.",
    )

    status_box = gr.Markdown("Ready. Upload a checkpoint or enter a Hub repo ID, then click Run.")

    results_table = gr.Dataframe(
        headers=["Dataset", "MASE", "N series"],
        datatype=["str", "str", "str"],
        row_count=(len(MONASH_DATASETS) + 1, "fixed"),
        col_count=(3, "fixed"),
        label="Evaluation Results",
        elem_id="results-table",
        interactive=False,
    )

    # ---- Handlers ----

    def eval_from_upload(file_obj, max_s: int):
        if file_obj is None:
            yield [], "⚠️ Please upload a model_state.pt file first."
            return
        try:
            model = _load_model(file_obj.name)
        except Exception as exc:
            yield [], f"❌ Failed to load checkpoint: {exc}"
            return
        for rows, status in _run_eval(model, int(max_s)):
            yield rows, status

    def eval_from_hub(repo_id: str, max_s: int):
        if not repo_id.strip():
            yield [], "⚠️ Please enter a HF repo ID."
            return
        try:
            model = _load_model_from_hub(repo_id.strip())
        except Exception as exc:
            yield [], f"❌ Failed to load from Hub '{repo_id}': {exc}"
            return
        for rows, status in _run_eval(model, int(max_s)):
            yield rows, status

    CHECKPOINT_BUCKET_ID = os.environ.get(
        "CHECKPOINT_BUCKET_ID", "iravikr/trichronos-checkpoints"
    )

    def eval_from_data(max_s: int):
        # Prefer the mounted /data snapshot; fall back to downloading the
        # latest checkpoint straight from the dataset repo. The dataset-volume
        # mount is a point-in-time snapshot and does NOT pick up checkpoints
        # the training Space pushes after this Space booted, so the download
        # path is what actually gets the freshest checkpoint.
        ckpt_path = None
        step_note = ""
        mounted = Path("/data/model_state.pt")
        if mounted.exists():
            ckpt_path = str(mounted)
            step_file = Path("/data/step.txt")
            if step_file.exists():
                step_note = f" (step {step_file.read_text().strip()}, from /data)"
        else:
            yield [], f"⬇️ /data empty — pulling latest checkpoint from `{CHECKPOINT_BUCKET_ID}` …"
            try:
                from huggingface_hub import hf_hub_download
                ckpt_path = hf_hub_download(
                    repo_id=CHECKPOINT_BUCKET_ID,
                    filename="model_state.pt",
                    repo_type="dataset",
                )
                try:
                    step_txt = hf_hub_download(
                        repo_id=CHECKPOINT_BUCKET_ID,
                        filename="step.txt",
                        repo_type="dataset",
                    )
                    step_note = f" (step {Path(step_txt).read_text().strip()}, from dataset repo)"
                except Exception:
                    step_note = " (from dataset repo)"
            except Exception as exc:
                yield [], (
                    f"⚠️ No checkpoint found. `/data` is empty and downloading "
                    f"`model_state.pt` from `{CHECKPOINT_BUCKET_ID}` failed: {exc}"
                )
                return

        try:
            model = _load_model(ckpt_path)
        except Exception as exc:
            yield [], f"❌ Failed to load checkpoint: {exc}"
            return
        yield [], f"✅ Loaded latest checkpoint{step_note}. Evaluating …"
        for rows, status in _run_eval(model, int(max_s)):
            yield rows, status

    upload_eval_btn.click(
        fn=eval_from_upload,
        inputs=[ckpt_upload, max_series],
        outputs=[results_table, status_box],
    )

    hub_eval_btn.click(
        fn=eval_from_hub,
        inputs=[hub_repo, max_series],
        outputs=[results_table, status_box],
    )

    data_eval_btn.click(
        fn=eval_from_data,
        inputs=[max_series],
        outputs=[results_table, status_box],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
