"""
spaces/train/app.py — TriChronos Training Monitor
Gradio UI that:
  • Launches train.py as a managed subprocess
  • Streams live logs to the browser
  • Shows elapsed time, budget consumption, and latest loss
  • Provides Start / Stop training buttons
  • Auto-resumes if a checkpoint exists on startup

Runs inside the Docker training Space on HF (port 7860).
"""

from __future__ import annotations

import os
import subprocess
import threading
import time
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

WALL_CLOCK_LIMIT = 7 * 3600 + 10 * 60   # 7 h 10 m
HOURLY_COST = 1.80                        # USD/hr for 1x L40S (actual Space hardware)
TOTAL_BUDGET = 15.00                      # USD hard cap
LOG_TAIL_LINES = 120                      # lines shown in UI

CHECKPOINT_DIR = Path("checkpoints")
LOG_FILE = Path("train.log")

# ---------------------------------------------------------------------------
# Global training state (module-level, protected by _lock)
# ---------------------------------------------------------------------------

_proc: subprocess.Popen | None = None
_lock = threading.Lock()
_train_start: float | None = None
_log_file_handle = None


# ---------------------------------------------------------------------------
# Subprocess management
# ---------------------------------------------------------------------------

def _is_running() -> bool:
    with _lock:
        return _proc is not None and _proc.poll() is None


def _log_writer(proc: subprocess.Popen):
    """Thread: reads stdout from train.py, writes to log file."""
    with open(LOG_FILE, "a", buffering=1) as f:
        for line in proc.stdout:
            f.write(line)
            f.flush()


def start_training() -> tuple[str, str]:
    """
    Start train.py as a subprocess.
    Returns (status_message, button_label).
    """
    global _proc, _train_start

    if _is_running():
        return "⚠️ Training is already running.", gr.update()

    CHECKPOINT_DIR.mkdir(exist_ok=True)
    LOG_FILE.parent.mkdir(exist_ok=True)

    # Auto-resume if checkpoint exists
    resume = (CHECKPOINT_DIR / "model_state.pt").exists()
    cmd = ["python", "train.py", "--batch-size", "32"]
    if resume:
        cmd.append("--resume")

    with _lock:
        _train_start = time.time()
        _proc = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env={**os.environ},
        )

    # Background thread writes logs to file
    threading.Thread(target=_log_writer, args=(_proc,), daemon=True).start()

    mode = "Resuming from checkpoint" if resume else "Starting fresh"
    return f"✅ {mode} — PID {_proc.pid}", gr.update()


def stop_training() -> tuple[str, str]:
    """Send SIGTERM to train.py (triggers graceful checkpoint + Space pause)."""
    global _proc

    with _lock:
        proc = _proc

    if proc is None or proc.poll() is not None:
        return "ℹ️ No training process running.", gr.update()

    proc.terminate()
    try:
        proc.wait(timeout=30)
    except subprocess.TimeoutExpired:
        proc.kill()

    return "🛑 Training stopped (checkpoint saved).", gr.update()


# ---------------------------------------------------------------------------
# Status helpers
# ---------------------------------------------------------------------------

def _read_log_tail() -> str:
    if not LOG_FILE.exists():
        return "(no log yet — start training to begin)"
    try:
        with open(LOG_FILE, "r") as f:
            lines = f.readlines()
        return "".join(lines[-LOG_TAIL_LINES:])
    except Exception as exc:
        return f"(error reading log: {exc})"


def _read_latest_loss() -> str:
    loss_file = CHECKPOINT_DIR / "loss.txt"
    if loss_file.exists():
        try:
            return loss_file.read_text().strip()
        except Exception:
            pass
    return "N/A"


def _read_latest_step() -> str:
    step_file = CHECKPOINT_DIR / "step.txt"
    if step_file.exists():
        try:
            return f"{int(step_file.read_text().strip()):,}"
        except Exception:
            pass
    return "0"


def get_status() -> tuple[str, str, str, str, str, str]:
    """
    Returns:
        status_icon, status_text, elapsed_str, budget_str, loss_str, step_str
    """
    running = _is_running()
    status_icon = "🟢 Running" if running else "⚫ Idle"

    # Time + budget
    if _train_start is not None:
        elapsed_s = time.time() - _train_start
        elapsed_h = elapsed_s / 3600
        pct = min(elapsed_s / WALL_CLOCK_LIMIT * 100, 100)
        cost = elapsed_h * HOURLY_COST
        elapsed_str = f"{elapsed_h:.2f} h  ({pct:.1f}% of budget)"
        budget_str = f"${cost:.2f} spent  /  ${TOTAL_BUDGET:.2f} total"
    else:
        elapsed_str = "—"
        budget_str = f"$0.00  /  ${TOTAL_BUDGET:.2f}"

    loss_str = _read_latest_loss()
    step_str = _read_latest_step()
    log_text = _read_log_tail()

    return status_icon, elapsed_str, budget_str, loss_str, step_str, log_text


# ---------------------------------------------------------------------------
# Auto-start on Space boot (if HF_TOKEN is set, indicating a real GPU Space)
# ---------------------------------------------------------------------------

def _maybe_autostart():
    """If we're in a real HF Space (SPACE_ID set) and not already running, start training."""
    if os.environ.get("TRICHRONOS_SPACE_ID") and not _is_running():
        time.sleep(3)   # Give Gradio time to fully start
        start_training()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CUSTOM_CSS = """
#log-box textarea {
    font-family: 'Courier New', monospace;
    font-size: 12px;
    background: #0d1117;
    color: #c9d1d9;
}
.metric-card {
    background: linear-gradient(135deg, #1a1f2e, #252b3b);
    border: 1px solid #30363d;
    border-radius: 8px;
    padding: 12px;
    text-align: center;
}
"""

with gr.Blocks(
    title="TriChronos Training Monitor",
    theme=gr.themes.Base(
        primary_hue=gr.themes.colors.indigo,
        neutral_hue=gr.themes.colors.slate,
    ),
    css=CUSTOM_CSS,
) as demo:

    gr.Markdown("""
# 🧠 TriChronos-0.1B — Training Monitor
**100M-parameter ternary-quantised time-series forecasting model**  
Dataset: [Salesforce/lotsa_data](https://huggingface.co/datasets/Salesforce/lotsa_data) · 
Model: [iravikr/trichronos-0.1b](https://huggingface.co/iravikr/trichronos-0.1b)
""")

    with gr.Row():
        with gr.Column(scale=1):
            status_icon = gr.Textbox(
                label="Status", value="⚫ Idle", interactive=False, elem_id="status"
            )
        with gr.Column(scale=2):
            elapsed_box = gr.Textbox(label="Elapsed / Budget %", value="—", interactive=False)
        with gr.Column(scale=2):
            budget_box = gr.Textbox(label="Cost", value="$0.00 / $18.00", interactive=False)
        with gr.Column(scale=1):
            loss_box = gr.Textbox(label="Latest Loss", value="N/A", interactive=False)
        with gr.Column(scale=1):
            step_box = gr.Textbox(label="Step", value="0", interactive=False)

    with gr.Row():
        start_btn = gr.Button("▶ Start Training", variant="primary", size="lg")
        stop_btn = gr.Button("⏹ Stop Training", variant="stop", size="lg")

    msg_box = gr.Textbox(label="Last action", value="", interactive=False)

    gr.Markdown("### Live Training Log")
    log_box = gr.Textbox(
        label="stdout",
        value="(no log yet)",
        lines=30,
        max_lines=30,
        interactive=False,
        elem_id="log-box",
    )

    # ---- Event handlers ----

    def on_start():
        msg, _ = start_training()
        return msg

    def on_stop():
        msg, _ = stop_training()
        return msg

    def refresh():
        icon, elapsed, budget, loss, step, log = get_status()
        return icon, elapsed, budget, loss, step, log

    start_btn.click(fn=on_start, outputs=[msg_box])
    stop_btn.click(fn=on_stop, outputs=[msg_box])

    # Auto-refresh every 3 seconds
    timer = gr.Timer(value=3)
    timer.tick(
        fn=refresh,
        outputs=[status_icon, elapsed_box, budget_box, loss_box, step_box, log_box],
    )

    # Initial load
    demo.load(
        fn=refresh,
        outputs=[status_icon, elapsed_box, budget_box, loss_box, step_box, log_box],
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Auto-start training in background when Space boots
    threading.Thread(target=_maybe_autostart, daemon=True).start()

    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        show_error=True,
    )
