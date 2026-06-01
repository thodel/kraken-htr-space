"""
Gradio 6.x front-end for kraken+ training on HF Spaces (A100 80 GB).

Workflow:
  1. Set HF_TOKEN and OUTPUT_REPO as Space secrets.
  2. Click "Prepare GT" — streams dataset, writes ~80 GB of PNG + .gt.txt files.
  3. Click "Start Training" — runs ketos train; logs stream live.
  4. Click "Push Checkpoints" — uploads .mlmodel files to your output repo.
"""

import os
import subprocess
import sys
import threading
from pathlib import Path

import gradio as gr

_BASE      = Path("/tmp/kraken_data")
LOG_FILE   = _BASE / "train.log"
MODELS_DIR = _BASE / "models"
GT_DIR     = _BASE / "ground_truth"

_proc: subprocess.Popen | None = None
_stage = "idle"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run(cmd: list[str], label: str):
    global _proc, _stage
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    _stage = label
    with open(LOG_FILE, "a") as log:
        log.write(f"\n{'='*60}\n[{label}] {' '.join(cmd)}\n{'='*60}\n")
        log.flush()
        _proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT, text=True)
    _proc.wait()
    _stage = "idle"


def _tail_log(n: int = 80) -> str:
    if not LOG_FILE.exists():
        return "(no log yet)"
    lines = LOG_FILE.read_text(errors="replace").splitlines()
    return "\n".join(lines[-n:])


def _status() -> str:
    if _proc and _proc.poll() is None:
        return f"🟡 Running: {_stage}"
    stats_file = GT_DIR / "stats.json"
    if stats_file.exists():
        import json
        s = json.loads(stats_file.read_text())
        ckpts = len(list(MODELS_DIR.glob("*.mlmodel"))) if MODELS_DIR.exists() else 0
        return (
            f"✅ GT ready — {s.get('train', 0):,} train / {s.get('val', 0):,} val  |  "
            f"{ckpts} checkpoint(s)"
        )
    return "⚪ Idle — ready to start"


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_prepare(limit_str: str, val_ratio: float):
    if _proc and _proc.poll() is None:
        return "A process is already running."
    cmd = [sys.executable, "train.py", "prepare", "--val-ratio", str(val_ratio)]
    try:
        cmd += ["--limit", str(int(limit_str))]
    except (ValueError, TypeError):
        pass
    threading.Thread(target=_run, args=(cmd, "prepare"), daemon=True).start()
    return "Prepare started — see log below."


def action_train(epochs: int, batch_size: int, lr: float, resume_path: str):
    if _proc and _proc.poll() is None:
        return "A process is already running."
    cmd = [
        sys.executable, "train.py", "train",
        "--epochs",     str(int(epochs)),
        "--batch-size", str(int(batch_size)),
        "--lr",         str(lr),
        "--workers",    "8",
        "--device",     "cuda:0",
    ]
    if resume_path.strip():
        cmd += ["--resume", resume_path.strip()]
    threading.Thread(target=_run, args=(cmd, "train"), daemon=True).start()
    return "Training started — see log below."


def action_push():
    output_repo = os.environ.get("OUTPUT_REPO", "").strip()
    if not output_repo:
        return "Set the OUTPUT_REPO secret first."
    if _proc and _proc.poll() is None:
        return "A process is already running."
    cmd = [sys.executable, "train.py", "push", "--output-repo", output_repo]
    threading.Thread(target=_run, args=(cmd, "push"), daemon=True).start()
    return f"Pushing checkpoints to {output_repo} …"


def action_stop():
    if _proc and _proc.poll() is None:
        _proc.terminate()
        return "Process terminated."
    return "Nothing running."


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Kraken+ HTR Training") as demo:
    gr.Markdown(
        "## Kraken+ Medieval HTR Training\n"
        "Training on [dh-unibe/image-text_medieval-scripts_xiv-xv-xvi](https://huggingface.co/datasets/dh-unibe/image-text_medieval-scripts_xiv-xv-xvi) "
        "· A100 80 GB · ~22 h · ~$55\n\n"
        "**Before starting:** set `HF_TOKEN` and `OUTPUT_REPO` as Space secrets."
    )

    status_box = gr.Textbox(label="Status", interactive=False, value=_status)

    # Step 1 — Prepare
    gr.Markdown("### Step 1 — Prepare Ground Truth (~1.5 h, ~80 GB)")
    with gr.Row():
        limit_in     = gr.Textbox(label="Sample limit (blank = all ~497K)", value="")
        val_ratio_in = gr.Slider(0.05, 0.2, value=0.1, step=0.05, label="Validation ratio")
    prepare_btn = gr.Button("Prepare GT", variant="primary")
    prepare_out = gr.Textbox(label="", interactive=False)
    prepare_btn.click(action_prepare, inputs=[limit_in, val_ratio_in], outputs=prepare_out)

    # Step 2 — Train
    gr.Markdown("### Step 2 — Train (~21 h)")
    with gr.Row():
        epochs_in     = gr.Number(label="Epochs",        value=50,   precision=0)
        batch_size_in = gr.Number(label="Batch size",    value=32,   precision=0)
        lr_in         = gr.Number(label="Learning rate", value=2e-4)
    resume_in = gr.Textbox(label="Resume from checkpoint (optional path)", value="")
    train_btn = gr.Button("Start Training", variant="primary")
    train_out = gr.Textbox(label="", interactive=False)
    train_btn.click(action_train, inputs=[epochs_in, batch_size_in, lr_in, resume_in], outputs=train_out)

    # Step 3 — Push
    gr.Markdown("### Step 3 — Push Checkpoints to Hub")
    push_btn = gr.Button("Push Checkpoints")
    push_out = gr.Textbox(label="", interactive=False)
    push_btn.click(action_push, outputs=push_out)

    # Stop
    with gr.Row():
        stop_btn = gr.Button("Stop", variant="stop")
        stop_out = gr.Textbox(label="", interactive=False, scale=3)
    stop_btn.click(action_stop, outputs=stop_out)

    # Live log — refreshed by a Timer every 5 s (Gradio 6.x API)
    gr.Markdown("### Live Log (last 80 lines)")
    log_box = gr.Textbox(label="", interactive=False, lines=25, max_lines=25, value=_tail_log)

    timer = gr.Timer(value=5)
    timer.tick(fn=_tail_log,  outputs=log_box)
    timer.tick(fn=_status,    outputs=status_box)


if __name__ == "__main__":
    demo.launch()
