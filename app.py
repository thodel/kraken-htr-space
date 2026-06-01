"""
Gradio 6.x front-end for kraken+ training on HF Spaces (A100 80 GB).

Workflow:
  1. Set HF_TOKEN and OUTPUT_REPO as Space secrets.
  2. Configure training params, tick "Auto-start training after prepare" if desired.
  3. Click "Prepare GT" — downloads dataset, writes JPEG + .gt.txt ground truth.
     If auto-start is enabled, training begins automatically when prepare finishes.
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


def _run_pipeline(cmds: list[tuple[list[str], str]]):
    """Run multiple (cmd, label) pairs sequentially in one thread."""
    for cmd, label in cmds:
        _run(cmd, label)
        # stop the pipeline if a step failed
        if _proc and _proc.returncode != 0:
            with open(LOG_FILE, "a") as log:
                log.write(f"\n[pipeline] step '{label}' failed (exit {_proc.returncode}) — aborting.\n")
            break


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


def _train_cmd(epochs, batch_size, lr, resume_path):
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
    return cmd


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_prepare(limit_str, val_ratio, auto_train, epochs, batch_size, lr, resume_path):
    if _proc and _proc.poll() is None:
        return "A process is already running."

    prepare_cmd = [sys.executable, "train.py", "prepare", "--val-ratio", str(val_ratio)]
    try:
        prepare_cmd += ["--limit", str(int(limit_str))]
    except (ValueError, TypeError):
        pass

    if auto_train:
        pipeline = [
            (prepare_cmd, "prepare"),
            (_train_cmd(epochs, batch_size, lr, resume_path), "train"),
        ]
        threading.Thread(target=_run_pipeline, args=(pipeline,), daemon=True).start()
        return "Prepare started — training will begin automatically when done. See log below."
    else:
        threading.Thread(target=_run, args=(prepare_cmd, "prepare"), daemon=True).start()
        return "Prepare started — see log below."


def action_train(epochs, batch_size, lr, resume_path):
    if _proc and _proc.poll() is None:
        return "A process is already running."
    threading.Thread(
        target=_run, args=(_train_cmd(epochs, batch_size, lr, resume_path), "train"), daemon=True
    ).start()
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

    # Training params (shown above prepare so they're set before clicking)
    gr.Markdown("### Training Parameters")
    with gr.Row():
        epochs_in     = gr.Number(label="Epochs",        value=50,   precision=0)
        batch_size_in = gr.Number(label="Batch size",    value=32,   precision=0)
        lr_in         = gr.Number(label="Learning rate", value=2e-4)
    resume_in = gr.Textbox(label="Resume from checkpoint (optional path)", value="")

    # Step 1 — Prepare (+ optional auto-chain)
    gr.Markdown("### Step 1 — Prepare Ground Truth (~30 min)")
    with gr.Row():
        limit_in     = gr.Textbox(label="Sample limit (blank = all ~497K)", value="")
        val_ratio_in = gr.Slider(0.05, 0.2, value=0.1, step=0.05, label="Validation ratio")
    auto_train_in = gr.Checkbox(
        label="Auto-start training immediately after prepare finishes",
        value=True,
    )
    prepare_btn = gr.Button("Prepare GT  →  (Train)", variant="primary")
    prepare_out = gr.Textbox(label="", interactive=False)
    prepare_btn.click(
        action_prepare,
        inputs=[limit_in, val_ratio_in, auto_train_in, epochs_in, batch_size_in, lr_in, resume_in],
        outputs=prepare_out,
    )

    # Step 2 — Train (manual override)
    gr.Markdown("### Step 2 — Train (manual, only needed if auto-start is off)")
    train_btn = gr.Button("Start Training", variant="secondary")
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

    # Live log
    gr.Markdown("### Live Log (last 80 lines)")
    log_box = gr.Textbox(label="", interactive=False, lines=25, max_lines=25, value=_tail_log)

    timer = gr.Timer(value=5)
    timer.tick(fn=_tail_log, outputs=log_box)
    timer.tick(fn=_status,   outputs=status_box)


if __name__ == "__main__":
    demo.launch()
