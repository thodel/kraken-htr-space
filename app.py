"""
Gradio 6.x front-end for kraken+ training on HF Spaces (A100 80 GB).

Full pipeline (auto-chained when "Auto-start" is ticked):
  prepare → save-gt → compile → train → push checkpoints

Individual steps can also be triggered manually.

Space secrets required:
  HF_TOKEN    — write-access HF token
  OUTPUT_REPO — HF model repo for checkpoints, e.g. thodel/kraken-plus-medieval
  GT_REPO     — HF dataset repo for ground-truth storage (default: thodel/kraken-htr-data)
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
    """Run (cmd, label) pairs sequentially; abort on first non-zero exit."""
    for cmd, label in cmds:
        _run(cmd, label)
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
        arrows = sum(1 for f in ("train.arrow", "val.arrow") if (GT_DIR / f).exists())
        return (
            f"✅ GT ready — {s.get('train', 0):,} train / {s.get('val', 0):,} val  |  "
            f"{arrows}/2 arrow file(s)  |  {ckpts} checkpoint(s)"
        )
    return "⚪ Idle — ready to start"


def _py(stage: str, *extra: str) -> list[str]:
    return [sys.executable, "train.py", stage, *extra]


def _train_cmd(epochs, batch_size, lr, resume_path):
    cmd = _py("train",
              "--epochs",     str(int(epochs)),
              "--batch-size", str(int(batch_size)),
              "--lr",         str(lr))
    if resume_path.strip():
        cmd += ["--resume", resume_path.strip()]
    return cmd


# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------

def action_prepare(limit_str, val_ratio, auto_train, epochs, batch_size, lr, resume_path):
    if _proc and _proc.poll() is None:
        return "A process is already running."
    prepare_cmd = _py("prepare", "--val-ratio", str(val_ratio))
    try:
        prepare_cmd += ["--limit", str(int(limit_str))]
    except (ValueError, TypeError):
        pass

    if auto_train:
        pipeline = [
            (prepare_cmd,                    "prepare"),
            (_py("save-gt"),                 "save-gt"),   # persist to Hub immediately
            (_py("compile"),                 "compile"),
            (_train_cmd(epochs, batch_size, lr, resume_path), "train"),
        ]
        threading.Thread(target=_run_pipeline, args=(pipeline,), daemon=True).start()
        return "Pipeline started: prepare → save-gt → compile → train. See log below."
    else:
        threading.Thread(target=_run, args=(prepare_cmd, "prepare"), daemon=True).start()
        return "Prepare started — see log below."


def action_save_gt():
    if _proc and _proc.poll() is None:
        return "A process is already running."
    threading.Thread(target=_run, args=(_py("save-gt"), "save-gt"), daemon=True).start()
    return "Saving ground truth to Hub — see log below."


def action_load_gt():
    if _proc and _proc.poll() is None:
        return "A process is already running."
    threading.Thread(target=_run, args=(_py("load-gt"), "load-gt"), daemon=True).start()
    return "Loading ground truth from Hub — see log below."


def action_compile():
    if _proc and _proc.poll() is None:
        return "A process is already running."
    threading.Thread(target=_run, args=(_py("compile"), "compile"), daemon=True).start()
    return "Compiling Arrow datasets — see log below."


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
    threading.Thread(target=_run, args=(_py("push", "--output-repo", output_repo), "push"), daemon=True).start()
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
        "Dataset: [dh-unibe/image-text_medieval-scripts_xiv-xv-xvi](https://huggingface.co/datasets/dh-unibe/image-text_medieval-scripts_xiv-xv-xvi) "
        "· A100 80 GB · ~32 h total\n\n"
        "**Secrets required:** `HF_TOKEN`, `OUTPUT_REPO`, optionally `GT_REPO` (default: `thodel/kraken-htr-data`)"
    )

    status_box = gr.Textbox(label="Status", interactive=False, value=_status)

    # --- Training params ---
    gr.Markdown("### Training Parameters")
    with gr.Row():
        epochs_in     = gr.Number(label="Epochs",        value=50,   precision=0)
        batch_size_in = gr.Number(label="Batch size",    value=32,   precision=0)
        lr_in         = gr.Number(label="Learning rate", value=2e-4)
    resume_in = gr.Textbox(label="Resume from checkpoint (optional)", value="")

    # --- Step 1: Prepare ---
    gr.Markdown("### Step 1 — Prepare Ground Truth (~10 h streaming)")
    with gr.Row():
        limit_in     = gr.Textbox(label="Sample limit (blank = all ~497K)", value="")
        val_ratio_in = gr.Slider(0.05, 0.2, value=0.1, step=0.05, label="Val ratio")
    auto_train_in = gr.Checkbox(
        label="Auto-run full pipeline after prepare: save-gt → compile → train",
        value=True,
    )
    prepare_btn = gr.Button("▶ Prepare GT", variant="primary")
    prepare_out = gr.Textbox(label="", interactive=False)
    prepare_btn.click(
        action_prepare,
        inputs=[limit_in, val_ratio_in, auto_train_in, epochs_in, batch_size_in, lr_in, resume_in],
        outputs=prepare_out,
    )

    # --- Step 2: Save / Load GT ---
    gr.Markdown("### Step 2 — Save / Restore Ground Truth (Hub ↔ /tmp)")
    gr.Markdown(
        "Ground truth lives in `/tmp` and is lost on Space rebuild. "
        "**Save** pushes it to `GT_REPO` on the Hub. **Load** restores it — "
        "use this after a restart to skip re-preparing."
    )
    with gr.Row():
        save_gt_btn = gr.Button("💾 Save GT → Hub", variant="primary")
        load_gt_btn = gr.Button("📥 Load GT ← Hub")
    gt_out = gr.Textbox(label="", interactive=False)
    save_gt_btn.click(action_save_gt, outputs=gt_out)
    load_gt_btn.click(action_load_gt, outputs=gt_out)

    # --- Step 3: Compile ---
    gr.Markdown("### Step 3 — Compile to Arrow Binary (~20 min)")
    compile_btn = gr.Button("⚙ Compile GT → Arrow")
    compile_out = gr.Textbox(label="", interactive=False)
    compile_btn.click(action_compile, outputs=compile_out)

    # --- Step 4: Train ---
    gr.Markdown("### Step 4 — Train (~21 h)")
    train_btn = gr.Button("🚀 Start Training", variant="secondary")
    train_out = gr.Textbox(label="", interactive=False)
    train_btn.click(action_train, inputs=[epochs_in, batch_size_in, lr_in, resume_in], outputs=train_out)

    # --- Step 5: Push checkpoints ---
    gr.Markdown("### Step 5 — Push Checkpoints to Hub")
    push_btn = gr.Button("⬆ Push Checkpoints")
    push_out = gr.Textbox(label="", interactive=False)
    push_btn.click(action_push, outputs=push_out)

    # --- Stop ---
    with gr.Row():
        stop_btn = gr.Button("⏹ Stop", variant="stop")
        stop_out = gr.Textbox(label="", interactive=False, scale=3)
    stop_btn.click(action_stop, outputs=stop_out)

    # --- Live log ---
    gr.Markdown("### Live Log (last 80 lines)")
    log_box = gr.Textbox(label="", interactive=False, lines=25, max_lines=25, value=_tail_log)

    timer = gr.Timer(value=5)
    timer.tick(fn=_tail_log, outputs=log_box)
    timer.tick(fn=_status,   outputs=status_box)


if __name__ == "__main__":
    demo.launch()
