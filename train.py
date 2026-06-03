#!/usr/bin/env python3
"""
Core training logic for kraken+ on dh-unibe/image-text_medieval-scripts_xiv-xv-xvi.
Called by app.py via subprocess so the Gradio UI stays responsive.

Stages:
  prepare   — stream HF dataset → JPEG + .gt.txt ground-truth pairs (in /tmp)
  save-gt   — push /tmp ground-truth folder to HF Hub dataset (persistent)
  load-gt   — restore ground-truth from HF Hub dataset back to /tmp
  compile   — convert JPEG + .gt.txt → Arrow binary files for ketos
  train     — invoke ketos train -f binary
  push      — upload .mlmodel checkpoints to HF Hub model repo
"""

import argparse
import io
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Optional

from lxml import etree
from PIL import Image
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DATASET_ID = "dh-unibe/image-text_medieval-scripts_xiv-xv-xvi"

VGSL_SPEC = (
    "[256,64,0,1 "
    "Cr4,2,8,4,2 "
    "Cr4,2,32,1,1 "
    "Mp4,2,4,2 "
    "Cr3,3,64,1,1 "
    "Mp1,2,1,2 "
    "S1(1x0)1,3 "
    "Lbx256 Do0.5 "
    "Lbx256 Do0.5 "
    "Lbx256 Do0.5 "
    "Cr255,1,85,1,1]"
)

_BASE       = Path(os.environ.get("KRAKEN_DATA_DIR", "/tmp/kraken_data"))
GT_DIR      = _BASE / "ground_truth"
MODELS_DIR  = _BASE / "models"
LOG_FILE    = _BASE / "train.log"

# ---------------------------------------------------------------------------
# PageXML helpers
# ---------------------------------------------------------------------------

def _extract_text(xml_bytes: bytes) -> Optional[str]:
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return None
    nodes = root.findall(".//{*}TextLine/{*}TextEquiv/{*}Unicode")
    if not nodes:
        return None
    texts = [n.text.strip() for n in nodes if n.text and n.text.strip()]
    return " ".join(texts) if texts else None


# ---------------------------------------------------------------------------
# Stage 1 – prepare ground truth
# ---------------------------------------------------------------------------

def _process_one(args: tuple) -> tuple[int, Optional[str]]:
    """Process a single (index, sample) into JPEG + .gt.txt. Thread-safe."""
    idx, sample, img_dir = args
    xml_raw = sample.get("xml_content") or ""
    xml_bytes = xml_raw.encode() if isinstance(xml_raw, str) else xml_raw
    try:
        root = etree.fromstring(xml_bytes)
        nodes = root.findall(".//{*}TextLine/{*}TextEquiv/{*}Unicode")
        texts = [n.text.strip() for n in nodes if n.text and n.text.strip()]
        text = " ".join(texts) if texts else None
    except Exception:
        text = None
    if not text:
        return idx, None

    img_obj = sample["image"]
    try:
        if isinstance(img_obj, dict):
            img = Image.open(io.BytesIO(img_obj["bytes"])).convert("RGB")
        elif isinstance(img_obj, bytes):
            img = Image.open(io.BytesIO(img_obj)).convert("RGB")
        else:
            img = img_obj.convert("RGB")
    except Exception:
        return idx, None

    stem = f"{idx:07d}"
    img_path = Path(img_dir) / f"{stem}.jpg"
    img.save(img_path, "JPEG", quality=95)
    (Path(img_dir) / f"{stem}.gt.txt").write_text(text, encoding="utf-8")
    return idx, str(img_path)


def prepare(limit: Optional[int], val_ratio: float):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datasets import load_dataset

    img_dir = GT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    # Streaming avoids downloading 1 TB to disk (which would fill the A100's 1 TB drive).
    # ThreadPoolExecutor parallelises the CPU-bound JPEG encode across 8 threads,
    # giving ~800–1600 samples/s vs 0.6 samples/s with sequential streaming.
    N_WORKERS = 8
    BATCH     = 128   # samples buffered before submitting to the pool

    print(f"[prepare] Streaming {DATASET_ID} with {N_WORKERS} worker threads …", flush=True)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    manifest_train, manifest_val = [], []
    skipped  = 0
    total    = 0
    img_dir_s = str(img_dir)

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        pbar = tqdm(total=limit or 497_400, desc="prepare", file=sys.stdout)
        buf: list[tuple] = []

        def flush_buf():
            nonlocal skipped, total
            futures = {pool.submit(_process_one, item): item[0] for item in buf}
            for fut in as_completed(futures):
                i, path = fut.result()
                pbar.update(1)
                total += 1
                if path is None:
                    skipped += 1
                else:
                    if i % round(1 / val_ratio) == 0:
                        manifest_val.append(path)
                    else:
                        manifest_train.append(path)
            buf.clear()

        for i, sample in enumerate(ds):
            if limit and i >= limit:
                break
            buf.append((i, sample, img_dir_s))
            if len(buf) >= BATCH:
                flush_buf()

        if buf:
            flush_buf()
        pbar.close()

    (GT_DIR / "train.txt").write_text("\n".join(manifest_train), encoding="utf-8")
    (GT_DIR / "val.txt").write_text("\n".join(manifest_val), encoding="utf-8")

    stats = {
        "total": total,
        "train": len(manifest_train),
        "val": len(manifest_val),
        "skipped": skipped,
    }
    (GT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[prepare] done — {json.dumps(stats)}", flush=True)


# ---------------------------------------------------------------------------
# Stage 2 – compile jpg+gt.txt → Arrow binary dataset
# ---------------------------------------------------------------------------

def compile_gt():
    """Convert .jpg + .gt.txt manifest pairs to Arrow binary files for ketos train -f binary."""
    import pyarrow as pa

    for split in ("train", "val"):
        manifest = GT_DIR / f"{split}.txt"
        if not manifest.exists():
            sys.exit(f"[compile] ERROR: {manifest} not found — run prepare first")

        output = GT_DIR / f"{split}.arrow"
        paths  = [p for p in manifest.read_text().splitlines() if p.strip()]

        schema = pa.schema([
            pa.field("image", pa.binary()),
            pa.field("text",  pa.utf8()),
        ])

        BATCH = 500
        imgs, txts = [], []

        print(f"[compile] {split}: {len(paths):,} samples → {output}", flush=True)
        with pa.ipc.new_file(output, schema) as writer:
            for i, img_path_str in enumerate(tqdm(paths, desc=f"compile/{split}", file=sys.stdout)):
                img_p = Path(img_path_str)
                gt_p  = img_p.with_suffix(".gt.txt")
                if not img_p.exists() or not gt_p.exists():
                    continue
                imgs.append(img_p.read_bytes())
                txts.append(gt_p.read_text(encoding="utf-8").strip())
                if len(imgs) >= BATCH:
                    writer.write_batch(pa.record_batch(
                        [pa.array(imgs, pa.binary()), pa.array(txts, pa.utf8())], schema=schema
                    ))
                    imgs, txts = [], []
            if imgs:
                writer.write_batch(pa.record_batch(
                    [pa.array(imgs, pa.binary()), pa.array(txts, pa.utf8())], schema=schema
                ))
        print(f"[compile] {split} done → {output}", flush=True)


# ---------------------------------------------------------------------------
# Stage 3 – train
# ---------------------------------------------------------------------------

def train(
    epochs: int,
    batch_size: int,
    lr: float,
    resume: Optional[str],
):
    """Invoke ketos train with correct kraken 7.x flags."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_manifest = GT_DIR / "train.txt"
    val_manifest   = GT_DIR / "val.txt"
    if not train_manifest.exists():
        sys.exit("[train] ERROR: ground truth not found — run prepare first")

    # kraken 7.x ketos train flags (verified against `ketos train --help`):
    #   -s  VGSL spec
    #   -f  format: 'path' = image + .gt.txt sidecar pairs
    #   -t  manifest file listing training image paths
    #   -e  manifest file listing evaluation image paths
    #   -o  output directory for checkpoints
    #   -N  number of epochs
    #   -B  batch size
    #   -r  learning rate
    #   --device and --workers do NOT exist in kraken 7.x (Lightning auto-detects GPU)
    # ketos train 7.x does NOT accept '-f path' at runtime (bug: listed in --help
    # but rejected by validation). Use '-f binary' with pre-compiled Arrow files.
    train_arrow = GT_DIR / "train.arrow"
    val_arrow   = GT_DIR / "val.arrow"
    if not train_arrow.exists():
        sys.exit("[train] ERROR: train.arrow not found — run compile first")

    cmd = [
        "ketos", "train",
        "-f", "binary",
        "-s", VGSL_SPEC,
        "-e", str(val_arrow),
        "-o", str(MODELS_DIR / "kraken_plus"),
        "-N", str(epochs),
        "-B", str(batch_size),
        "-r", str(lr),
        "--optimizer",       "Adam",
        "--schedule",        "reduceonplateau",
        "--lag",             "5",
        "--min-delta",       "0.005",
        str(train_arrow),    # positional GROUND_TRUTH
    ]
    if resume:
        cmd += ["--resume", resume]

    print("[train] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Save / load ground truth to/from HF Hub
# ---------------------------------------------------------------------------

GT_REPO_DEFAULT = "thodel/kraken-htr-data"   # private dataset repo for GT storage


def save_gt():
    """
    Push the entire ground_truth folder (JPEG + .gt.txt + manifests + Arrow files)
    to the HF Hub dataset repo for persistent storage across Space restarts.
    Uses upload_large_folder for efficient parallel chunked uploads.
    """
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    gt_repo = os.environ.get("GT_REPO", GT_REPO_DEFAULT)

    if not GT_DIR.exists():
        sys.exit("[save-gt] ERROR: ground_truth dir not found — run prepare first")

    api = HfApi()
    api.create_repo(gt_repo, repo_type="dataset", exist_ok=True, private=True, token=token)

    # Count files for a progress hint
    all_files = list(GT_DIR.rglob("*"))
    n_files = sum(1 for f in all_files if f.is_file())
    print(f"[save-gt] uploading {n_files:,} files → {gt_repo}", flush=True)

    # upload_large_folder splits into multiple commits automatically (no 25K file limit)
    api.upload_large_folder(
        folder_path=str(GT_DIR),
        repo_id=gt_repo,
        repo_type="dataset",
        path_in_repo="ground_truth",
        token=token,
        num_workers=8,
        print_report_every=60,
    )
    print(f"[save-gt] done → https://huggingface.co/datasets/{gt_repo}", flush=True)


def load_gt():
    """
    Download the ground_truth folder from the HF Hub dataset repo back into /tmp.
    Skips files that already exist (idempotent — safe to re-run).
    """
    from huggingface_hub import snapshot_download
    token = os.environ.get("HF_TOKEN")
    gt_repo = os.environ.get("GT_REPO", GT_REPO_DEFAULT)

    print(f"[load-gt] downloading ground_truth from {gt_repo} …", flush=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    local_path = snapshot_download(
        repo_id=gt_repo,
        repo_type="dataset",
        local_dir=str(_BASE),
        token=token,
        ignore_patterns=["*.gitattributes", "README.md"],
    )
    print(f"[load-gt] done → {local_path}", flush=True)

    # Verify manifests are present
    for f in ("train.txt", "val.txt", "stats.json"):
        p = GT_DIR / f
        if p.exists():
            print(f"[load-gt] ✓ {f}", flush=True)
        else:
            print(f"[load-gt] ✗ {f} missing", flush=True)


# ---------------------------------------------------------------------------
# Checkpoint upload
# ---------------------------------------------------------------------------

def push_checkpoints(output_repo: str):
    """Upload all .mlmodel files in MODELS_DIR to the HF Hub model repo."""
    from huggingface_hub import HfApi
    api = HfApi()
    token = os.environ.get("HF_TOKEN")
    api.create_repo(output_repo, repo_type="model", exist_ok=True, token=token)
    for ckpt in sorted(MODELS_DIR.glob("*.mlmodel")):
        print(f"[push] uploading {ckpt.name} → {output_repo}", flush=True)
        api.upload_file(
            path_or_fileobj=str(ckpt),
            path_in_repo=ckpt.name,
            repo_id=output_repo,
            repo_type="model",
            token=token,
        )
    print("[push] done", flush=True)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="stage", required=True)

    pr = sub.add_parser("prepare")
    pr.add_argument("--limit",     type=int,   default=None)
    pr.add_argument("--val-ratio", type=float, default=0.1)

    tr = sub.add_parser("train")
    tr.add_argument("--epochs",     type=int,   default=50)
    tr.add_argument("--batch-size", type=int,   default=32)
    tr.add_argument("--lr",         type=float, default=2e-4)
    tr.add_argument("--resume",     type=str,   default=None)

    sub.add_parser("compile")
    sub.add_parser("save-gt")
    sub.add_parser("load-gt")

    pu = sub.add_parser("push")
    pu.add_argument("--output-repo", type=str, required=True)

    args = p.parse_args()

    if args.stage == "prepare":
        prepare(args.limit, args.val_ratio)
    elif args.stage == "compile":
        compile_gt()
    elif args.stage == "save-gt":
        save_gt()
    elif args.stage == "load-gt":
        load_gt()
    elif args.stage == "train":
        train(args.epochs, args.batch_size, args.lr, args.resume)
    elif args.stage == "push":
        push_checkpoints(args.output_repo)


if __name__ == "__main__":
    main()
