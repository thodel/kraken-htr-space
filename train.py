#!/usr/bin/env python3
"""
Core training logic for kraken+ on dh-unibe/image-text_medieval-scripts_xiv-xv-xvi.
Called by app.py via subprocess so the Gradio UI stays responsive.

Stages (set via --stage):
  prepare  — stream HF dataset → PNG + .gt.txt ground-truth pairs
  train    — invoke ketos train with the kraken+ VGSL spec
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

_BASE       = Path(__file__).parent / "data"
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

def prepare(limit: Optional[int], val_ratio: float):
    from datasets import load_dataset

    img_dir = GT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    print(f"[prepare] Streaming {DATASET_ID} …", flush=True)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)

    manifest_train, manifest_val = [], []
    skipped = 0

    for i, sample in enumerate(tqdm(ds, total=limit or 497_400, desc="prepare", file=sys.stdout)):
        if limit and i >= limit:
            break

        xml_raw = sample.get("xml_content", "") or ""
        xml_bytes = xml_raw.encode() if isinstance(xml_raw, str) else xml_raw
        text = _extract_text(xml_bytes)
        if not text:
            skipped += 1
            continue

        img_obj = sample["image"]
        if isinstance(img_obj, dict):
            img = Image.open(io.BytesIO(img_obj["bytes"])).convert("RGB")
        elif isinstance(img_obj, bytes):
            img = Image.open(io.BytesIO(img_obj)).convert("RGB")
        else:
            img = img_obj.convert("RGB")

        stem = f"{i:07d}"
        img_path = img_dir / f"{stem}.png"
        txt_path = img_dir / f"{stem}.gt.txt"
        img.save(img_path, "PNG")
        txt_path.write_text(text, encoding="utf-8")

        entry = str(img_path)
        if i % round(1 / val_ratio) == 0:
            manifest_val.append(entry)
        else:
            manifest_train.append(entry)

    (GT_DIR / "train.txt").write_text("\n".join(manifest_train), encoding="utf-8")
    (GT_DIR / "val.txt").write_text("\n".join(manifest_val), encoding="utf-8")

    stats = {
        "total_iterated": i + 1,
        "train": len(manifest_train),
        "val": len(manifest_val),
        "skipped": skipped,
    }
    (GT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[prepare] done — {json.dumps(stats)}", flush=True)


# ---------------------------------------------------------------------------
# Stage 2 – train
# ---------------------------------------------------------------------------

def train(
    epochs: int,
    batch_size: int,
    lr: float,
    workers: int,
    device: str,
    resume: Optional[str],
):
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_manifest = GT_DIR / "train.txt"
    val_manifest   = GT_DIR / "val.txt"
    if not train_manifest.exists():
        sys.exit("[train] ERROR: ground truth not found — run prepare first")

    cmd = [
        "ketos", "train",
        "--arch",             VGSL_SPEC,
        "--ground-truth",     str(train_manifest),
        "--evaluation-files", str(val_manifest),
        "--output",           str(MODELS_DIR / "kraken_plus"),
        "--epochs",           str(epochs),
        "--batch-size",       str(batch_size),
        "--lr",               str(lr),
        "--workers",          str(workers),
        "--device",           device,
        "--optimizer",        "Adam",
        "--schedule",         "reduceonplateau",
        "--lag",              "5",
        "--min-delta",        "0.005",
    ]
    if resume:
        cmd += ["--load", resume]

    print("[train] " + " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


# ---------------------------------------------------------------------------
# Checkpoint upload
# ---------------------------------------------------------------------------

def push_checkpoints(output_repo: str):
    """Upload all .mlmodel files in MODELS_DIR to the HF Hub."""
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
    tr.add_argument("--workers",    type=int,   default=8)
    tr.add_argument("--device",     type=str,   default="cuda:0")
    tr.add_argument("--resume",     type=str,   default=None)

    pu = sub.add_parser("push")
    pu.add_argument("--output-repo", type=str, required=True)

    args = p.parse_args()

    if args.stage == "prepare":
        prepare(args.limit, args.val_ratio)
    elif args.stage == "train":
        train(args.epochs, args.batch_size, args.lr, args.workers, args.device, args.resume)
    elif args.stage == "push":
        push_checkpoints(args.output_repo)


if __name__ == "__main__":
    main()
