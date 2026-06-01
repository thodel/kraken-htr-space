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

_BASE       = Path("/tmp/kraken_data")
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

# Worker function must be module-level for multiprocessing pickling
def _process_batch(batch: dict, indices: list, img_dir: str, val_ratio: float):
    """Called by datasets.map — converts one batch to JPEG + .gt.txt pairs."""
    import io as _io
    from lxml import etree as _etree
    from PIL import Image as _Image

    img_dir = Path(img_dir)
    results = {"img_path": [], "has_text": []}

    for idx, xml_raw, img_obj in zip(indices, batch["xml_content"], batch["image"]):
        # --- extract text ---
        xml_bytes = (xml_raw or "").encode() if isinstance(xml_raw, str) else (xml_raw or b"")
        try:
            root = _etree.fromstring(xml_bytes)
            nodes = root.findall(".//{*}TextLine/{*}TextEquiv/{*}Unicode")
            texts = [n.text.strip() for n in nodes if n.text and n.text.strip()]
            text = " ".join(texts) if texts else None
        except Exception:
            text = None

        if not text:
            results["img_path"].append("")
            results["has_text"].append(False)
            continue

        # --- decode image ---
        try:
            if isinstance(img_obj, dict):
                img = _Image.open(_io.BytesIO(img_obj["bytes"])).convert("RGB")
            elif isinstance(img_obj, bytes):
                img = _Image.open(_io.BytesIO(img_obj)).convert("RGB")
            else:
                img = img_obj.convert("RGB")
        except Exception:
            results["img_path"].append("")
            results["has_text"].append(False)
            continue

        stem = f"{idx:07d}"
        img_path = img_dir / f"{stem}.jpg"
        txt_path = img_dir / f"{stem}.gt.txt"
        img.save(img_path, "JPEG", quality=95)
        txt_path.write_text(text, encoding="utf-8")

        results["img_path"].append(str(img_path))
        results["has_text"].append(True)

    return results


def prepare(limit: Optional[int], val_ratio: float):
    import multiprocessing
    from datasets import load_dataset

    img_dir = GT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    num_proc = min(8, multiprocessing.cpu_count())

    print(f"[prepare] Downloading {DATASET_ID} to local cache (num_proc={num_proc}) …", flush=True)
    # Non-streaming: downloads all parquet files to HF cache first,
    # then processes locally — orders of magnitude faster than streaming.
    ds = load_dataset(DATASET_ID, split="train", num_proc=num_proc, verification_mode="no_checks")

    if limit:
        ds = ds.select(range(limit))
        print(f"[prepare] Limited to {limit} samples", flush=True)

    print(f"[prepare] Processing {len(ds):,} samples with {num_proc} workers …", flush=True)

    processed = ds.map(
        _process_batch,
        batched=True,
        batch_size=256,
        with_indices=True,
        num_proc=num_proc,
        fn_kwargs={"img_dir": str(img_dir), "val_ratio": val_ratio},
        desc="prepare",
    )

    # Build manifests from results
    manifest_train, manifest_val = [], []
    skipped = 0
    for i, row in enumerate(processed):
        if not row["has_text"]:
            skipped += 1
            continue
        entry = row["img_path"]
        if i % round(1 / val_ratio) == 0:
            manifest_val.append(entry)
        else:
            manifest_train.append(entry)

    (GT_DIR / "train.txt").write_text("\n".join(manifest_train), encoding="utf-8")
    (GT_DIR / "val.txt").write_text("\n".join(manifest_val), encoding="utf-8")

    stats = {
        "total": len(processed),
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
    cmd = [
        "ketos", "train",
        "-f", "path",
        "-s", VGSL_SPEC,
        "-t", str(train_manifest),
        "-e", str(val_manifest),
        "-o", str(MODELS_DIR / "kraken_plus"),
        "-N", str(epochs),
        "-B", str(batch_size),
        "-r", str(lr),
        "--optimizer",       "Adam",
        "--schedule",        "reduceonplateau",
        "--lag",             "5",
        "--min-delta",       "0.005",
    ]
    if resume:
        cmd += ["--resume", resume]

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
    tr.add_argument("--resume",     type=str,   default=None)

    pu = sub.add_parser("push")
    pu.add_argument("--output-repo", type=str, required=True)

    args = p.parse_args()

    if args.stage == "prepare":
        prepare(args.limit, args.val_ratio)
    elif args.stage == "train":
        train(args.epochs, args.batch_size, args.lr, args.resume)
    elif args.stage == "push":
        push_checkpoints(args.output_repo)


if __name__ == "__main__":
    main()
