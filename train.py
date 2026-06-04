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
MODELS_DIR  = Path(os.environ.get("KRAKEN_MODELS_DIR", str(_BASE / "models")))
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

def _extract_line_crops(
    xml_bytes: bytes,
    img: Image.Image,
    page_idx: int,
    img_dir: Path,
) -> list[str]:
    """
    Parse PageXML, crop each TextLine from the page image, save as
    JPEG + .gt.txt.  Returns list of saved image paths.

    Naming: {page_idx:07d}_{line_idx:04d}.jpg
    """
    saved = []
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return saved

    for line_idx, ln in enumerate(root.findall(".//{*}TextLine")):
        # --- transcription ---
        u = ln.find(".//{*}TextEquiv/{*}Unicode")
        text = (u.text or "").strip() if u is not None else ""
        if not text:
            continue

        # --- bounding box from polygon Coords ---
        coords_el = ln.find(".//{*}Coords")
        if coords_el is None:
            continue
        pts_str = coords_el.get("points", "").strip()
        if not pts_str:
            continue
        try:
            pts = [(int(a), int(b)) for a, b in (p.split(",") for p in pts_str.split())]
            x0 = max(0,          min(p[0] for p in pts))
            y0 = max(0,          min(p[1] for p in pts))
            x1 = min(img.width,  max(p[0] for p in pts))
            y1 = min(img.height, max(p[1] for p in pts))
        except (ValueError, IndexError):
            continue

        if x1 - x0 < 20 or y1 - y0 < 10:   # skip degenerate crops
            continue

        crop = img.crop((x0, y0, x1, y1))
        stem     = f"{page_idx:07d}_{line_idx:04d}"
        img_path = img_dir / f"{stem}.jpg"
        gt_path  = img_dir / f"{stem}.gt.txt"
        crop.save(img_path, "JPEG", quality=95)
        gt_path.write_text(text, encoding="utf-8")
        saved.append(str(img_path))

    return saved


def _process_one(args: tuple) -> tuple[int, list[str]]:
    """
    Process one page sample: decode image, extract all TextLine crops,
    write JPEG + .gt.txt per line.  Returns (page_idx, [saved_paths]).
    """
    idx, sample, img_dir = args

    # Decode page image
    img_obj = sample["image"]
    try:
        if isinstance(img_obj, dict):
            img = Image.open(io.BytesIO(img_obj["bytes"])).convert("RGB")
        elif isinstance(img_obj, bytes):
            img = Image.open(io.BytesIO(img_obj)).convert("RGB")
        else:
            img = img_obj.convert("RGB")
    except Exception:
        return idx, []

    xml_raw   = sample.get("xml_content") or ""
    xml_bytes = xml_raw.encode() if isinstance(xml_raw, str) else xml_raw
    paths     = _extract_line_crops(xml_bytes, img, idx, Path(img_dir))
    return idx, paths


PROGRESS_FILE = GT_DIR.parent / "progress.json"   # written every SAVE_EVERY items
SAVE_EVERY    = 1_000


def _load_progress() -> int:
    """Return the last fully-processed stream index, or -1 if starting fresh."""
    pf = Path(os.environ.get("KRAKEN_DATA_DIR", "/tmp/kraken_data")) / "progress.json"
    if pf.exists():
        try:
            return int(json.loads(pf.read_text()).get("last_stream_index", -1))
        except Exception:
            pass
    return -1


def _save_progress(last_idx: int) -> None:
    pf = Path(os.environ.get("KRAKEN_DATA_DIR", "/tmp/kraken_data")) / "progress.json"
    pf.write_text(json.dumps({"last_stream_index": last_idx}))


def _rebuild_manifests(img_dir: Path, val_ratio: float) -> dict:
    """
    Scan all JPEG+.gt.txt line-crop pairs and rebuild train/val manifests.
    Files are named {page_idx:07d}_{line_idx:04d}.jpg; val split is based
    on page_idx so whole pages go to one split (no line leakage).
    """
    manifest_train, manifest_val = [], []
    val_step = round(1 / val_ratio)
    for img_path in sorted(img_dir.glob("*.jpg")):
        if not img_path.with_suffix(".gt.txt").exists():
            continue
        # extract page index from stem "PPPPPPP_LLLL"
        try:
            page_idx = int(img_path.stem.split("_")[0])
        except ValueError:
            continue
        if page_idx % val_step == 0:
            manifest_val.append(str(img_path))
        else:
            manifest_train.append(str(img_path))
    (img_dir.parent / "train.txt").write_text("\n".join(manifest_train), encoding="utf-8")
    (img_dir.parent / "val.txt").write_text("\n".join(manifest_val), encoding="utf-8")
    return {"train": len(manifest_train), "val": len(manifest_val)}


def prepare(limit: Optional[int], val_ratio: float):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datasets import load_dataset

    img_dir = GT_DIR / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    N_WORKERS = 8
    BATCH     = 128

    # --- Resume support ---
    last_done = _load_progress()
    start_idx = last_done + 1

    if start_idx > 0:
        # Count files actually on disk to report accurate progress
        n_on_disk = sum(1 for _ in img_dir.glob("*.jpg"))
        print(f"[prepare] Resuming — last stream index: {last_done:,} "
              f"| files on disk: {n_on_disk:,}", flush=True)
    else:
        print(f"[prepare] Starting fresh", flush=True)

    print(f"[prepare] Streaming {DATASET_ID} from index {start_idx:,} …", flush=True)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    if start_idx > 0:
        ds = ds.skip(start_idx)

    skipped   = 0
    new_total = 0
    img_dir_s = str(img_dir)
    last_saved_idx = last_done

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        pbar = tqdm(
            total=max(0, (limit or 497_400) - start_idx),
            desc="prepare (pages)",
            file=sys.stdout,
        )
        buf: list[tuple] = []
        offset = -1

        def flush_buf(up_to_idx: int):
            nonlocal skipped, new_total, last_saved_idx
            futures = {pool.submit(_process_one, item): item[0] for item in buf}
            for fut in as_completed(futures):
                _, paths = fut.result()
                pbar.update(1)
                new_total += 1
                if not paths:
                    skipped += 1
            buf.clear()
            if up_to_idx - last_saved_idx >= SAVE_EVERY:
                _save_progress(up_to_idx)
                last_saved_idx = up_to_idx

        for offset, sample in enumerate(ds):
            i = start_idx + offset
            if limit and i >= limit:
                break

            # Resume check: if the first line crop for this page exists, skip it
            first_crop = img_dir / f"{i:07d}_0000.jpg"
            if first_crop.exists():
                pbar.update(1)
                new_total += 1
                continue

            buf.append((i, sample, img_dir_s))
            if len(buf) >= BATCH:
                flush_buf(i)

        if buf:
            flush_buf(start_idx + offset)
        pbar.close()

    # Save final progress
    final_idx = start_idx + offset if offset >= 0 else last_done
    _save_progress(final_idx)

    # Rebuild manifests from everything on disk (old + new)
    counts = _rebuild_manifests(img_dir, val_ratio)
    total_lines = counts["train"] + counts["val"]
    stats = {
        "last_stream_index": final_idx,
        "pages_this_run": new_total,
        "pages_skipped_no_lines": skipped,
        "total_line_crops": total_lines,
        **counts,
    }
    (GT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[prepare] done — {json.dumps(stats)}", flush=True)
    print(f"[prepare] ~{total_lines:,} line crops total "
          f"(~{total_lines/(final_idx+1):.1f} lines/page avg)", flush=True)


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
    num_gpus: int,
    resume: Optional[str],
):
    """
    Invoke ketos train with correct kraken 7.x flags.

    Multi-GPU: the ketos root command accepts -d cuda:0,cuda:1 which is passed
    to KrakenTrainer(devices=[0,1], accelerator='cuda') via Lightning.
    Each GPU receives `batch_size` samples → effective batch = batch_size × num_gpus.
    """
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    train_arrow = GT_DIR / "train.arrow"
    val_arrow   = GT_DIR / "val.arrow"
    if not train_arrow.exists():
        sys.exit("[train] ERROR: train.arrow not found — run compile first")

    # Build device string: "cuda:0" for 1 GPU, "cuda:0,cuda:1" for 2, etc.
    device_str = ",".join(f"cuda:{i}" for i in range(num_gpus))

    cmd = [
        "ketos",
        "-d", device_str,
        "--workers", str(min(8, num_gpus * 4)),  # data loader workers
        "train",
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
    tr.add_argument("--num-gpus",   type=int,   default=1)
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
        train(args.epochs, args.batch_size, args.lr, args.num_gpus, args.resume)
    elif args.stage == "push":
        push_checkpoints(args.output_repo)


if __name__ == "__main__":
    main()
