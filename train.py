#!/usr/bin/env python3
"""
Core training logic for kraken+ on dh-unibe/image-text_medieval-scripts_xiv-xv-xvi.
Called by app.py via subprocess so the Gradio UI stays responsive.

Stages:
  prepare   — stream HF dataset, extract TextLine crops, write Arrow chunks
              directly (no intermediate JPEG files → no inode exhaustion).
              On completion merges chunks into train.arrow + val.arrow.
  save-gt   — push ground_truth/ folder to HF Hub dataset (persistent)
  load-gt   — restore ground_truth/ from HF Hub dataset
  compile   — no-op if Arrow files already exist (kept for compatibility)
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
# Stage 1 – prepare: stream → Arrow chunks → merge (no intermediate files)
#
# Creates only:  ground_truth/chunks/NNNNNNN.arrow  (one per CHUNK_PAGES pages)
#                ground_truth/train.arrow
#                ground_truth/val.arrow
#                progress.json
# Zero JPEG / .gt.txt files → no inode exhaustion.
# ---------------------------------------------------------------------------

# Arrow schemas
_CHUNK_SCHEMA = None   # lazy-initialised (avoids importing pyarrow at module load)
_FINAL_SCHEMA = None

CHUNK_PAGES  = 5_000   # pages per chunk file  (~75K lines, ~2 GB per chunk)
WRITE_LINES  = 1_000   # lines per Arrow record batch
N_WORKERS    = 8
PAGE_BATCH   = 128     # pages buffered before flushing to Arrow


def _arrow_schemas():
    import pyarrow as pa
    global _CHUNK_SCHEMA, _FINAL_SCHEMA
    if _CHUNK_SCHEMA is None:
        _CHUNK_SCHEMA = pa.schema([
            pa.field("image",    pa.binary()),
            pa.field("text",     pa.utf8()),
            pa.field("page_idx", pa.int32()),
        ])
        _FINAL_SCHEMA = pa.schema([
            pa.field("image", pa.binary()),
            pa.field("text",  pa.utf8()),
        ])
    return _CHUNK_SCHEMA, _FINAL_SCHEMA


def _load_progress() -> int:
    pf = _BASE / "progress.json"
    if pf.exists():
        try:
            return int(json.loads(pf.read_text()).get("last_stream_index", -1))
        except Exception:
            pass
    return -1


def _save_progress(last_idx: int) -> None:
    _BASE.mkdir(parents=True, exist_ok=True)
    (_BASE / "progress.json").write_text(json.dumps({"last_stream_index": last_idx}))


def _line_crops_from_page(sample: dict) -> list[tuple[bytes, str]]:
    """
    Decode a page image + PageXML and return (jpeg_bytes, text) for every
    valid TextLine crop.  Pure function — no disk I/O.
    """
    img_obj = sample["image"]
    try:
        if isinstance(img_obj, dict):
            img = Image.open(io.BytesIO(img_obj["bytes"])).convert("RGB")
        elif isinstance(img_obj, bytes):
            img = Image.open(io.BytesIO(img_obj)).convert("RGB")
        else:
            img = img_obj.convert("RGB")
    except Exception:
        return []

    xml_raw   = sample.get("xml_content") or ""
    xml_bytes = xml_raw.encode() if isinstance(xml_raw, str) else xml_raw

    results = []
    try:
        root = etree.fromstring(xml_bytes)
    except etree.XMLSyntaxError:
        return results

    for ln in root.findall(".//{*}TextLine"):
        u    = ln.find(".//{*}TextEquiv/{*}Unicode")
        text = (u.text or "").strip() if u is not None else ""
        if not text:
            continue

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

        if x1 - x0 < 20 or y1 - y0 < 10:
            continue

        buf = io.BytesIO()
        img.crop((x0, y0, x1, y1)).save(buf, "JPEG", quality=95)
        results.append((buf.getvalue(), text))

    return results


def _process_one(args: tuple) -> tuple[int, list[tuple[bytes, str]]]:
    """Thread worker: returns (page_idx, [(jpeg_bytes, text), …])."""
    idx, sample = args
    return idx, _line_crops_from_page(sample)


def _merge_chunks(chunks_dir: Path, gt_dir: Path, val_ratio: float) -> dict:
    """Merge all chunk Arrow files into train.arrow + val.arrow."""
    import pyarrow as pa
    _, final_schema = _arrow_schemas()

    val_step    = round(1 / val_ratio)
    chunk_files = sorted(chunks_dir.glob("*.arrow"))
    print(f"[merge] {len(chunk_files)} chunk files → train.arrow + val.arrow", flush=True)

    n_train = n_val = 0
    with pa.ipc.new_file(gt_dir / "train.arrow", final_schema) as tw, \
         pa.ipc.new_file(gt_dir / "val.arrow",   final_schema) as vw:

        for cf in tqdm(chunk_files, desc="merge chunks", file=sys.stdout):
            with pa.ipc.open_file(cf) as reader:
                for bi in range(reader.num_record_batches):
                    batch     = reader.get_batch(bi)
                    page_idxs = batch.column("page_idx").to_pylist()
                    t_idx = [i for i, p in enumerate(page_idxs) if p % val_step != 0]
                    v_idx = [i for i, p in enumerate(page_idxs) if p % val_step == 0]
                    if t_idx:
                        tw.write_batch(pa.record_batch([
                            batch.column("image").take(t_idx),
                            batch.column("text").take(t_idx),
                        ], schema=final_schema))
                        n_train += len(t_idx)
                    if v_idx:
                        vw.write_batch(pa.record_batch([
                            batch.column("image").take(v_idx),
                            batch.column("text").take(v_idx),
                        ], schema=final_schema))
                        n_val += len(v_idx)

    print(f"[merge] done — train: {n_train:,}  val: {n_val:,}", flush=True)
    return {"train": n_train, "val": n_val}


def prepare(limit: Optional[int], val_ratio: float):
    import pyarrow as pa
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from datasets import load_dataset, disable_caching

    chunk_schema, _ = _arrow_schemas()
    chunks_dir = GT_DIR / "chunks"
    chunks_dir.mkdir(parents=True, exist_ok=True)
    GT_DIR.mkdir(parents=True, exist_ok=True)

    # Resume
    last_done = _load_progress()
    start_idx = last_done + 1
    n_chunks  = len(list(chunks_dir.glob("*.arrow")))
    if start_idx > 0:
        print(f"[prepare] Resuming from page {start_idx:,} | {n_chunks} chunks on disk", flush=True)
    else:
        print(f"[prepare] Starting fresh", flush=True)

    disable_caching()
    print(f"[prepare] Streaming {DATASET_ID} from index {start_idx:,} …", flush=True)
    ds = load_dataset(DATASET_ID, split="train", streaming=True)
    if start_idx > 0:
        ds = ds.skip(start_idx)

    # Mutable chunk-writer state (wrapped in list to allow reassignment in closures)
    cw_state = {
        "start":  start_idx,
        "count":  0,
        "writer": pa.ipc.new_file(chunks_dir / f"{start_idx:07d}.arrow", chunk_schema),
    }
    # Line batch buffers
    b_imgs: list[bytes] = []
    b_txts: list[str]   = []
    b_pidx: list[int]   = []

    total_pages = total_lines = skipped = 0
    offset = -1

    def _flush_lines():
        if not b_imgs:
            return
        cw_state["writer"].write_batch(pa.record_batch([
            pa.array(b_imgs, pa.binary()),
            pa.array(b_txts, pa.utf8()),
            pa.array(b_pidx, pa.int32()),
        ], schema=chunk_schema))
        b_imgs.clear(); b_txts.clear(); b_pidx.clear()

    def _rotate_chunk(up_to_page: int):
        _flush_lines()
        cw_state["writer"].close()
        _save_progress(up_to_page)
        print(f"[prepare] chunk {cw_state['start']:07d}.arrow saved "
              f"(pages {cw_state['start']:,}–{up_to_page:,})", flush=True)
        nxt = up_to_page + 1
        cw_state.update({
            "start":  nxt,
            "count":  0,
            "writer": pa.ipc.new_file(chunks_dir / f"{nxt:07d}.arrow", chunk_schema),
        })

    with ThreadPoolExecutor(max_workers=N_WORKERS) as pool:
        pbar = tqdm(total=max(0, (limit or 497_400) - start_idx),
                    desc="prepare (pages)", file=sys.stdout)
        buf: list[tuple] = []
        max_i = start_idx

        def flush_buf():
            nonlocal total_pages, total_lines, skipped
            futs = {pool.submit(_process_one, item): item[0] for item in buf}
            for fut in as_completed(futs):
                page_idx, crops = fut.result()
                pbar.update(1)
                total_pages += 1
                cw_state["count"] += 1
                if not crops:
                    skipped += 1
                for img_b, txt in crops:
                    b_imgs.append(img_b); b_txts.append(txt); b_pidx.append(page_idx)
                    total_lines += 1
                    if len(b_imgs) >= WRITE_LINES:
                        _flush_lines()
            buf.clear()
            if cw_state["count"] >= CHUNK_PAGES:
                _rotate_chunk(max_i)

        for offset, sample in enumerate(ds):
            i = start_idx + offset
            if limit and i >= limit:
                break
            # Resume check: skip pages already in a completed chunk
            chunk_file = chunks_dir / f"{(i // CHUNK_PAGES) * CHUNK_PAGES:07d}.arrow"
            if chunk_file.exists() and i < last_done:
                pbar.update(1)
                total_pages += 1
                continue
            buf.append((i, sample))
            max_i = i
            if len(buf) >= PAGE_BATCH:
                flush_buf()

        if buf:
            flush_buf()
        pbar.close()

    # Close final (possibly partial) chunk
    _flush_lines()
    cw_state["writer"].close()
    final_idx = start_idx + offset if offset >= 0 else last_done
    _save_progress(final_idx)

    # Merge all chunks → train.arrow + val.arrow
    counts = _merge_chunks(chunks_dir, GT_DIR, val_ratio)
    stats  = {
        "last_stream_index": final_idx,
        "total_pages": total_pages,
        "total_lines": total_lines,
        "pages_skipped_no_lines": skipped,
        **counts,
    }
    (GT_DIR / "stats.json").write_text(json.dumps(stats, indent=2))
    print(f"[prepare] done — {json.dumps(stats)}", flush=True)


# ---------------------------------------------------------------------------
# Stage 2 – compile (no-op: prepare now writes Arrow directly)
# ---------------------------------------------------------------------------

def compile_gt():
    """No-op: prepare() now writes train.arrow + val.arrow directly.
    Kept so existing pipelines that call 'compile' don't break."""
    train_arrow = GT_DIR / "train.arrow"
    val_arrow   = GT_DIR / "val.arrow"
    if train_arrow.exists() and val_arrow.exists():
        print("[compile] Arrow files already present — skipping (prepare wrote them directly)", flush=True)
        return
    sys.exit("[compile] ERROR: Arrow files not found — run prepare first")


# ---------------------------------------------------------------------------
# (old compile helper kept only for reference — no longer called)
# ---------------------------------------------------------------------------

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
