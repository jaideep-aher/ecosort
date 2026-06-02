"""Download the TrashNet dataset and materialize a compact, on-disk image set.

The full HuggingFace mirror of TrashNet decodes to ~3.7 GB, but the repository
also hosts ``dataset-resized.zip`` (~43 MB) — the canonical ~2,500-image set
resized to 512x384 by the original authors. We download that archive directly
via ``huggingface_hub``, extract it, and reorganize the images into
``data/raw/<class>/``. A stratified train/val/test split index is then written
to ``data/processed/split_index.json``.

Run:  python scripts/make_dataset.py
"""

from __future__ import annotations

import json
import os
import random
import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from PIL import Image

from config import (
    CLASS_NAMES,
    HF_DATASET_ID,
    RAW_DIR,
    SPLIT_INDEX_PATH,
    SPLIT_SEED,
    TEST_FRACTION,
    VAL_FRACTION,
)

RESIZED_ARCHIVE = "dataset-resized.zip"
# Optional cap per class (keeps training fast); default keeps the full set.
PER_CLASS_CAP = int(os.environ.get("ECOSORT_PER_CLASS", "1000"))
# Longest-side size for stored JPEGs (decoupled from model input size).
STORE_MAX_SIDE = 512
JPEG_QUALITY = 88


def _resize_keep_aspect(img: Image.Image, max_side: int) -> Image.Image:
    """Resize so the longest side equals ``max_side`` (only downscales)."""
    img = img.convert("RGB")
    w, h = img.size
    scale = min(1.0, max_side / float(max(w, h)))
    if scale < 1.0:
        img = img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.LANCZOS)
    return img


def _count_existing() -> Dict[str, int]:
    counts = {}
    for cls in CLASS_NAMES:
        d = RAW_DIR / cls
        counts[cls] = len(list(d.glob("*.jpg"))) if d.exists() else 0
    return counts


def download_and_materialize(per_class_cap: int = PER_CLASS_CAP) -> Dict[str, int]:
    """Download the resized TrashNet archive and save per-class JPEGs.

    Returns a mapping of class name -> number of images saved.
    """
    existing = _count_existing()
    if all(existing[c] >= 100 for c in CLASS_NAMES):
        print(f"[make_dataset] Found existing images {existing}; skipping download.")
        return existing

    from huggingface_hub import hf_hub_download

    cache_dir = RAW_DIR / "_hfcache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"[make_dataset] Downloading {RESIZED_ARCHIVE} from {HF_DATASET_ID}...")
    zip_path = hf_hub_download(
        repo_id=HF_DATASET_ID,
        filename=RESIZED_ARCHIVE,
        repo_type="dataset",
        cache_dir=str(cache_dir),
    )

    extract_dir = cache_dir / "extracted"
    extract_dir.mkdir(parents=True, exist_ok=True)
    print(f"[make_dataset] Extracting archive...")
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(extract_dir)

    saved: Dict[str, int] = defaultdict(int)
    for cls in CLASS_NAMES:
        (RAW_DIR / cls).mkdir(parents=True, exist_ok=True)
        # Archive lays out images as dataset-resized/<class>/<name>.jpg
        src_dirs = list(extract_dir.glob(f"**/{cls}"))
        src_files: List[Path] = []
        for sd in src_dirs:
            if sd.is_dir():
                src_files.extend(sorted(sd.glob("*.jpg")))
                src_files.extend(sorted(sd.glob("*.jpeg")))
                src_files.extend(sorted(sd.glob("*.png")))
        for src in src_files:
            if saved[cls] >= per_class_cap:
                break
            try:
                img = _resize_keep_aspect(Image.open(src), STORE_MAX_SIDE)
            except Exception as exc:  # noqa: BLE001 - skip unreadable images
                print(f"[make_dataset] skipping unreadable image {src.name}: {exc}")
                continue
            out_path = RAW_DIR / cls / f"{cls}_{saved[cls]:04d}.jpg"
            img.save(out_path, "JPEG", quality=JPEG_QUALITY)
            saved[cls] += 1

    # Clean up the transient download/extract cache to reclaim disk.
    shutil.rmtree(cache_dir, ignore_errors=True)

    final = dict(saved)
    print(f"[make_dataset] Saved per-class counts: {final}")
    return final


def build_split_index() -> Dict[str, List[dict]]:
    """Create a stratified train/val/test split over the materialized images."""
    rng = random.Random(SPLIT_SEED)
    splits: Dict[str, List[dict]] = {"train": [], "val": [], "test": []}

    for label, cls in enumerate(CLASS_NAMES):
        files = sorted((RAW_DIR / cls).glob("*.jpg"))
        rng.shuffle(files)
        n = len(files)
        n_test = int(round(n * TEST_FRACTION))
        n_val = int(round(n * VAL_FRACTION))
        test_files = files[:n_test]
        val_files = files[n_test:n_test + n_val]
        train_files = files[n_test + n_val:]
        for split_name, flist in (("train", train_files), ("val", val_files), ("test", test_files)):
            for f in flist:
                splits[split_name].append({
                    "path": str(f.relative_to(RAW_DIR.parent.parent)),
                    "label": label,
                    "class": cls,
                })

    for split_name in splits:
        rng.shuffle(splits[split_name])

    SPLIT_INDEX_PATH.write_text(json.dumps(splits, indent=2))
    sizes = {k: len(v) for k, v in splits.items()}
    print(f"[make_dataset] Split sizes: {sizes} -> {SPLIT_INDEX_PATH}")
    return splits


def load_split_index() -> Dict[str, List[dict]]:
    """Load the persisted split index, building it if necessary."""
    if not SPLIT_INDEX_PATH.exists():
        return build_split_index()
    return json.loads(SPLIT_INDEX_PATH.read_text())


def main() -> None:
    download_and_materialize()
    build_split_index()


if __name__ == "__main__":
    main()
