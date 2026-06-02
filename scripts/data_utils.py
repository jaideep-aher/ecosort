"""Shared data-access helpers used across training, evaluation, and the app."""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
from PIL import Image

from config import PROJECT_ROOT
from make_dataset import load_split_index


def resolve_path(rel_or_abs: str) -> Path:
    """Resolve a split-index path (stored relative to the project root)."""
    p = Path(rel_or_abs)
    return p if p.is_absolute() else (PROJECT_ROOT / p)


def load_image(path: str) -> Image.Image:
    """Load an image as RGB."""
    return Image.open(resolve_path(path)).convert("RGB")


def get_split(split: str) -> Tuple[List[str], np.ndarray]:
    """Return (image_paths, labels) for a split: 'train' | 'val' | 'test'."""
    index = load_split_index()
    items = index[split]
    paths = [it["path"] for it in items]
    labels = np.array([it["label"] for it in items], dtype=np.int64)
    return paths, labels
