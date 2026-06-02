"""Classical computer-vision feature extraction for the non-deep-learning model.

Each image is described by a fixed-length, hand-crafted feature vector that
captures both *colour* and *texture/shape* — the two cues a human uses to tell
recyclables apart:

* HSV colour histograms (32 bins per channel) + per-channel colour moments
  (mean, std) — captures, e.g., the brown of cardboard vs. the green of glass.
* Histogram of Oriented Gradients (HOG) on the grayscale image — captures
  texture and edge structure (crumpled paper vs. smooth metal).

The same ``extract_features`` function is reused at inference time by the web
app, guaranteeing train/serve parity.
"""

from __future__ import annotations

from typing import List

import numpy as np
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import hog
from tqdm import tqdm

from config import CLASSIC_FEATURES_PATH, CLASSIC_IMAGE_SIZE
from data_utils import get_split, load_image

HSV_BINS = 32
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)


def _color_features(rgb: np.ndarray) -> np.ndarray:
    """HSV histograms + colour moments from an HxWx3 uint8 array."""
    from skimage.color import rgb2hsv

    hsv = rgb2hsv(rgb)  # values in [0, 1]
    feats: List[np.ndarray] = []
    for ch in range(3):
        hist, _ = np.histogram(hsv[:, :, ch], bins=HSV_BINS, range=(0.0, 1.0), density=True)
        feats.append(hist.astype(np.float32))
        feats.append(np.array([hsv[:, :, ch].mean(), hsv[:, :, ch].std()], dtype=np.float32))
    return np.concatenate(feats)


def _texture_features(rgb: np.ndarray) -> np.ndarray:
    """HOG descriptor from the grayscale image."""
    gray = rgb2gray(rgb)
    return hog(
        gray,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        block_norm="L2-Hys",
        feature_vector=True,
    ).astype(np.float32)


def extract_features(image: Image.Image, size: int = CLASSIC_IMAGE_SIZE) -> np.ndarray:
    """Compute the full classical feature vector for a single PIL image."""
    img = image.convert("RGB").resize((size, size), Image.BILINEAR)
    rgb = np.asarray(img, dtype=np.uint8)
    return np.concatenate([_color_features(rgb), _texture_features(rgb)])


def _featurize_paths(paths: List[str]) -> np.ndarray:
    rows = [extract_features(load_image(p)) for p in tqdm(paths, desc="features")]
    return np.vstack(rows).astype(np.float32)


def build_feature_matrices() -> dict:
    """Extract features for all splits and persist them as a single .npz."""
    out = {}
    for split in ("train", "val", "test"):
        paths, labels = get_split(split)
        X = _featurize_paths(paths)
        out[f"X_{split}"] = X
        out[f"y_{split}"] = labels
        print(f"[build_features] {split}: X={X.shape}")
    np.savez_compressed(CLASSIC_FEATURES_PATH, **out)
    print(f"[build_features] Saved -> {CLASSIC_FEATURES_PATH}")
    return out


def load_feature_matrices() -> dict:
    """Load cached feature matrices, building them if missing."""
    if not CLASSIC_FEATURES_PATH.exists():
        return build_feature_matrices()
    data = np.load(CLASSIC_FEATURES_PATH)
    return {k: data[k] for k in data.files}


def main() -> None:
    build_feature_matrices()


if __name__ == "__main__":
    main()
