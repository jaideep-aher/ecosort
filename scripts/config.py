"""Central configuration for the EcoSort waste-classification project.

Defines canonical paths, class metadata, and shared hyperparameters so that
the data pipeline, models, experiments, and web app all agree on conventions.
"""

from __future__ import annotations

import os
from pathlib import Path

# --- Filesystem layout -------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
OUTPUTS_DIR = DATA_DIR / "outputs"
MODELS_DIR = PROJECT_ROOT / "models"

for _d in (RAW_DIR, PROCESSED_DIR, OUTPUTS_DIR, MODELS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --- Dataset -----------------------------------------------------------------
# TrashNet: ~2,500 images of household waste across 6 recyclable categories.
# Source: Yang & Thung, "Classification of Trash for Recyclability Status",
# Stanford CS229 (2016). HuggingFace mirror: garythung/trashnet
HF_DATASET_ID = "garythung/trashnet"

CLASS_NAMES = ["cardboard", "glass", "metal", "paper", "plastic", "trash"]
NUM_CLASSES = len(CLASS_NAMES)

# Human-facing guidance shown in the web app (the product layer).
BIN_GUIDANCE = {
    "cardboard": {
        "bin": "Recycling (paper/cardboard)",
        "tip": "Flatten boxes and keep them dry. Greasy cardboard (e.g. pizza boxes) goes to compost or trash.",
        "color": "#b07d56",
    },
    "glass": {
        "bin": "Recycling (glass)",
        "tip": "Rinse bottles and jars. Remove lids. Broken glass is usually not recyclable curbside.",
        "color": "#4aa3a2",
    },
    "metal": {
        "bin": "Recycling (metal)",
        "tip": "Rinse cans. Aluminum and steel are infinitely recyclable.",
        "color": "#8d99ae",
    },
    "paper": {
        "bin": "Recycling (paper)",
        "tip": "Keep paper clean and dry. Shredded paper often needs to be bagged.",
        "color": "#e9c46a",
    },
    "plastic": {
        "bin": "Recycling (plastic) — check local resin code",
        "tip": "Rinse containers. Films and bags usually need special drop-off, not curbside.",
        "color": "#2a9d8f",
    },
    "trash": {
        "bin": "Landfill / general waste",
        "tip": "Not recyclable in standard streams. Consider whether parts can be separated.",
        "color": "#6c757d",
    },
}

# --- Splits ------------------------------------------------------------------
SPLIT_SEED = 42
VAL_FRACTION = 0.15
TEST_FRACTION = 0.15

# --- Image preprocessing -----------------------------------------------------
IMAGE_SIZE = 224          # input size for the deep model
CLASSIC_IMAGE_SIZE = 128  # smaller canvas for classical feature extraction

# ImageNet normalization (MobileNet was pretrained on ImageNet).
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# --- Deep model training -----------------------------------------------------
DEEP_BATCH_SIZE = 32
DEEP_EPOCHS = int(os.environ.get("ECOSORT_EPOCHS", "8"))
DEEP_LR = 1e-3
DEEP_WEIGHT_DECAY = 1e-4

# --- Artifact paths ----------------------------------------------------------
SPLIT_INDEX_PATH = PROCESSED_DIR / "split_index.json"
CLASSIC_FEATURES_PATH = PROCESSED_DIR / "classic_features.npz"

NAIVE_MODEL_PATH = MODELS_DIR / "naive_baseline.joblib"
CLASSIC_MODEL_PATH = MODELS_DIR / "classical_rf.joblib"
DEEP_MODEL_PATH = MODELS_DIR / "deep_mobilenet.pt"

METRICS_PATH = OUTPUTS_DIR / "metrics.json"
EXPERIMENT_PATH = OUTPUTS_DIR / "experiment_robustness.json"


def device() -> str:
    """Return the best available torch device string (mps/cuda/cpu)."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
