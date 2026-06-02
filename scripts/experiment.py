"""Focused experiment: robustness to real-world image corruptions.

Motivation
----------
EcoSort is meant to run on photos taken by everyday users with phone cameras,
under varied lighting and motion. A model that is accurate on clean studio
images but collapses under mild blur/noise/brightness shifts is not deployable.
This experiment quantifies how the *classical* model and the *deep* model degrade
under five controlled corruptions at increasing severities, on the test set.

This mirrors the corruption-robustness methodology of Hendrycks & Dietterich
(ICLR 2019, "Benchmarking Neural Network Robustness to Common Corruptions and
Perturbations": https://arxiv.org/abs/1903.12261), adapted to TrashNet.

Output: ``data/outputs/experiment_robustness.json`` and
``data/outputs/experiment_robustness.png``.
"""

from __future__ import annotations

import json
from typing import Callable, Dict, List

import numpy as np
from PIL import Image, ImageFilter
from sklearn.metrics import f1_score

from config import EXPERIMENT_PATH, OUTPUTS_DIR
from data_utils import get_split, load_image

SEVERITIES = [0, 1, 2, 3]


# --- Corruption functions (operate on PIL RGB images) ------------------------
def corrupt_gaussian_noise(img: Image.Image, sev: int) -> Image.Image:
    if sev == 0:
        return img
    std = [0, 12, 25, 45][sev]
    arr = np.asarray(img).astype(np.float32)
    arr += np.random.normal(0, std, arr.shape)
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))


def corrupt_blur(img: Image.Image, sev: int) -> Image.Image:
    if sev == 0:
        return img
    radius = [0, 1.0, 2.0, 3.5][sev]
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def corrupt_brightness(img: Image.Image, sev: int) -> Image.Image:
    if sev == 0:
        return img
    from PIL import ImageEnhance
    factor = [1.0, 1.4, 1.8, 0.45][sev]  # over/under-exposure
    return ImageEnhance.Brightness(img).enhance(factor)


def corrupt_jpeg(img: Image.Image, sev: int) -> Image.Image:
    if sev == 0:
        return img
    import io
    quality = [100, 30, 15, 7][sev]
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def corrupt_occlusion(img: Image.Image, sev: int) -> Image.Image:
    if sev == 0:
        return img
    frac = [0, 0.15, 0.3, 0.45][sev]
    arr = np.asarray(img).copy()
    h, w = arr.shape[:2]
    bh, bw = int(h * frac), int(w * frac)
    y0, x0 = (h - bh) // 2, (w - bw) // 2
    arr[y0:y0 + bh, x0:x0 + bw] = 0
    return Image.fromarray(arr)


CORRUPTIONS: Dict[str, Callable[[Image.Image, int], Image.Image]] = {
    "gaussian_noise": corrupt_gaussian_noise,
    "blur": corrupt_blur,
    "brightness": corrupt_brightness,
    "jpeg": corrupt_jpeg,
    "occlusion": corrupt_occlusion,
}


def _score_deep(images: List[Image.Image], y_true: np.ndarray, clf) -> float:
    preds = clf.predict_proba_batch(images).argmax(axis=1)
    return float(f1_score(y_true, preds, average="macro"))


def _score_classical(images: List[Image.Image], y_true: np.ndarray, clf) -> float:
    from build_features import extract_features
    X = np.vstack([extract_features(im) for im in images])
    preds = clf.predict_proba_features(X).argmax(axis=1)
    return float(f1_score(y_true, preds, average="macro"))


def run_experiment(seed: int = 42) -> Dict:
    """Measure macro-F1 vs. corruption severity for classical & deep models."""
    np.random.seed(seed)
    from deep_model import DeepClassifier
    from classical_model import ClassicalModel

    deep = DeepClassifier.load()
    classical = ClassicalModel.load()

    paths, y_true = get_split("test")
    base_images = [load_image(p) for p in paths]

    results: Dict = {"severities": SEVERITIES, "corruptions": {}}
    for name, fn in CORRUPTIONS.items():
        deep_scores, classical_scores = [], []
        for sev in SEVERITIES:
            corrupted = [fn(im, sev) for im in base_images]
            deep_scores.append(_score_deep(corrupted, y_true, deep))
            classical_scores.append(_score_classical(corrupted, y_true, classical))
        results["corruptions"][name] = {"deep": deep_scores, "classical": classical_scores}
        print(f"[experiment] {name}: deep={deep_scores} classical={classical_scores}")

    _plot(results)
    EXPERIMENT_PATH.write_text(json.dumps(results, indent=2))
    print(f"[experiment] wrote {EXPERIMENT_PATH}")
    return results


def _plot(results: Dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    corr = results["corruptions"]
    fig, axes = plt.subplots(1, len(corr), figsize=(4 * len(corr), 4), sharey=True)
    for ax, (name, scores) in zip(axes, corr.items()):
        ax.plot(SEVERITIES, scores["deep"], "o-", label="deep", color="#2a9d8f")
        ax.plot(SEVERITIES, scores["classical"], "s--", label="classical", color="#e76f51")
        ax.set_title(name)
        ax.set_xlabel("severity")
        ax.set_ylim(0, 1)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("macro-F1")
    axes[0].legend()
    fig.suptitle("Robustness to common image corruptions (TrashNet test set)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "experiment_robustness.png", dpi=130)
    plt.close(fig)


if __name__ == "__main__":
    run_experiment()
