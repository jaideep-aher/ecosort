"""Evaluate all three models on the held-out test set and emit metrics + plots.

Produces:
* ``data/outputs/metrics.json`` — accuracy, macro/weighted F1, and per-class
  precision/recall/F1 for the naive, classical, and deep models.
* ``data/outputs/confusion_<model>.png`` — confusion matrices.
* ``data/outputs/error_analysis.json`` — the deep model's most confident
  mistakes (used for the report's error-analysis section).

Metric rationale: because the classes are imbalanced (e.g. 'trash' is ~5% of the
data), plain accuracy is misleading. We treat **macro-F1** as the primary metric
since it weights every recyclable category equally.
"""

from __future__ import annotations

import json
from typing import Dict, List

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from config import CLASS_NAMES, METRICS_PATH, OUTPUTS_DIR
from data_utils import get_split, load_image


def _metrics_block(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted")),
        "per_class": classification_report(
            y_true, y_pred, target_names=CLASS_NAMES, output_dict=True, zero_division=0
        ),
    }


def _plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, title: str, fname: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(CLASS_NAMES))))
    cm_norm = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(cm_norm, cmap="Greens", vmin=0, vmax=1)
    ax.set_xticks(range(len(CLASS_NAMES)), CLASS_NAMES, rotation=45, ha="right")
    ax.set_yticks(range(len(CLASS_NAMES)), CLASS_NAMES)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    for i in range(len(CLASS_NAMES)):
        for j in range(len(CLASS_NAMES)):
            ax.text(j, i, f"{cm[i, j]}", ha="center", va="center",
                    color="white" if cm_norm[i, j] > 0.5 else "black", fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / fname, dpi=130)
    plt.close(fig)


def _deep_test_predictions():
    """Return (y_true, y_pred, probs, paths) for the deep model on the test set."""
    from deep_model import DeepClassifier

    clf = DeepClassifier.load()
    paths, y_true = get_split("test")
    images = [load_image(p) for p in paths]
    probs = clf.predict_proba_batch(images)
    return y_true, probs.argmax(axis=1), probs, paths


def _error_analysis(y_true, y_pred, probs, paths, top_k: int = 8) -> List[Dict]:
    """Surface the most confident misclassifications for the report."""
    errors = []
    for i in range(len(y_true)):
        if y_pred[i] != y_true[i]:
            errors.append({
                "path": paths[i],
                "true": CLASS_NAMES[int(y_true[i])],
                "predicted": CLASS_NAMES[int(y_pred[i])],
                "confidence": float(probs[i].max()),
            })
    errors.sort(key=lambda d: d["confidence"], reverse=True)
    return errors[:top_k]


def evaluate_all() -> Dict:
    """Evaluate naive, classical, and deep models; persist metrics + plots."""
    from naive_model import NaiveBaseline
    from classical_model import ClassicalModel
    from build_features import load_feature_matrices

    paths, y_test = get_split("test")
    results: Dict = {"primary_metric": "macro_f1", "models": {}}

    # 1) Naive baseline
    naive = NaiveBaseline.load()
    naive_pred = naive.predict(len(y_test))
    results["models"]["naive"] = _metrics_block(y_test, naive_pred)
    _plot_confusion(y_test, naive_pred, "Naive baseline", "confusion_naive.png")

    # 2) Classical model
    feats = load_feature_matrices()
    classical = ClassicalModel.load()
    cl_pred = classical.predict_proba_features(feats["X_test"]).argmax(axis=1)
    results["models"]["classical"] = _metrics_block(y_test, cl_pred)
    _plot_confusion(y_test, cl_pred, "Classical RF (colour+HOG)", "confusion_classical.png")

    # 3) Deep model
    y_true, deep_pred, probs, dpaths = _deep_test_predictions()
    results["models"]["deep"] = _metrics_block(y_true, deep_pred)
    _plot_confusion(y_true, deep_pred, "Deep MobileNetV3-Small", "confusion_deep.png")

    # Error analysis on the deployed (deep) model.
    errors = _error_analysis(y_true, deep_pred, probs, dpaths)
    (OUTPUTS_DIR / "error_analysis.json").write_text(json.dumps(errors, indent=2))

    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"[evaluate] wrote {METRICS_PATH}")
    for name, block in results["models"].items():
        print(f"  {name:10s} acc={block['accuracy']:.3f}  macroF1={block['macro_f1']:.3f}")
    return results


if __name__ == "__main__":
    evaluate_all()
