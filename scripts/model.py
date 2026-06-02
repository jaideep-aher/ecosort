"""Unified model interface + training orchestrator for EcoSort.

This module ties the three modeling approaches together:

* ``train_all()`` trains the naive, classical, and deep models in sequence and is
  invoked by ``setup.py``.
* ``EcoSortPredictor`` is the single inference entry point used by the web app.
  The **deployed** model is the deep MobileNetV3 (best test performance); the
  classical model is also exposed for side-by-side comparison in the report.

Grad-CAM overlay rendering uses a matplotlib 'jet' colormap blended over the
original image and returned as a base64 PNG for the browser.
"""

from __future__ import annotations

import base64
import io
from typing import Dict, List

import numpy as np
from PIL import Image

from config import BIN_GUIDANCE, CLASS_NAMES


def train_all() -> None:
    """Train and persist all three models (naive -> classical -> deep)."""
    from naive_model import train_naive
    from classical_model import train_classical
    from deep_model import train_deep

    print("\n=== [1/3] Naive baseline ===")
    train_naive()
    print("\n=== [2/3] Classical ML (RandomForest on colour+HOG) ===")
    train_classical()
    print("\n=== [3/3] Deep learning (MobileNetV3-Small) ===")
    train_deep()
    print("\nAll models trained.")


def _overlay_heatmap(image: Image.Image, cam: np.ndarray, alpha: float = 0.45) -> str:
    """Blend a Grad-CAM heatmap over the image; return a base64 PNG data URI."""
    import cv2

    rgb = np.asarray(image.convert("RGB"))
    h, w = rgb.shape[:2]
    cam_resized = cv2.resize((cam * 255).astype(np.uint8), (w, h))
    heat = cv2.applyColorMap(cam_resized, cv2.COLORMAP_JET)
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)
    blended = (alpha * heat + (1 - alpha) * rgb).astype(np.uint8)

    buf = io.BytesIO()
    Image.fromarray(blended).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


class EcoSortPredictor:
    """Lazy-loading inference facade for the deployed deep model.

    Designed for the web server: the model is loaded once and reused. The
    classical model is loaded on demand if a comparison is requested.
    """

    def __init__(self) -> None:
        self._deep = None
        self._classical = None

    @property
    def deep(self):
        if self._deep is None:
            from deep_model import DeepClassifier
            self._deep = DeepClassifier.load()
        return self._deep

    @property
    def classical(self):
        if self._classical is None:
            from classical_model import ClassicalModel
            self._classical = ClassicalModel.load()
        return self._classical

    def warmup(self) -> None:
        """Force-load the deployed model (call at server startup)."""
        _ = self.deep

    def predict(self, image: Image.Image, explain: bool = True,
                compare_classical: bool = False) -> Dict:
        """Run inference and return a JSON-serializable result dict."""
        probs = self.deep.predict_proba_image(image)
        ranked = sorted(
            ({"class": CLASS_NAMES[i],
              "probability": float(probs[i]),
              **BIN_GUIDANCE[CLASS_NAMES[i]]} for i in range(len(CLASS_NAMES))),
            key=lambda d: d["probability"],
            reverse=True,
        )
        top = ranked[0]
        result: Dict = {
            "prediction": top["class"],
            "confidence": top["probability"],
            "bin": top["bin"],
            "tip": top["tip"],
            "color": top["color"],
            "ranking": ranked,
        }

        if explain:
            cam, _ = self.deep.grad_cam(image, target_class=CLASS_NAMES.index(top["class"]))
            result["heatmap"] = _overlay_heatmap(image, cam)

        if compare_classical:
            try:
                cprobs = self.classical.predict_proba_image(image)
                ci = int(np.argmax(cprobs))
                result["classical"] = {
                    "prediction": CLASS_NAMES[ci],
                    "confidence": float(cprobs[ci]),
                }
            except Exception as exc:  # noqa: BLE001 - comparison is best-effort
                result["classical"] = {"error": str(exc)}

        return result


if __name__ == "__main__":
    train_all()
