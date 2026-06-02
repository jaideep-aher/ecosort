"""Flask application factory for the EcoSort inference web app.

Serves a single-page UI and a JSON prediction API. The deployed model is the
fine-tuned MobileNetV3 (loaded once at startup); the app performs *inference
only*, never training, per the project requirements.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path

from flask import Flask, jsonify, render_template, request
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))

from config import CLASS_NAMES, METRICS_PATH  # noqa: E402
from model import EcoSortPredictor  # noqa: E402

MAX_UPLOAD_MB = 10


def create_app() -> Flask:
    """Construct and configure the Flask app."""
    app = Flask(
        __name__,
        template_folder=str(Path(__file__).parent / "templates"),
        static_folder=str(Path(__file__).parent / "static"),
    )
    app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

    predictor = EcoSortPredictor()

    @app.route("/")
    def index():
        return render_template("index.html", classes=CLASS_NAMES)

    @app.route("/api/health")
    def health():
        return jsonify({"status": "ok", "classes": CLASS_NAMES})

    @app.route("/api/metrics")
    def metrics():
        if METRICS_PATH.exists():
            return jsonify(json.loads(METRICS_PATH.read_text()))
        return jsonify({"error": "metrics not available"}), 404

    @app.route("/api/predict", methods=["POST"])
    def predict():
        if "image" not in request.files:
            return jsonify({"error": "No image uploaded (field 'image')."}), 400
        file = request.files["image"]
        if not file.filename:
            return jsonify({"error": "Empty filename."}), 400
        try:
            image = Image.open(io.BytesIO(file.read())).convert("RGB")
        except Exception:  # noqa: BLE001
            return jsonify({"error": "Could not read the uploaded image."}), 400

        explain = request.form.get("explain", "true").lower() != "false"
        compare = request.form.get("compare", "false").lower() == "true"
        try:
            result = predictor.predict(image, explain=explain, compare_classical=compare)
        except FileNotFoundError:
            return jsonify({"error": "Model not found. Train the model first (python setup.py)."}), 503
        return jsonify(result)

    # Warm up the model at startup so the first request is fast.
    try:
        predictor.warmup()
    except Exception as exc:  # noqa: BLE001 - allow app to boot for health checks
        app.logger.warning("Model warmup deferred: %s", exc)

    return app


app = create_app()
