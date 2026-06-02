"""Classical (non-deep) ML model (Approach 2 of 3).

Pipeline: hand-crafted colour + HOG features (see ``build_features.py``) ->
standardization -> Random Forest classifier. Random Forests are a strong,
low-variance baseline for tabular feature vectors, handle the mild class
imbalance via ``class_weight='balanced'``, and need no GPU.

We grid-search a small set of forest hyperparameters on the validation split.

scikit-learn RandomForest:
https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html
"""

from __future__ import annotations

from typing import Dict, Tuple

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from build_features import extract_features, load_feature_matrices
from config import CLASSIC_MODEL_PATH

PARAM_GRID = [
    {"n_estimators": 300, "max_depth": None, "min_samples_leaf": 1},
    {"n_estimators": 400, "max_depth": 30, "min_samples_leaf": 2},
    {"n_estimators": 500, "max_depth": None, "min_samples_leaf": 1},
]


def _make_pipeline(params: Dict) -> Pipeline:
    return Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=params["n_estimators"],
            max_depth=params["max_depth"],
            min_samples_leaf=params["min_samples_leaf"],
            class_weight="balanced",
            n_jobs=-1,
            random_state=42,
        )),
    ])


class ClassicalModel:
    """Random-Forest-on-handcrafted-features classifier."""

    def __init__(self, pipeline: Pipeline | None = None, best_params: Dict | None = None):
        self.pipeline = pipeline
        self.best_params = best_params or {}

    def fit_with_search(self) -> "ClassicalModel":
        data = load_feature_matrices()
        X_tr, y_tr = data["X_train"], data["y_train"]
        X_val, y_val = data["X_val"], data["y_val"]

        from sklearn.metrics import f1_score

        best_score, best_pipe, best_params = -1.0, None, None
        for params in PARAM_GRID:
            pipe = _make_pipeline(params)
            pipe.fit(X_tr, y_tr)
            val_pred = pipe.predict(X_val)
            score = f1_score(y_val, val_pred, average="macro")
            print(f"[classical] params={params} val_macroF1={score:.4f}")
            if score > best_score:
                best_score, best_pipe, best_params = score, pipe, params

        # Refit best config on train+val for the final model.
        X_all = np.vstack([X_tr, X_val])
        y_all = np.concatenate([y_tr, y_val])
        best_pipe = _make_pipeline(best_params).fit(X_all, y_all)
        self.pipeline, self.best_params = best_pipe, best_params
        print(f"[classical] best params={best_params} (val macroF1={best_score:.4f})")
        return self

    def predict_proba_features(self, X: np.ndarray) -> np.ndarray:
        return self.pipeline.predict_proba(X)

    def predict_proba_image(self, image) -> np.ndarray:
        feats = extract_features(image).reshape(1, -1)
        return self.pipeline.predict_proba(feats)[0]

    def save(self) -> None:
        joblib.dump({"pipeline": self.pipeline, "best_params": self.best_params},
                    CLASSIC_MODEL_PATH, compress=3)

    @staticmethod
    def load() -> "ClassicalModel":
        blob = joblib.load(CLASSIC_MODEL_PATH)
        return ClassicalModel(pipeline=blob["pipeline"], best_params=blob.get("best_params", {}))


def train_classical() -> ClassicalModel:
    """Train, tune, and persist the classical model."""
    model = ClassicalModel().fit_with_search()
    model.save()
    print(f"[classical] saved -> {CLASSIC_MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_classical()
