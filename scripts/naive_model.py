"""Naive baseline model (Approach 1 of 3).

A trivial reference point that ignores image content entirely and predicts based
only on the training-label distribution. Two strategies are supported:

* ``most_frequent`` — always predict the majority class (here: ``paper``).
* ``stratified`` — sample predictions from the training class prior.

Any "real" model must beat this to justify its complexity. We report the
majority-class baseline as the headline naive number.

Uses scikit-learn's DummyClassifier:
https://scikit-learn.org/stable/modules/generated/sklearn.dummy.DummyClassifier.html
"""

from __future__ import annotations

import joblib
import numpy as np
from sklearn.dummy import DummyClassifier

from config import NAIVE_MODEL_PATH, NUM_CLASSES
from data_utils import get_split


class NaiveBaseline:
    """Wraps a DummyClassifier so it shares the project's predict interface."""

    def __init__(self, strategy: str = "most_frequent"):
        self.strategy = strategy
        self.clf = DummyClassifier(strategy=strategy, random_state=42)

    def fit(self) -> "NaiveBaseline":
        # The dummy classifier only needs labels; features are placeholders.
        _, y = get_split("train")
        X = np.zeros((len(y), 1), dtype=np.float32)
        self.clf.fit(X, y)
        return self

    def predict_proba(self, n: int) -> np.ndarray:
        X = np.zeros((n, 1), dtype=np.float32)
        proba = self.clf.predict_proba(X)
        # Align to full class set in case a class is unseen.
        full = np.zeros((n, NUM_CLASSES), dtype=np.float32)
        full[:, self.clf.classes_] = proba
        return full

    def predict(self, n: int) -> np.ndarray:
        return self.predict_proba(n).argmax(axis=1)

    def save(self) -> None:
        joblib.dump(self.clf, NAIVE_MODEL_PATH)

    @staticmethod
    def load() -> "NaiveBaseline":
        obj = NaiveBaseline()
        obj.clf = joblib.load(NAIVE_MODEL_PATH)
        return obj


def train_naive() -> NaiveBaseline:
    """Train and persist the naive majority-class baseline."""
    model = NaiveBaseline(strategy="most_frequent").fit()
    model.save()
    print(f"[naive] saved -> {NAIVE_MODEL_PATH}")
    return model


if __name__ == "__main__":
    train_naive()
