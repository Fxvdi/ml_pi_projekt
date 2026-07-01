"""Adapter for sklearn-style anomaly detectors."""

from typing import Any

from .base import BaseDetector


class SklearnDetectorAdapter(BaseDetector):
    """Wrap estimators exposing fit and anomaly scoring methods."""

    def __init__(self, estimator: Any) -> None:
        self.estimator = estimator

    def fit(self, features: object, labels: object | None = None) -> "SklearnDetectorAdapter":
        if labels is None:
            self.estimator.fit(features)
        else:
            self.estimator.fit(features, labels)
        return self

    def score_samples(self, features: object) -> object:
        if hasattr(self.estimator, "score_samples"):
            return self.estimator.score_samples(features)
        if hasattr(self.estimator, "decision_function"):
            return self.estimator.decision_function(features)
        raise AttributeError("Estimator must implement score_samples or decision_function.")