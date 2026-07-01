"""Adapter for PyOD anomaly detectors."""

from __future__ import annotations

from importlib import import_module
from typing import Any

from .base import BaseDetector


class PyODDetectorAdapter(BaseDetector):
    """Wrap a PyOD detector so it matches the project detector interface."""

    def __init__(self, estimator: Any, *, invert_scores: bool = False) -> None:
        self.estimator = estimator
        self.invert_scores = invert_scores

    @classmethod
    def from_class_path(
        cls,
        module_path: str,
        class_name: str,
        /,
        *args: object,
        invert_scores: bool = False,
        **kwargs: object,
    ) -> "PyODDetectorAdapter":
        """Create an adapter from a fully qualified PyOD class path."""

        module = import_module(module_path)
        estimator_class = getattr(module, class_name)
        return cls(estimator_class(*args, **kwargs), invert_scores=invert_scores)

    def fit(self, features: object, labels: object | None = None) -> "PyODDetectorAdapter":
        if labels is None:
            self.estimator.fit(features)
        else:
            self.estimator.fit(features, labels)
        return self

    def score_samples(self, features: object) -> object:
        if hasattr(self.estimator, "decision_function"):
            scores = self.estimator.decision_function(features)
        elif hasattr(self.estimator, "score_samples"):
            scores = self.estimator.score_samples(features)
        else:
            raise AttributeError("PyOD estimator must expose decision_function or score_samples.")

        return -scores if self.invert_scores else scores