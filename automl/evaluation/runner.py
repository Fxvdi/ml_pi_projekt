"""End-to-end evaluation of a detector on a dataset."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..data.tep import TEPDataset
from ..detectors.base import BaseDetector
from .protocol import evaluate_score_arrays, score_in_batches


@dataclass(slots=True)
class EvaluationResult:
    """Metrics collected for a single detector run."""

    metrics: dict[str, float]
    train_time_seconds: float
    threshold: float | None


def evaluate_detector(
    detector: BaseDetector,
    train_dataset: TEPDataset,
    test_dataset: TEPDataset,
    *,
    contamination: float = 0.05,
    threshold: float | None = None,
) -> EvaluationResult:
    train_features = np.asarray(train_dataset.features)
    test_features = np.asarray(test_dataset.features)

    start_time = perf_counter()
    detector.fit(train_features, train_dataset.labels)
    train_time_seconds = perf_counter() - start_time

    test_scores = _score_in_batches(detector, test_features)
    metrics: dict[str, float] = {"train_time_seconds": train_time_seconds}
    train_scores = _score_in_batches(detector, train_features)
    evaluation_metrics, resolved_threshold = evaluate_score_arrays(
        test_dataset.labels,
        train_scores,
        test_scores,
        contamination=contamination,
        threshold=threshold,
    )
    metrics.update(evaluation_metrics)

    return EvaluationResult(metrics=metrics, train_time_seconds=train_time_seconds, threshold=resolved_threshold)