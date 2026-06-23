"""End-to-end evaluation of a detector on a dataset."""

from dataclasses import dataclass
from time import perf_counter

import numpy as np

from ..data.tep import TEPDataset
from ..detectors.base import BaseDetector
from .metrics import f1_from_scores, pr_auc, roc_auc


def _score_in_batches(detector: BaseDetector, features: np.ndarray, *, batch_size: int = 50000) -> np.ndarray:
    """Score rows in batches to keep memory usage stable on large datasets."""

    if features.ndim == 1:
        features = features.reshape(-1, 1)

    if len(features) <= batch_size:
        return np.asarray(detector.score_samples(features), dtype=float)

    batches: list[np.ndarray] = []
    for start_index in range(0, len(features), batch_size):
        batch = features[start_index : start_index + batch_size]
        batches.append(np.asarray(detector.score_samples(batch), dtype=float))

    return np.concatenate(batches, axis=0)


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
    resolved_threshold = threshold

    if resolved_threshold is None:
        train_scores = _score_in_batches(detector, train_features)
        resolved_threshold = float(np.quantile(train_scores, 1.0 - contamination))

    if test_dataset.labels is not None:
        labels = np.asarray(test_dataset.labels)
        metrics["pr_auc"] = pr_auc(labels, test_scores)
        metrics["roc_auc"] = roc_auc(labels, test_scores)
        if resolved_threshold is not None:
            metrics["f1"] = f1_from_scores(labels, test_scores, resolved_threshold)

    return EvaluationResult(metrics=metrics, train_time_seconds=train_time_seconds, threshold=resolved_threshold)