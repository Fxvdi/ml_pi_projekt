"""Shared evaluation protocol for anomaly-detection benchmarking."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from .metrics import f1_from_scores, pr_auc, roc_auc


def score_in_batches(scorer: object, features: object, *, batch_size: int = 50000) -> np.ndarray:
    """Score rows in batches to keep memory usage stable on large datasets."""

    values = np.asarray(features)
    if values.ndim == 1:
        values = values.reshape(-1, 1)

    score_function = getattr(scorer, "score_samples", None)
    if score_function is None:
        raise AttributeError("Scorer must implement score_samples.")

    if len(values) <= batch_size:
        return np.asarray(score_function(values), dtype=float)

    batches: list[np.ndarray] = []
    for start_index in range(0, len(values), batch_size):
        batch = values[start_index : start_index + batch_size]
        batches.append(np.asarray(score_function(batch), dtype=float))

    return np.concatenate(batches, axis=0)


def resolve_threshold(train_scores: Sequence[float], *, contamination: float = 0.05, threshold: float | None = None) -> float:
    """Resolve the threshold used for binary metrics."""

    if threshold is not None:
        return float(threshold)

    return float(np.quantile(np.asarray(train_scores, dtype=float), 1.0 - contamination))


def evaluate_score_arrays(
    labels: Sequence[int] | np.ndarray | None,
    train_scores: Sequence[float] | np.ndarray,
    test_scores: Sequence[float] | np.ndarray,
    *,
    contamination: float = 0.05,
    threshold: float | None = None,
) -> tuple[dict[str, float], float]:
    """Evaluate precomputed anomaly scores with the project metrics."""

    metrics: dict[str, float] = {}
    train_scores_array = np.asarray(train_scores, dtype=float)
    test_scores_array = np.asarray(test_scores, dtype=float)
    resolved_threshold = resolve_threshold(train_scores_array, contamination=contamination, threshold=threshold)

    if labels is not None:
        labels_array = np.asarray(labels)
        metrics["pr_auc"] = pr_auc(labels_array, test_scores_array)
        metrics["roc_auc"] = roc_auc(labels_array, test_scores_array)
        metrics["f1"] = f1_from_scores(labels_array, test_scores_array, resolved_threshold)

    return metrics, resolved_threshold