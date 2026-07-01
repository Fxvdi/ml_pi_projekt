"""Metrics for anomaly-detection model comparison."""

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def pr_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(average_precision_score(y_true, y_scores))


def roc_auc(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    return float(roc_auc_score(y_true, y_scores))


def f1_from_scores(y_true: np.ndarray, y_scores: np.ndarray, threshold: float) -> float:
    y_pred = (y_scores >= threshold).astype(int)
    return float(f1_score(y_true, y_pred))