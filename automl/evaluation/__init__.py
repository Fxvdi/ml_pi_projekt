"""Evaluation utilities for anomaly detection."""

from .metrics import f1_from_scores, pr_auc, roc_auc
from .protocol import evaluate_score_arrays, resolve_threshold, score_in_batches
from .runner import EvaluationResult, evaluate_detector

__all__ = [
	"EvaluationResult",
	"evaluate_detector",
	"evaluate_score_arrays",
	"f1_from_scores",
	"pr_auc",
	"resolve_threshold",
	"roc_auc",
	"score_in_batches",
]