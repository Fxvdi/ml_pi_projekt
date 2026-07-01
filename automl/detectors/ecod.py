"""Empirical CDF Outlier Detection (ECOD) detector."""

from __future__ import annotations

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector


class ECODDetector(BaseDetector):
    """Simple ECOD-style detector using empirical CDF tail scores."""

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        use_scaler: bool = True,
        tail_smoothing: float = 1e-12,
    ) -> None:
        self.contamination = contamination
        self.use_scaler = use_scaler
        self.tail_smoothing = tail_smoothing
        self.scaler = StandardScaler() if use_scaler else None
        self._sorted_columns: list[np.ndarray] = []
        self._sample_count = 0

    def fit(self, features: object, labels: object | None = None) -> "ECODDetector":
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if self.scaler is not None:
            values = self.scaler.fit_transform(values)

        self._sample_count = values.shape[0]
        self._sorted_columns = [np.sort(column.astype(float)) for column in values.T]
        return self

    def score_samples(self, features: object) -> object:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if self.scaler is not None:
            values = self.scaler.transform(values)

        scores = np.zeros(values.shape[0], dtype=float)
        denominator = float(self._sample_count + 1)

        for column_index, column in enumerate(values.T):
            sorted_column = self._sorted_columns[column_index]
            left_tail = np.searchsorted(sorted_column, column, side="right") / denominator
            right_tail = 1.0 - np.searchsorted(sorted_column, column, side="left") / denominator
            scores += -np.log(np.maximum(left_tail, self.tail_smoothing))
            scores += -np.log(np.maximum(right_tail, self.tail_smoothing))

        return scores / max(values.shape[1] * 2, 1)