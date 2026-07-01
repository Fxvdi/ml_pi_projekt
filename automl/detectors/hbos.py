"""Histogram-based Outlier Score (HBOS) detector."""

from __future__ import annotations

import math

import numpy as np
from sklearn.preprocessing import StandardScaler

from .base import BaseDetector


class HBOSDetector(BaseDetector):
    """Simple HBOS implementation with optional standard scaling."""

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        n_bins: int = 10,
        alpha: float = 1e-12,
        use_scaler: bool = True,
    ) -> None:
        self.contamination = contamination
        self.n_bins = n_bins
        self.alpha = alpha
        self.use_scaler = use_scaler
        self.scaler = StandardScaler() if use_scaler else None
        self._bin_edges: list[np.ndarray] = []
        self._bin_densities: list[np.ndarray] = []

    def fit(self, features: object, labels: object | None = None) -> "HBOSDetector":
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if self.scaler is not None:
            values = self.scaler.fit_transform(values)

        self._bin_edges = []
        self._bin_densities = []

        for column in values.T:
            column_values = np.asarray(column, dtype=float)
            if np.allclose(column_values, column_values[0]):
                center = float(column_values[0])
                edges = np.array([center - 0.5, center + 0.5], dtype=float)
                densities = np.array([1.0], dtype=float)
            else:
                edges = np.histogram_bin_edges(column_values, bins=self.n_bins)
                counts, edges = np.histogram(column_values, bins=edges, density=False)
                widths = np.diff(edges)
                total = float(column_values.shape[0])
                densities = counts.astype(float) / np.maximum(total * widths, self.alpha)

            self._bin_edges.append(edges)
            self._bin_densities.append(np.maximum(densities, self.alpha))

        return self

    def score_samples(self, features: object) -> object:
        values = np.asarray(features, dtype=float)
        if values.ndim == 1:
            values = values.reshape(-1, 1)

        if self.scaler is not None:
            values = self.scaler.transform(values)

        scores = np.zeros(values.shape[0], dtype=float)
        for column_index, column in enumerate(values.T):
            edges = self._bin_edges[column_index]
            densities = self._bin_densities[column_index]
            bin_indices = np.searchsorted(edges[1:-1], column, side="right")
            bin_indices = np.clip(bin_indices, 0, len(densities) - 1)
            column_densities = densities[bin_indices]
            scores += -np.log(np.maximum(column_densities, self.alpha))

        return scores / max(values.shape[1], 1)