"""Common detector interface for anomaly detection models."""

from abc import ABC, abstractmethod


class BaseDetector(ABC):
    """Shared API for all interchangeable detectors."""

    @abstractmethod
    def fit(self, features: object, labels: object | None = None) -> "BaseDetector":
        """Train the detector."""

    @abstractmethod
    def score_samples(self, features: object) -> object:
        """Return anomaly scores where higher means more anomalous."""