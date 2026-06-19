"""EllipticEnvelope detector for anomaly detection."""

from sklearn.covariance import EllipticEnvelope

from .base import BaseDetector


class EllipticEnvelopeDetector(BaseDetector):
    """Wrapper around sklearn's EllipticEnvelope with anomaly-score output."""

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        support_fraction: float | None = None,
        assume_centered: bool = False,
        random_state: int | None = None,
    ) -> None:
        self.estimator = EllipticEnvelope(
            contamination=contamination,
            support_fraction=support_fraction,
            assume_centered=assume_centered,
            random_state=random_state,
        )

    def fit(self, features: object, labels: object | None = None) -> "EllipticEnvelopeDetector":
        self.estimator.fit(features)
        return self

    def score_samples(self, features: object) -> object:
        return -self.estimator.decision_function(features)