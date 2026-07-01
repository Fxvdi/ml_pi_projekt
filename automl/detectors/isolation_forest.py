"""Isolation Forest detector for the minimal unsupervised workflow."""

from sklearn.ensemble import IsolationForest

from .base import BaseDetector


class IsolationForestDetector(BaseDetector):
    """Thin wrapper around sklearn's IsolationForest with anomaly-score output."""

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        random_state: int | None = None,
        n_estimators: int = 100,
        max_samples: float | int | str = "auto",
        max_features: float = 1.0,
        bootstrap: bool = False,
    ) -> None:
        self.estimator = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=n_estimators,
            max_samples=max_samples,
            max_features=max_features,
            bootstrap=bootstrap,
        )

    def fit(self, features: object, labels: object | None = None) -> "IsolationForestDetector":
        self.estimator.fit(features)
        return self

    def score_samples(self, features: object) -> object:
        return -self.estimator.score_samples(features)