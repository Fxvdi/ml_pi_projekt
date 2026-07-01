"""Local Outlier Factor detector for the minimal unsupervised workflow."""

from sklearn.neighbors import LocalOutlierFactor

from .base import BaseDetector


class LocalOutlierFactorDetector(BaseDetector):
    """Wrapper around sklearn's LocalOutlierFactor in novelty mode."""

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        n_neighbors: int = 20,
        leaf_size: int = 30,
        metric: str = "minkowski",
        algorithm: str = "auto",
    ) -> None:
        self.estimator = LocalOutlierFactor(
            contamination=contamination,
            n_neighbors=n_neighbors,
            leaf_size=leaf_size,
            metric=metric,
            algorithm=algorithm,
            novelty=True,
        )

    def fit(self, features: object, labels: object | None = None) -> "LocalOutlierFactorDetector":
        self.estimator.fit(features)
        return self

    def score_samples(self, features: object) -> object:
        return -self.estimator.score_samples(features)