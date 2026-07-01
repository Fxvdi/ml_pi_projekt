"""One-Class SVM detector for anomaly detection."""

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM

from .base import BaseDetector


class OneClassSVMDetector(BaseDetector):
    """Wrapper around sklearn's OneClassSVM with feature scaling."""

    def __init__(
        self,
        *,
        contamination: float = 0.05,
        nu: float = 0.05,
        kernel: str = "rbf",
        gamma: str | float = "scale",
        degree: int = 3,
        coef0: float = 0.0,
    ) -> None:
        self.contamination = contamination
        self.estimator = make_pipeline(
            StandardScaler(),
            OneClassSVM(
                nu=nu,
                kernel=kernel,
                gamma=gamma,
                degree=degree,
                coef0=coef0,
            ),
        )

    def fit(self, features: object, labels: object | None = None) -> "OneClassSVMDetector":
        self.estimator.fit(features)
        return self

    def score_samples(self, features: object) -> object:
        one_class_svm = self.estimator.named_steps["oneclasssvm"]
        scaled_features = self.estimator.named_steps["standardscaler"].transform(features)
        return -one_class_svm.score_samples(scaled_features)