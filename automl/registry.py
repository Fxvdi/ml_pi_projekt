"""Registry for interchangeable anomaly detectors."""

from collections.abc import Callable
import inspect
from importlib import import_module

from .detectors import COPODDetector, ECODDetector, EllipticEnvelopeDetector, HBOSDetector, IsolationForestDetector, LocalOutlierFactorDetector, OneClassSVMDetector, PyODDetectorAdapter


def _filtered_factory(constructor: Callable[..., object]) -> Callable[..., object]:
    """Build a factory that ignores unsupported keyword arguments."""

    signature = inspect.signature(constructor)

    def factory(**kwargs: object) -> object:
        filtered_kwargs = {
            key: value
            for key, value in kwargs.items()
            if key in signature.parameters
        }
        return constructor(**filtered_kwargs)

    return factory


def _pyod_factory(module_path: str, class_name: str, *, invert_scores: bool = False) -> Callable[..., object]:
    """Build a lazily imported PyOD-backed detector factory."""

    def factory(**kwargs: object) -> object:
        try:
            module = import_module(module_path)
        except ImportError as exc:
            raise RuntimeError(
                "PyOD is not installed. Install it to use the PyOD benchmark registry."
            ) from exc

        detector_class = getattr(module, class_name)
        return PyODDetectorAdapter(detector_class(**kwargs), invert_scores=invert_scores)

    return factory


class DetectorRegistry:
    """Map detector names to factories."""

    def __init__(self) -> None:
        self._factories: dict[str, Callable[..., object]] = {}

    def register(self, name: str, factory: Callable[..., object]) -> None:
        self._factories[name] = factory

    def create(self, name: str, **kwargs: object) -> object:
        try:
            factory = self._factories[name]
        except KeyError as exc:
            raise KeyError(f"Unknown detector: {name}") from exc

        return factory(**kwargs)

    def names(self) -> list[str]:
        return sorted(self._factories)

    def update_from(self, other: "DetectorRegistry") -> None:
        for name, factory in other._factories.items():
            self._factories[name] = factory


def build_default_registry() -> DetectorRegistry:
    """Create a registry with the initial interchangeable detectors."""

    registry = DetectorRegistry()
    registry.register("elliptic_envelope", _filtered_factory(EllipticEnvelopeDetector))
    registry.register("hbos", _filtered_factory(HBOSDetector))
    registry.register("copod", _filtered_factory(COPODDetector))
    registry.register("ecod", _filtered_factory(ECODDetector))
    registry.register("isolation_forest", _filtered_factory(IsolationForestDetector))
    registry.register("local_outlier_factor", _filtered_factory(LocalOutlierFactorDetector))
    registry.register("one_class_svm", _filtered_factory(OneClassSVMDetector))
    return registry


def build_pyod_registry() -> DetectorRegistry:
    """Create a registry with a small set of PyOD detectors.

    The import is lazy so the repository stays usable without PyOD installed.
    """

    registry = DetectorRegistry()
    registry.register("pyod_ecod", _pyod_factory("pyod.models.ecod", "ECOD"))
    registry.register("pyod_copod", _pyod_factory("pyod.models.copod", "COPOD"))
    registry.register("pyod_hbos", _pyod_factory("pyod.models.hbos", "HBOS"))
    registry.register("pyod_iforest", _pyod_factory("pyod.models.iforest", "IForest"))
    registry.register("pyod_knn", _pyod_factory("pyod.models.knn", "KNN"))
    registry.register("pyod_pca", _pyod_factory("pyod.models.pca", "PCA"))
    registry.register("pyod_ocsvm", _pyod_factory("pyod.models.ocsvm", "OCSVM"))
    return registry


def build_combined_registry() -> DetectorRegistry:
    """Create a registry that exposes both built-in and PyOD-backed detectors."""

    registry = build_default_registry()
    registry.update_from(build_pyod_registry())
    return registry