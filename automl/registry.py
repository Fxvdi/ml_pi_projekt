"""Registry for interchangeable anomaly detectors."""

from collections.abc import Callable
import inspect

from .detectors import COPODDetector, ECODDetector, EllipticEnvelopeDetector, HBOSDetector, IsolationForestDetector, LocalOutlierFactorDetector, OneClassSVMDetector


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