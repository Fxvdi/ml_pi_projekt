"""Simple random search over detector names and parameter samples."""

from collections.abc import Callable, Sequence
from random import Random

from .base import SearchStrategy


class RandomSearch(SearchStrategy):
    """Generate random candidates from a supplied parameter space."""

    def __init__(self, parameter_space: dict[str, dict[str, Sequence[object] | Callable[[Random], object]]], random_state: int | None = None) -> None:
        self.parameter_space = parameter_space
        self.random = Random(random_state)

    def suggest(self, detector_names: list[str], max_trials: int) -> list[dict[str, object]]:
        candidates: list[dict[str, object]] = []
        for _ in range(max_trials):
            detector_name = self.random.choice(detector_names)
            parameters: dict[str, object] = {}
            for parameter_name, values in self.parameter_space.get(detector_name, {}).items():
                if callable(values):
                    parameters[parameter_name] = values(self.random)
                else:
                    parameters[parameter_name] = self.random.choice(list(values))
            candidates.append({"detector": detector_name, "parameters": parameters})
        return candidates