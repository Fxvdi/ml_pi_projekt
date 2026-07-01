"""Common interface for search strategies."""

from abc import ABC, abstractmethod


class SearchStrategy(ABC):
    """Suggest detector configurations for evaluation."""

    @abstractmethod
    def suggest(self, detector_names: list[str], max_trials: int) -> list[dict[str, object]]:
        """Return candidate configurations to evaluate."""