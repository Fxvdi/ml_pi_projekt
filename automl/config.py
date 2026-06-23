"""Configuration objects for the AutoML pipeline."""

from dataclasses import dataclass


@dataclass(slots=True)
class AutoMLConfig:
    """Top-level configuration for search, evaluation, and orchestration."""

    metric: str = "pr_auc"
    max_trials: int = 25
    random_state: int | None = None
    contamination: float = 0.05
    resource_fractions: tuple[float, ...] = (0.25, 0.5, 1.0)
    reduction_factor: int = 3
    hyperband_min_resource_fraction: float = 0.125
    parameter_budget_levels: int = 3