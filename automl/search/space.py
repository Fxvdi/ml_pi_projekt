"""Default search spaces for interchangeable anomaly detectors."""

from __future__ import annotations

from math import ceil


def build_default_search_space(*, random_state: int | None = None) -> dict[str, dict[str, list[object]]]:
    """Return a small search space for the built-in detectors."""

    return {
        # Robust, fast baseline
        "isolation_forest": {
            "contamination": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
            "n_estimators": [100, 200, 300, 500, 800],
            "max_samples": ["auto", 0.25, 0.5, 0.75, 0.9],
            "max_features": [0.25, 0.5, 0.75, 1.0],
            "bootstrap": [False, True],
            "random_state": [random_state],
        },

        # Distance-based detector, more sensitive to feature scaling
        "local_outlier_factor": {
            "contamination": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
            "n_neighbors": [5, 10, 20, 30, 50, 75, 100],
            "leaf_size": [20, 30, 40, 60],
            "metric": ["minkowski", "euclidean", "manhattan", "chebyshev"],
            "algorithm": ["auto", "ball_tree", "kd_tree"],
        },

        # Kernel-based detector, usually more expensive
        "one_class_svm": {
            "contamination": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
            "nu": [0.001, 0.01, 0.03, 0.05, 0.1, 0.2],
            "kernel": ["rbf", "sigmoid", "linear"],
            "gamma": ["scale", "auto"],
            "degree": [2, 3, 4, 5],
            "coef0": [0.0, 0.1, 0.5, 1.0],
        },

        # Classical robust covariance method
        "elliptic_envelope": {
            "contamination": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
            "support_fraction": [None, 0.4, 0.5, 0.7, 0.9],
            "assume_centered": [False, True],
            "random_state": [random_state],
        },

        # Histogram / density-style methods
        "hbos": {
            "contamination": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
            "n_bins": [5, 10, 20, 30, 50, 75],
            "alpha": [1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4],
            "use_scaler": [True, False],
        },

        "copod": {
            "contamination": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
            "use_scaler": [True, False],
            "tail_smoothing": [1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4],
        },

        "ecod": {
            "contamination": [0.005, 0.01, 0.03, 0.05, 0.1, 0.15],
            "use_scaler": [True, False],
            "tail_smoothing": [1e-12, 1e-10, 1e-9, 1e-8, 1e-6, 1e-4],
        },
    }


def build_budgeted_search_spaces(
    *,
    random_state: int | None = None,
    levels: int = 3,
) -> list[dict[str, dict[str, list[object]]]]:
    """Return progressively larger search spaces for parameter budgeting."""

    if levels < 1:
        raise ValueError("levels must be at least 1")

    base_space = build_default_search_space(random_state=random_state)
    if levels == 1:
        return [base_space]

    budgeted_spaces: list[dict[str, dict[str, list[object]]]] = []
    for level in range(levels):
        fraction = (level + 1) / levels
        budgeted_spaces.append(_limit_search_space(base_space, fraction))

    return budgeted_spaces


def _limit_search_space(
    search_space: dict[str, dict[str, list[object]]],
    fraction: float,
) -> dict[str, dict[str, list[object]]]:
    limited_space: dict[str, dict[str, list[object]]] = {}

    for detector_name, parameters in search_space.items():
        limited_parameters: dict[str, list[object]] = {}
        for parameter_name, values in parameters.items():
            values_list = list(values)
            if len(values_list) <= 1:
                limited_parameters[parameter_name] = values_list
                continue

            keep_count = max(1, min(len(values_list), ceil(len(values_list) * fraction)))
            limited_parameters[parameter_name] = values_list[:keep_count]

        limited_space[detector_name] = limited_parameters

    return limited_space