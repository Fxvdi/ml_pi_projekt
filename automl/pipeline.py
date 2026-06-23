"""High-level orchestration for AutoML anomaly detection."""

from dataclasses import dataclass, replace
import math
from pathlib import Path
from typing import Any

from .config import AutoMLConfig
from .data.tep import TEPSplits, load_tep_splits
from .registry import DetectorRegistry, build_default_registry
from .evaluation.runner import EvaluationResult, evaluate_detector
from .search.random_search import RandomSearch
from .search.space import build_budgeted_search_spaces, build_default_search_space


@dataclass(slots=True)
class AutoMLResult:
    """Result of a completed AutoML run."""

    detector_name: str
    parameters: dict[str, Any]
    metrics: dict[str, float]
    score_metric: str = ""
    score_value: float | None = None
    strategy_name: str = ""
    evaluated_candidates: int = 0
    parameter_budget_level: int | None = None


class AutoMLPipeline:
    """Wire together registry, search, and evaluation components."""

    def __init__(self, config: AutoMLConfig, registry: DetectorRegistry | None = None) -> None:
        self.config = config
        self.registry = registry or build_default_registry()

    def _evaluate_candidate(
        self,
        detector_name: str,
        parameters: dict[str, Any],
        train_dataset: Any,
        test_dataset: Any,
        parameter_budget_level: int | None = None,
    ) -> tuple[float, AutoMLResult] | None:
        detector = self.registry.create(detector_name, **parameters)
        evaluation: EvaluationResult = evaluate_detector(
            detector=detector,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            contamination=self.config.contamination,
        )

        score = evaluation.metrics.get(self.config.metric)
        if score is None:
            return None

        return score, AutoMLResult(
            detector_name=detector_name,
            parameters=parameters,
            metrics=evaluation.metrics,
            score_metric=self.config.metric,
            score_value=score,
            parameter_budget_level=parameter_budget_level,
        )

    def _finalize_result(self, result: AutoMLResult, *, strategy_name: str, evaluated_candidates: int) -> AutoMLResult:
        return replace(result, strategy_name=strategy_name, evaluated_candidates=evaluated_candidates)

    def _evaluate_candidate_pool(
        self,
        candidates: list[dict[str, object]],
        dataset: TEPSplits,
        resource_fractions: list[float],
        parameter_budget_level: int | None = None,
    ) -> AutoMLResult | None:
        survivors = candidates
        best_result: AutoMLResult | None = None

        for fraction in resource_fractions:
            train_subset = dataset.training_dataset().subset(fraction, random_state=self.config.random_state)
            scored_candidates: list[tuple[float, dict[str, object], AutoMLResult]] = []

            for candidate in survivors:
                detector_name = str(candidate["detector"])
                parameters = dict(candidate["parameters"])
                evaluation_result = self._evaluate_candidate(
                    detector_name,
                    parameters,
                    train_subset,
                    dataset.evaluation_dataset(),
                    parameter_budget_level=parameter_budget_level,
                )
                if evaluation_result is None:
                    continue

                score, result = evaluation_result
                scored_candidates.append((score, candidate, result))

            if not scored_candidates:
                continue

            scored_candidates.sort(key=lambda item: item[0], reverse=True)
            best_result = scored_candidates[0][2]

            if fraction < 1.0 and len(scored_candidates) > 1:
                keep_count = max(1, len(scored_candidates) // max(2, self.config.reduction_factor))
                survivors = [candidate for _, candidate, _ in scored_candidates[:keep_count]]
            else:
                survivors = [candidate for _, candidate, _ in scored_candidates]

        return best_result

    def run(self, dataset: TEPSplits, detector_names: list[str] | None = None) -> AutoMLResult:
        selected_detectors = detector_names or self.registry.names()
        best_result: AutoMLResult | None = None
        best_score = float("-inf")

        for detector_name in selected_detectors:
            evaluation_result = self._evaluate_candidate(
                detector_name,
                {
                    "contamination": self.config.contamination,
                    "random_state": self.config.random_state,
                },
                dataset.training_dataset(),
                dataset.evaluation_dataset(),
            )

            if evaluation_result is None:
                continue

            score, candidate = evaluation_result

            if score > best_score:
                best_score = score
                best_result = candidate

        if best_result is None:
            raise ValueError(f"No detector produced the requested metric: {self.config.metric}")

        return self._finalize_result(best_result, strategy_name="minimal", evaluated_candidates=len(selected_detectors))

    def run_random_search(self, dataset: TEPSplits, detector_names: list[str] | None = None) -> AutoMLResult:
        """Search detector/parameter combinations with Random Search and return the best result."""

        selected_detectors = detector_names or self.registry.names()
        search_space = build_default_search_space(random_state=self.config.random_state)
        search = RandomSearch(parameter_space=search_space, random_state=self.config.random_state)
        candidates = search.suggest(selected_detectors, self.config.max_trials)

        best_result: AutoMLResult | None = None
        best_score = float("-inf")

        for candidate in candidates:
            detector_name = str(candidate["detector"])
            parameters = dict(candidate["parameters"])
            evaluation_result = self._evaluate_candidate(
                detector_name,
                parameters,
                dataset.training_dataset(),
                dataset.evaluation_dataset(),
            )
            if evaluation_result is None:
                continue

            score, result = evaluation_result

            if score > best_score:
                best_score = score
                best_result = result

        if best_result is None:
            raise ValueError(f"No detector produced the requested metric: {self.config.metric}")

        return self._finalize_result(best_result, strategy_name="random_search", evaluated_candidates=len(candidates))

    def run_successive_halving(self, dataset: TEPSplits, detector_names: list[str] | None = None) -> AutoMLResult:
        """Run successive halving over randomly sampled detector candidates."""

        selected_detectors = detector_names or self.registry.names()
        budgeted_spaces = build_budgeted_search_spaces(
            random_state=self.config.random_state,
            levels=max(1, self.config.parameter_budget_levels),
        )
        search_space = budgeted_spaces[0]
        search = RandomSearch(parameter_space=search_space, random_state=self.config.random_state)
        candidates = search.suggest(selected_detectors, self.config.max_trials)

        resource_fractions = sorted({fraction for fraction in self.config.resource_fractions if 0 < fraction <= 1})
        if not resource_fractions or resource_fractions[-1] < 1.0:
            resource_fractions.append(1.0)

        result = self._evaluate_candidate_pool(candidates, dataset, resource_fractions, parameter_budget_level=0)
        if result is None:
            raise ValueError(f"No detector produced the requested metric: {self.config.metric}")

        return self._finalize_result(result, strategy_name="successive_halving", evaluated_candidates=len(candidates))

    def run_hyperband(self, dataset: TEPSplits, detector_names: list[str] | None = None) -> AutoMLResult:
        """Run a small Hyperband-style search over successive-halving brackets."""

        selected_detectors = detector_names or self.registry.names()
        budgeted_spaces = build_budgeted_search_spaces(
            random_state=self.config.random_state,
            levels=max(1, self.config.parameter_budget_levels),
        )

        eta = max(2, self.config.reduction_factor)
        min_resource_fraction = min(max(self.config.hyperband_min_resource_fraction, 0.01), 1.0)
        s_max = max(0, int(math.floor(math.log(1.0 / min_resource_fraction, eta))))

        best_result: AutoMLResult | None = None
        best_score = float("-inf")
        evaluated_candidates = 0

        for bracket in range(s_max, -1, -1):
            bracket_trial_count = min(
                self.config.max_trials,
                max(1, int(math.ceil(((s_max + 1) / (bracket + 1)) * (eta**bracket)))),
            )
            initial_fraction = min(1.0, min_resource_fraction * (eta ** (s_max - bracket)))
            resource_fractions = [min(1.0, initial_fraction * (eta**step)) for step in range(bracket + 1)]
            parameter_budget_level = min(bracket, len(budgeted_spaces) - 1)
            search = RandomSearch(parameter_space=budgeted_spaces[parameter_budget_level], random_state=self.config.random_state)

            candidates = search.suggest(selected_detectors, bracket_trial_count)
            evaluated_candidates += len(candidates)
            result = self._evaluate_candidate_pool(
                candidates,
                dataset,
                resource_fractions,
                parameter_budget_level=parameter_budget_level,
            )
            if result is None:
                continue

            score = result.metrics.get(self.config.metric)
            if score is None:
                continue

            if score > best_score:
                best_score = score
                best_result = result

        if best_result is None:
            raise ValueError(f"No detector produced the requested metric: {self.config.metric}")

        return self._finalize_result(best_result, strategy_name="hyperband", evaluated_candidates=evaluated_candidates)


def run_minimal_workflow(
    data_dir: str | Path,
    config: AutoMLConfig | None = None,
    detector_name: str = "isolation_forest",
) -> AutoMLResult:
    """Load the TEP splits, train the baseline detector, and return metrics."""

    resolved_config = config or AutoMLConfig()
    pipeline = AutoMLPipeline(resolved_config)
    splits = load_tep_splits(data_dir)
    return pipeline.run(splits, detector_names=[detector_name])


def run_comparison_workflow(
    data_dir: str | Path,
    detector_names: list[str] | None = None,
    config: AutoMLConfig | None = None,
) -> AutoMLResult:
    """Run the pipeline on an explicit list of detectors, or all registered ones."""

    resolved_config = config or AutoMLConfig()
    pipeline = AutoMLPipeline(resolved_config)
    splits = load_tep_splits(data_dir)
    return pipeline.run(splits, detector_names=detector_names)


def run_random_search_workflow(
    data_dir: str | Path,
    detector_names: list[str] | None = None,
    config: AutoMLConfig | None = None,
) -> AutoMLResult:
    """Run the pipeline with Random Search over detector parameters."""

    resolved_config = config or AutoMLConfig()
    pipeline = AutoMLPipeline(resolved_config)
    splits = load_tep_splits(data_dir)
    return pipeline.run_random_search(splits, detector_names=detector_names)


def run_successive_halving_workflow(
    data_dir: str | Path,
    detector_names: list[str] | None = None,
    config: AutoMLConfig | None = None,
) -> AutoMLResult:
    """Run successive halving over detector candidates."""

    resolved_config = config or AutoMLConfig()
    pipeline = AutoMLPipeline(resolved_config)
    splits = load_tep_splits(data_dir)
    return pipeline.run_successive_halving(splits, detector_names=detector_names)


def run_hyperband_workflow(
    data_dir: str | Path,
    detector_names: list[str] | None = None,
    config: AutoMLConfig | None = None,
) -> AutoMLResult:
    """Run Hyperband over detector candidates."""

    resolved_config = config or AutoMLConfig()
    pipeline = AutoMLPipeline(resolved_config)
    splits = load_tep_splits(data_dir)
    return pipeline.run_hyperband(splits, detector_names=detector_names)


def run_search_workflow(
    data_dir: str | Path,
    detector_names: list[str] | None = None,
    config: AutoMLConfig | None = None,
) -> AutoMLResult:
    """Run the default random-search search mode over all or selected detectors."""

    return run_random_search_workflow(data_dir, detector_names=detector_names, config=config)