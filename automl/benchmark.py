"""Benchmark orchestration for repeated AutoML comparison runs."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Callable, Sequence

from .config import AutoMLConfig
from .persistence import build_benchmark_metadata, result_to_record
from .pipeline import AutoMLResult
from .registry import DetectorRegistry, build_combined_registry, build_default_registry, build_pyod_registry


_DEFAULT_SPLIT_PLAN: dict[str, Any] = {
    "training": "train_fault_free",
    "validation": "subset(train_fault_free)",
    "test": ["test_fault_free", "test_faulty"],
    "faulty_training": "train_faulty (explorative only)",
}


@dataclass(slots=True)
class BenchmarkCase:
    """A single benchmark execution and its metadata."""

    strategy_name: str
    registry_name: str
    seed: int | None
    detector_names: list[str]
    result: AutoMLResult
    metadata: dict[str, Any]

    def to_record(self) -> dict[str, Any]:
        record = result_to_record(self.result)
        record["metadata"] = self.metadata
        record["benchmark"] = {
            "strategy_name": self.strategy_name,
            "registry_name": self.registry_name,
            "seed": self.seed,
            "detector_names": list(self.detector_names),
        }
        return record


@dataclass(slots=True)
class BenchmarkSuite:
    """Collection of benchmark cases that can be persisted together."""

    cases: list[BenchmarkCase]

    def records(self) -> list[dict[str, Any]]:
        return [case.to_record() for case in self.cases]

    def save_jsonl(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with output_path.open("w", encoding="utf-8") as handle:
            for record in self.records():
                handle.write(json.dumps(record, ensure_ascii=False))
                handle.write("\n")

        return output_path

    def best_case(self) -> BenchmarkCase:
        scored_cases = [case for case in self.cases if case.result.score_value is not None]
        if not scored_cases:
            raise ValueError("No benchmark case contains a score value.")
        return max(scored_cases, key=lambda case: case.result.score_value or float("-inf"))

    def aggregate(self) -> list["BenchmarkAggregate"]:
        grouped_cases: dict[tuple[str, str, tuple[str, ...]], list[BenchmarkCase]] = {}

        for case in self.cases:
            key = (case.registry_name, case.strategy_name, tuple(case.detector_names))
            grouped_cases.setdefault(key, []).append(case)

        aggregates: list[BenchmarkAggregate] = []
        for (registry_name, strategy_name, detector_names), cases in grouped_cases.items():
            aggregates.append(
                BenchmarkAggregate(
                    registry_name=registry_name,
                    strategy_name=strategy_name,
                    detector_names=detector_names,
                    seeds=tuple(case.seed for case in cases),
                    cases=cases,
                )
            )

        return sorted(aggregates, key=lambda item: (item.registry_name, item.strategy_name, item.detector_names))

    def aggregate_records(self) -> list[dict[str, Any]]:
        return [aggregate.to_record() for aggregate in self.aggregate()]

    def save_aggregate_json(self, path: str | Path) -> Path:
        output_path = Path(path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(self.aggregate_records(), indent=2, ensure_ascii=False), encoding="utf-8")
        return output_path


@dataclass(slots=True)
class BenchmarkAggregate:
    """Aggregated results for repeated benchmark runs."""

    registry_name: str
    strategy_name: str
    detector_names: tuple[str, ...]
    seeds: tuple[int | None, ...]
    cases: list[BenchmarkCase]

    def metric_names(self) -> list[str]:
        return sorted({metric_name for case in self.cases for metric_name in case.result.metrics})

    def selected_detectors(self) -> list[str]:
        return sorted({case.result.detector_name for case in self.cases})

    def metric_values(self, metric_name: str) -> list[float]:
        return [float(case.result.metrics[metric_name]) for case in self.cases if metric_name in case.result.metrics]

    def metric_summary(self, metric_name: str) -> dict[str, float | int | None]:
        values = self.metric_values(metric_name)
        if not values:
            return {"mean": None, "std": None, "count": 0}

        mean_value = fmean(values)
        std_value = pstdev(values) if len(values) > 1 else 0.0
        return {"mean": mean_value, "std": std_value, "count": len(values)}

    def to_record(self) -> dict[str, Any]:
        return {
            "registry_name": self.registry_name,
            "strategy_name": self.strategy_name,
            "detector_names": list(self.detector_names),
            "seeds": [seed for seed in self.seeds],
            "runs": len(self.cases),
            "selected_detectors": self.selected_detectors(),
            "metrics": {metric_name: self.metric_summary(metric_name) for metric_name in self.metric_names()},
        }

    def best_case(self) -> BenchmarkCase:
        scored_cases = [case for case in self.cases if case.result.score_value is not None]
        if not scored_cases:
            raise ValueError("No benchmark case contains a score value.")
        return max(scored_cases, key=lambda case: case.result.score_value or float("-inf"))


class BenchmarkRunner:
    """Run the same benchmark protocol across registries, strategies, and seeds."""

    def __init__(
        self,
        data_dir: str | Path,
        config: AutoMLConfig | None = None,
        registry_factories: dict[str, Callable[[], DetectorRegistry]] | None = None,
        split_plan: dict[str, Any] | None = None,
    ) -> None:
        self.data_dir = Path(data_dir)
        self.config = config or AutoMLConfig()
        self.registry_factories = registry_factories or {
            "default": build_default_registry,
            "pyod": build_pyod_registry,
            "all": build_combined_registry,
        }
        self.split_plan = split_plan or dict(_DEFAULT_SPLIT_PLAN)

    def registry_names(self) -> list[str]:
        return sorted(self.registry_factories)

    def _registry_for_name(self, registry_name: str) -> DetectorRegistry:
        try:
            factory = self.registry_factories[registry_name]
        except KeyError as exc:
            available = ", ".join(self.registry_names())
            raise KeyError(f"Unknown registry: {registry_name}. Available registries: {available}") from exc
        return factory()

    def _detector_names_for_strategy(
        self,
        strategy_name: str,
        registry: DetectorRegistry,
        detector_names: Sequence[str] | None,
    ) -> list[str] | None:
        if detector_names is None:
            if strategy_name == "minimal":
                return [registry.names()[0]] if registry.names() else []
            return None
        return list(detector_names)

    def _run_strategy(
        self,
        strategy_name: str,
        registry: DetectorRegistry,
        detector_names: Sequence[str] | None,
        config: AutoMLConfig,
    ) -> AutoMLResult:
        from .pipeline import (
            run_comparison_workflow,
            run_hyperband_workflow,
            run_minimal_workflow,
            run_random_search_workflow,
            run_successive_halving_workflow,
        )

        if strategy_name == "minimal":
            detector_name = detector_names[0] if detector_names else registry.names()[0]
            return run_minimal_workflow(self.data_dir, config=config, detector_name=detector_name, registry=registry)
        if strategy_name in {"search", "random_search"}:
            return run_random_search_workflow(self.data_dir, detector_names=list(detector_names) if detector_names else None, config=config, registry=registry)
        if strategy_name == "compare":
            return run_comparison_workflow(self.data_dir, detector_names=list(detector_names) if detector_names else None, config=config, registry=registry)
        if strategy_name == "successive_halving":
            return run_successive_halving_workflow(self.data_dir, detector_names=list(detector_names) if detector_names else None, config=config, registry=registry)
        if strategy_name == "hyperband":
            return run_hyperband_workflow(self.data_dir, detector_names=list(detector_names) if detector_names else None, config=config, registry=registry)
        raise ValueError(f"Unknown strategy: {strategy_name}")

    def run_case(
        self,
        *,
        strategy_name: str,
        registry_name: str,
        seed: int | None = None,
        detector_names: Sequence[str] | None = None,
    ) -> BenchmarkCase:
        registry = self._registry_for_name(registry_name)
        resolved_config = replace(self.config, random_state=seed)
        selected_detectors = self._detector_names_for_strategy(strategy_name, registry, detector_names)
        result = self._run_strategy(strategy_name, registry, selected_detectors, resolved_config)

        benchmark_metadata = build_benchmark_metadata(
            data_dir=self.data_dir,
            strategy_name=result.strategy_name or strategy_name,
            random_state=seed,
            validation_fraction=resolved_config.validation_fraction,
            split_plan=self.split_plan,
            detector_names=[result.detector_name] if selected_detectors is None else selected_detectors,
            extra={
                "registry_name": registry_name,
                "strategy_name": strategy_name,
                "seed": seed,
                "selected_detector": result.detector_name,
            },
        )

        return BenchmarkCase(
            strategy_name=strategy_name,
            registry_name=registry_name,
            seed=seed,
            detector_names=[] if selected_detectors is None else list(selected_detectors),
            result=result,
            metadata=benchmark_metadata,
        )

    def run_suite(
        self,
        *,
        strategies: Sequence[str],
        registry_names: Sequence[str] | None = None,
        seeds: Sequence[int | None] = (None,),
        detector_names_by_registry: dict[str, Sequence[str] | None] | None = None,
    ) -> BenchmarkSuite:
        selected_registry_names = list(registry_names) if registry_names is not None else self.registry_names()
        cases: list[BenchmarkCase] = []

        for registry_name in selected_registry_names:
            registry = self._registry_for_name(registry_name)
            base_detectors = None
            if detector_names_by_registry is not None:
                base_detectors = detector_names_by_registry.get(registry_name)

            for seed in seeds:
                resolved_config = replace(self.config, random_state=seed)
                for strategy_name in strategies:
                    selected_detectors = self._detector_names_for_strategy(strategy_name, registry, base_detectors)
                    result = self._run_strategy(strategy_name, registry, selected_detectors, resolved_config)
                    benchmark_metadata = build_benchmark_metadata(
                        data_dir=self.data_dir,
                        strategy_name=result.strategy_name or strategy_name,
                        random_state=seed,
                        validation_fraction=resolved_config.validation_fraction,
                        split_plan=self.split_plan,
                        detector_names=[result.detector_name] if selected_detectors is None else selected_detectors,
                        extra={
                            "registry_name": registry_name,
                            "strategy_name": strategy_name,
                            "seed": seed,
                            "selected_detector": result.detector_name,
                        },
                    )
                    cases.append(
                        BenchmarkCase(
                            strategy_name=strategy_name,
                            registry_name=registry_name,
                            seed=seed,
                            detector_names=[] if selected_detectors is None else list(selected_detectors),
                            result=result,
                            metadata=benchmark_metadata,
                        )
                    )

        return BenchmarkSuite(cases=cases)