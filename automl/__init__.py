"""Core package for modular anomaly-detection AutoML."""

from __future__ import annotations

from typing import Any

__all__ = [
	"AutoMLConfig",
	"AutoMLPipeline",
	"AutoMLResult",
	"BenchmarkCase",
	"BenchmarkAggregate",
	"BenchmarkRunner",
	"BenchmarkSuite",
	"render_benchmark_report",
	"save_benchmark_report",
	"render_aggregated_benchmark_report",
	"save_aggregated_benchmark_report",
]


def __getattr__(name: str) -> Any:
	if name == "AutoMLConfig":
		from .config import AutoMLConfig

		return AutoMLConfig
	if name in {"BenchmarkAggregate", "BenchmarkCase", "BenchmarkRunner", "BenchmarkSuite"}:
		from .benchmark import BenchmarkAggregate, BenchmarkCase, BenchmarkRunner, BenchmarkSuite

		return {
			"BenchmarkAggregate": BenchmarkAggregate,
			"BenchmarkCase": BenchmarkCase,
			"BenchmarkRunner": BenchmarkRunner,
			"BenchmarkSuite": BenchmarkSuite,
		}[name]
	if name in {
		"render_aggregated_benchmark_report",
		"render_benchmark_report",
		"save_aggregated_benchmark_report",
		"save_benchmark_report",
	}:
		from .reporting import (
			render_aggregated_benchmark_report,
			render_benchmark_report,
			save_aggregated_benchmark_report,
			save_benchmark_report,
		)

		return {
			"render_aggregated_benchmark_report": render_aggregated_benchmark_report,
			"render_benchmark_report": render_benchmark_report,
			"save_aggregated_benchmark_report": save_aggregated_benchmark_report,
			"save_benchmark_report": save_benchmark_report,
		}[name]
	if name in {"AutoMLPipeline", "AutoMLResult"}:
		from .pipeline import AutoMLPipeline, AutoMLResult

		return {"AutoMLPipeline": AutoMLPipeline, "AutoMLResult": AutoMLResult}[name]
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")