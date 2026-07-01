"""Human-readable formatting for AutoML run results."""

from __future__ import annotations

from pathlib import Path

from .benchmark import BenchmarkCase, BenchmarkSuite
from .pipeline import AutoMLResult


def _format_float(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def format_run_result(result: AutoMLResult) -> str:
    """Render a compact summary for console output."""

    lines: list[str] = []
    if result.strategy_name:
        lines.append(f"Strategy: {result.strategy_name}")
    if result.evaluated_candidates:
        suffix = "candidate" if result.evaluated_candidates == 1 else "candidates"
        lines.append(f"Evaluated: {result.evaluated_candidates} {suffix}")

    lines.append(f"Best detector: {result.detector_name}")
    if result.score_metric:
        lines.append(f"Selected metric: {result.score_metric} = {_format_float(result.score_value)}")

    if result.parameter_budget_level is not None:
        lines.append(f"Parameter budget level: {result.parameter_budget_level}")

    train_time = result.metrics.get("train_time_seconds")
    if train_time is not None:
        lines.append(f"Train time: {_format_float(train_time)} s")

    metric_names = [name for name in result.metrics if name != "train_time_seconds"]
    if metric_names:
        lines.append("Metrics:")
        for metric_name in sorted(metric_names):
            lines.append(f"  {metric_name}: {_format_float(result.metrics[metric_name])}")

    if result.parameters:
        lines.append("Parameters:")
        for parameter_name in sorted(result.parameters):
            lines.append(f"  {parameter_name}: {result.parameters[parameter_name]}")

    return "\n".join(lines)


def _benchmark_case_metric(case: BenchmarkCase, metric_name: str) -> str:
    value = case.result.metrics.get(metric_name)
    if value is None:
        return "n/a"
    return f"{value:.4f}"


def render_benchmark_report(suite: BenchmarkSuite, *, title: str = "Benchmark Report") -> str:
    """Render a compact Markdown report for a benchmark suite."""

    if not suite.cases:
        return f"# {title}\n\nNo benchmark cases were executed."

    best_case = suite.best_case()
    best_score = "n/a" if best_case.result.score_value is None else f"{best_case.result.score_value:.4f}"
    best_metric = best_case.result.score_metric or "n/a"
    metric_names = sorted(
        {
            metric_name
            for case in suite.cases
            for metric_name in case.result.metrics
            if metric_name != "train_time_seconds"
        }
    )
    metric_names.insert(0, "train_time_seconds")

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"Best case: {best_case.registry_name} / {best_case.strategy_name} / seed={best_case.seed}")
    lines.append(f"Best detector: {best_case.result.detector_name} ({best_metric} = {best_score})")
    lines.append("")
    lines.append("## Cases")
    lines.append("")

    header = ["Registry", "Strategy", "Seed", "Detector", "Score metric", "Score value"] + [
        metric_name for metric_name in metric_names
    ]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for case in suite.cases:
        row = [
            case.registry_name,
            case.strategy_name,
            "n/a" if case.seed is None else str(case.seed),
            case.result.detector_name,
            case.result.score_metric or "n/a",
            "n/a" if case.result.score_value is None else f"{case.result.score_value:.4f}",
        ]
        row.extend(_benchmark_case_metric(case, metric_name) for metric_name in metric_names)
        lines.append("| " + " | ".join(row) + " |")

    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append(f"- Cases executed: {len(suite.cases)}")
    lines.append(f"- Registries: {', '.join(sorted({case.registry_name for case in suite.cases}))}")
    lines.append(f"- Strategies: {', '.join(sorted({case.strategy_name for case in suite.cases}))}")
    lines.append(f"- Seeds: {', '.join('n/a' if case.seed is None else str(case.seed) for case in suite.cases)}")

    return "\n".join(lines)


def save_benchmark_report(suite: BenchmarkSuite, path: str | Path, *, title: str = "Benchmark Report") -> Path:
    """Save the rendered benchmark report as Markdown."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_benchmark_report(suite, title=title), encoding="utf-8")
    return output_path


def _format_mean_std(mean_value: float | None, std_value: float | None) -> str:
    if mean_value is None:
        return "n/a"
    if std_value is None:
        return f"{mean_value:.4f}"
    return f"{mean_value:.4f} ± {std_value:.4f}"


def render_aggregated_benchmark_report(suite: BenchmarkSuite, *, title: str = "Aggregated Benchmark Report") -> str:
    """Render a Markdown report that aggregates repeated runs by registry, strategy, and detector set."""

    aggregates = suite.aggregate()
    if not aggregates:
        return f"# {title}\n\nNo benchmark cases were executed."

    lines: list[str] = [f"# {title}", ""]
    lines.append(f"Total cases: {len(suite.cases)}")
    lines.append(f"Aggregated groups: {len(aggregates)}")
    lines.append("")
    lines.append("## Aggregates")
    lines.append("")

    header = ["Registry", "Strategy", "Detectors", "Runs", "Seeds", "Selected detectors", "Metric summaries"]
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")

    for aggregate in aggregates:
        metric_summaries = []
        for metric_name in aggregate.metric_names():
            stats = aggregate.metric_summary(metric_name)
            metric_summaries.append(
                f"{metric_name}: {_format_mean_std(stats['mean'], stats['std'])} (n={stats['count']})"
            )

        row = [
            aggregate.registry_name,
            aggregate.strategy_name,
            ", ".join(aggregate.detector_names) if aggregate.detector_names else "n/a",
            str(len(aggregate.cases)),
            ", ".join("n/a" if seed is None else str(seed) for seed in aggregate.seeds),
            ", ".join(aggregate.selected_detectors()) or "n/a",
            "<br>".join(metric_summaries) if metric_summaries else "n/a",
        ]
        lines.append("| " + " | ".join(row) + " |")

    best_case = suite.best_case()
    lines.append("")
    lines.append("## Best Case")
    lines.append("")
    lines.append(f"- Registry: {best_case.registry_name}")
    lines.append(f"- Strategy: {best_case.strategy_name}")
    lines.append(f"- Seed: {'n/a' if best_case.seed is None else best_case.seed}")
    lines.append(f"- Detector: {best_case.result.detector_name}")
    lines.append(f"- Score metric: {best_case.result.score_metric or 'n/a'}")
    lines.append(f"- Score value: {_format_float(best_case.result.score_value)}")

    return "\n".join(lines)


def save_aggregated_benchmark_report(
    suite: BenchmarkSuite,
    path: str | Path,
    *,
    title: str = "Aggregated Benchmark Report",
) -> Path:
    """Save the aggregated benchmark report as Markdown."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_aggregated_benchmark_report(suite, title=title), encoding="utf-8")
    return output_path