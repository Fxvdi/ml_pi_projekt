"""Human-readable formatting for AutoML run results."""

from __future__ import annotations

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