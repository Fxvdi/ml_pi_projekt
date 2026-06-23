"""Persistence helpers for AutoML run results."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .pipeline import AutoMLResult


def _normalize_value(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _normalize_value(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_normalize_value(item) for item in value]
    if isinstance(value, list):
        return [_normalize_value(item) for item in value]
    if isinstance(value, set):
        return [_normalize_value(item) for item in sorted(value, key=str)]
    item_method = getattr(value, "item", None)
    if callable(item_method):
        try:
            return item_method()
        except Exception:
            pass
    return str(value)


def result_to_record(result: AutoMLResult) -> dict[str, Any]:
    """Convert a run result into a JSON-friendly dictionary."""

    return {
        "detector_name": result.detector_name,
        "parameters": _normalize_value(result.parameters),
        "metrics": _normalize_value(result.metrics),
        "score_metric": result.score_metric,
        "score_value": result.score_value,
        "strategy_name": result.strategy_name,
        "evaluated_candidates": result.evaluated_candidates,
        "parameter_budget_level": result.parameter_budget_level,
    }


def save_result_json(result: AutoMLResult, path: str | Path) -> Path:
    """Write a single run result as formatted JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result_to_record(result), indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def append_result_jsonl(result: AutoMLResult, path: str | Path) -> Path:
    """Append a run result as one JSON object per line."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(result_to_record(result), ensure_ascii=False))
        handle.write("\n")
    return output_path


def load_result_json(path: str | Path) -> dict[str, Any]:
    """Load a persisted run result from JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))