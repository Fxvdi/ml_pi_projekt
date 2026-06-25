"""Persistence helpers for AutoML run results."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Sequence

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


def build_benchmark_metadata(
    *,
    data_dir: str | Path,
    strategy_name: str,
    random_state: int | None = None,
    split_plan: dict[str, Any] | None = None,
    detector_names: Sequence[str] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a small metadata block for persisted benchmark runs."""

    metadata: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "data_dir": str(data_dir),
        "strategy_name": strategy_name,
        "random_state": random_state,
        "split_plan": None if split_plan is None else _normalize_value(split_plan),
        "detector_names": None if detector_names is None else list(detector_names),
    }

    if extra:
        metadata.update({str(key): _normalize_value(value) for key, value in extra.items()})

    return metadata


def save_result_json(
    result: AutoMLResult,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write a single run result as formatted JSON."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = result_to_record(result)
    if metadata:
        record["metadata"] = _normalize_value(metadata)
    output_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def append_result_jsonl(
    result: AutoMLResult,
    path: str | Path,
    *,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Append a run result as one JSON object per line."""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record = result_to_record(result)
    if metadata:
        record["metadata"] = _normalize_value(metadata)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
    return output_path


def load_result_json(path: str | Path) -> dict[str, Any]:
    """Load a persisted run result from JSON."""

    return json.loads(Path(path).read_text(encoding="utf-8"))