"""Simple command-line entry point for the AutoML workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from automl.config import AutoMLConfig


def _build_registry(registry_name: str):
    from automl.registry import build_combined_registry, build_default_registry, build_pyod_registry

    if registry_name == "default":
        return build_default_registry()
    if registry_name == "pyod":
        return build_pyod_registry()
    if registry_name == "all":
        return build_combined_registry()
    raise ValueError(f"Unknown registry: {registry_name}")


def _default_detector_for_registry(registry_name: str) -> str:
    if registry_name == "pyod":
        return "pyod_ecod"
    return "isolation_forest"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TEP anomaly-detection AutoML workflow.")
    parser.add_argument(
        "--data-dir",
        default="automl/data",
        help="Path to the folder containing the TEP parquet files.",
    )
    parser.add_argument(
        "--detector",
        default=None,
        help="Detector to run in minimal mode. Defaults to the first detector of the selected registry.",
    )
    parser.add_argument(
        "--registry",
        choices=["default", "pyod", "all"],
        default="default",
        help="Choose which detector registry to use.",
    )
    parser.add_argument(
        "--compare",
        nargs="*",
        help="Compare detectors. If no names are given, all registered detectors are used.",
    )
    parser.add_argument(
        "--strategy",
        choices=["minimal", "compare", "search", "random_search", "successive_halving", "hyperband"],
        default="minimal",
        help="Choose the execution strategy.",
    )
    parser.add_argument(
        "--save-result",
        help="Write the best result as JSON to the given file.",
    )
    parser.add_argument(
        "--append-history",
        help="Append the best result as JSONL to the given file.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        help="Random seed for reproducible benchmarking.",
    )
    parser.add_argument(
        "--validation-fraction",
        type=float,
        default=0.2,
        help="Fraction of the clean training data reserved for validation.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    from automl.persistence import append_result_jsonl, build_benchmark_metadata, save_result_json
    from automl.reporting import format_run_result
    from automl.pipeline import (
        run_comparison_workflow,
        run_hyperband_workflow,
        run_minimal_workflow,
        run_random_search_workflow,
        run_successive_halving_workflow,
    )

    data_dir = Path(args.data_dir)
    config = AutoMLConfig(random_state=args.random_state, validation_fraction=args.validation_fraction)
    registry = _build_registry(args.registry)
    registry_names = registry.names()

    if args.detector is None:
        detector_name = _default_detector_for_registry(args.registry)
    else:
        detector_name = args.detector

    if detector_name not in registry_names:
        parser.error(f"Unknown detector {detector_name!r} for registry {args.registry!r}. Available detectors: {', '.join(registry_names)}")

    requested_compare = args.compare or None
    if requested_compare is not None:
        unknown_detectors = sorted(set(requested_compare) - set(registry_names))
        if unknown_detectors:
            parser.error(
                "Unknown detector names for the selected registry: "
                + ", ".join(unknown_detectors)
                + ". Available detectors: "
                + ", ".join(registry_names)
            )

    if args.strategy in {"search", "random_search"}:
        detector_names = requested_compare
        result = run_random_search_workflow(data_dir, detector_names, config=config, registry=registry)
    elif args.strategy == "successive_halving":
        detector_names = requested_compare
        result = run_successive_halving_workflow(data_dir, detector_names, config=config, registry=registry)
    elif args.strategy == "hyperband":
        detector_names = requested_compare
        result = run_hyperband_workflow(data_dir, detector_names, config=config, registry=registry)
    elif args.compare is not None or args.strategy == "compare":
        detector_names = requested_compare
        result = run_comparison_workflow(data_dir, detector_names, config=config, registry=registry)
    else:
        result = run_minimal_workflow(data_dir, config=config, detector_name=detector_name, registry=registry)

    print(format_run_result(result))

    if requested_compare is not None:
        detector_names = requested_compare
    elif args.strategy == "minimal":
        detector_names = [detector_name]
    else:
        detector_names = registry_names
    benchmark_metadata = build_benchmark_metadata(
        data_dir=data_dir,
        strategy_name=result.strategy_name or args.strategy,
        random_state=args.random_state,
        validation_fraction=args.validation_fraction,
        split_plan={
            "training": "train_fault_free",
            "validation": "subset(train_fault_free)",
            "test": ["test_fault_free", "test_faulty"],
            "faulty_training": "train_faulty (explorative only)",
        },
        detector_names=detector_names,
        extra={
            "selected_detector": detector_name,
            "selected_registry": args.registry,
            "requested_strategy": args.strategy,
        },
    )

    if args.save_result:
        save_result_json(result, args.save_result, metadata=benchmark_metadata)

    if args.append_history:
        append_result_jsonl(result, args.append_history, metadata=benchmark_metadata)


if __name__ == "__main__":
    main()