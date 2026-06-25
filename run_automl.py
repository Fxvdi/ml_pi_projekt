"""Simple command-line entry point for the AutoML workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from automl.config import AutoMLConfig
from automl.pipeline import (
    run_comparison_workflow,
    run_hyperband_workflow,
    run_minimal_workflow,
    run_random_search_workflow,
    run_successive_halving_workflow,
)
from automl.persistence import append_result_jsonl, build_benchmark_metadata, save_result_json
from automl.reporting import format_run_result
from automl.registry import build_default_registry


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the TEP anomaly-detection AutoML workflow.")
    parser.add_argument(
        "--data-dir",
        default="automl/data",
        help="Path to the folder containing the TEP parquet files.",
    )
    parser.add_argument(
        "--detector",
        default="isolation_forest",
        choices=build_default_registry().names(),
        help="Detector to run in minimal mode.",
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    config = AutoMLConfig(random_state=args.random_state)

    if args.strategy in {"search", "random_search"}:
        detector_names = args.compare or None
        result = run_random_search_workflow(data_dir, detector_names, config=config)
    elif args.strategy == "successive_halving":
        detector_names = args.compare or None
        result = run_successive_halving_workflow(data_dir, detector_names, config=config)
    elif args.strategy == "hyperband":
        detector_names = args.compare or None
        result = run_hyperband_workflow(data_dir, detector_names, config=config)
    elif args.compare is not None or args.strategy == "compare":
        detector_names = args.compare or None
        result = run_comparison_workflow(data_dir, detector_names, config=config)
    else:
        result = run_minimal_workflow(data_dir, config=config, detector_name=args.detector)

    print(format_run_result(result))

    detector_names = args.compare if args.compare else [args.detector]
    benchmark_metadata = build_benchmark_metadata(
        data_dir=data_dir,
        strategy_name=result.strategy_name or args.strategy,
        random_state=args.random_state,
        split_plan={
            "training": "train_fault_free",
            "validation": "subset(train_fault_free)",
            "test": ["test_fault_free", "test_faulty"],
            "faulty_training": "train_faulty (explorative only)",
        },
        detector_names=detector_names,
        extra={
            "selected_detector": args.detector,
            "requested_strategy": args.strategy,
        },
    )

    if args.save_result:
        save_result_json(result, args.save_result, metadata=benchmark_metadata)

    if args.append_history:
        append_result_jsonl(result, args.append_history, metadata=benchmark_metadata)


if __name__ == "__main__":
    main()