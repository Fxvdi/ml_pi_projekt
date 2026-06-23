"""Simple command-line entry point for the AutoML workflow."""

from __future__ import annotations

import argparse
from pathlib import Path

from automl.pipeline import (
    run_comparison_workflow,
    run_hyperband_workflow,
    run_minimal_workflow,
    run_random_search_workflow,
    run_successive_halving_workflow,
)
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)

    if args.strategy in {"search", "random_search"}:
        detector_names = args.compare or None
        result = run_random_search_workflow(data_dir, detector_names)
    elif args.strategy == "successive_halving":
        detector_names = args.compare or None
        result = run_successive_halving_workflow(data_dir, detector_names)
    elif args.strategy == "hyperband":
        detector_names = args.compare or None
        result = run_hyperband_workflow(data_dir, detector_names)
    elif args.compare is not None or args.strategy == "compare":
        detector_names = args.compare or None
        result = run_comparison_workflow(data_dir, detector_names)
    else:
        result = run_minimal_workflow(data_dir, detector_name=args.detector)

    print(format_run_result(result))


if __name__ == "__main__":
    main()