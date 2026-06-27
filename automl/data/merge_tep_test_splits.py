"""Create a balanced TEP test split from the existing FaultFree and Faulty files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _sample_indices_by_label(
    frame: pd.DataFrame,
    *,
    target_count: int,
    label_column: str,
    random_state: int | None = None,
) -> np.ndarray:
    """Sample row indices while keeping at least one row per label when possible."""

    if label_column not in frame.columns:
        random_generator = np.random.default_rng(random_state)
        return np.sort(random_generator.choice(len(frame), size=target_count, replace=False))

    grouped = list(frame.groupby(label_column, sort=True))
    label_count = len(grouped)
    if target_count < label_count:
        raise ValueError(
            f"target_count={target_count} is too small to keep every label in {label_column!r} "
            f"(need at least {label_count})"
        )

    counts = np.asarray([len(group_frame) for _, group_frame in grouped], dtype=int)
    proportions = counts / counts.sum()
    ideal_allocations = target_count * proportions
    allocations = np.floor(ideal_allocations).astype(int)
    allocations = np.maximum(allocations, 1)

    remainder = target_count - int(allocations.sum())
    fractional_parts = ideal_allocations - np.floor(ideal_allocations)

    if remainder > 0:
        order = np.argsort(-fractional_parts, kind="mergesort")
        for index in order:
            if remainder == 0:
                break
            if allocations[index] < counts[index]:
                allocations[index] += 1
                remainder -= 1
    elif remainder < 0:
        order = np.argsort(fractional_parts, kind="mergesort")
        for index in order:
            if remainder == 0:
                break
            if allocations[index] > 1:
                allocations[index] -= 1
                remainder += 1

    if int(allocations.sum()) != target_count:
        raise ValueError("Could not distribute the requested sample size across labels")

    random_generator = np.random.default_rng(random_state)
    selected_indices: list[int] = []
    for allocation, (_, group_frame) in zip(allocations, grouped, strict=True):
        group_indices = group_frame.index.to_numpy()
        chosen_indices = random_generator.choice(group_indices, size=allocation, replace=False)
        selected_indices.extend(chosen_indices.tolist())

    return np.sort(np.asarray(selected_indices, dtype=int))


def _prepare_run(frame: pd.DataFrame) -> pd.DataFrame:
    """Return a copy of one run sorted by sample and reindexed from zero."""

    return frame.sort_values("sample").reset_index(drop=True)


def build_fault_onset_dataset(
    fault_free_path: str | Path,
    faulty_path: str | Path,
    *,
    fault_onset_fraction: float = 0.5,
    random_state: int | None = None,
) -> pd.DataFrame:
    """Build synthetic sequences where a clean run turns faulty mid-stream.

    The output keeps the temporal order within each synthetic run. Each generated
    run uses the same ``sample`` length as the source data, starts with clean rows,
    and then switches to a faulty segment at the requested onset fraction.
    """

    if not 0 < fault_onset_fraction < 1:
        raise ValueError("fault_onset_fraction must be in the interval (0, 1)")

    fault_free_frame = pd.read_parquet(Path(fault_free_path))
    faulty_frame = pd.read_parquet(Path(faulty_path))

    if len(fault_free_frame) == 0:
        raise ValueError("FaultFree test split is empty")
    if len(faulty_frame) == 0:
        raise ValueError("Faulty test split is empty")

    clean_runs = {
        simulation_run: _prepare_run(run_frame)
        for simulation_run, run_frame in fault_free_frame.groupby("simulationRun", sort=True)
    }
    faulty_segments = {
        (simulation_run, fault_number): _prepare_run(run_frame)
        for (simulation_run, fault_number), run_frame in faulty_frame.groupby(
            ["simulationRun", "faultNumber"], sort=True
        )
    }

    if not clean_runs:
        raise ValueError("No clean runs found in the FaultFree split")
    if not faulty_segments:
        raise ValueError("No faulty segments found in the Faulty split")

    available_fault_numbers = sorted({int(fault_number) for _, fault_number in faulty_segments})
    ordered_clean_runs = list(clean_runs.items())
    synthetic_runs: list[pd.DataFrame] = []
    synthetic_run_id = 1

    for fault_number in available_fault_numbers:
        for simulation_run, clean_run in ordered_clean_runs:
            faulty_segment = faulty_segments.get((simulation_run, fault_number))
            if faulty_segment is None:
                continue

            if len(clean_run) != len(faulty_segment):
                raise ValueError(
                    "Clean and faulty segments must have the same length for splicing: "
                    f"run={simulation_run!r}, faultNumber={fault_number!r}, "
                    f"clean={len(clean_run)}, faulty={len(faulty_segment)}"
                )

            onset_index = int(round(len(clean_run) * fault_onset_fraction))
            onset_index = max(1, min(onset_index, len(clean_run) - 1))

            clean_prefix = clean_run.iloc[:onset_index].copy()
            faulty_suffix = faulty_segment.iloc[onset_index:].copy()

            clean_prefix["faultNumber"] = 0
            clean_prefix["is_fault"] = 0
            faulty_suffix["is_fault"] = 1

            spliced_run = pd.concat([clean_prefix, faulty_suffix], ignore_index=True)
            spliced_run["sample"] = np.arange(1, len(spliced_run) + 1)
            spliced_run["simulationRun"] = synthetic_run_id
            synthetic_run_id += 1

            synthetic_runs.append(spliced_run)

    if not synthetic_runs:
        raise ValueError("Could not build any synthetic onset runs")

    combined_frame = pd.concat(synthetic_runs, ignore_index=True)
    return combined_frame.reset_index(drop=True)


def build_balanced_test_split(
    fault_free_path: str | Path,
    faulty_path: str | Path,
    *,
    faulty_to_clean_ratio: float = 1.0,
    fault_label_column: str = "faultNumber",
    random_state: int | None = None,
) -> pd.DataFrame:
    """Combine the two test files and cap faulty rows according to the requested ratio."""

    if faulty_to_clean_ratio <= 0:
        raise ValueError("faulty_to_clean_ratio must be greater than 0")

    fault_free_frame = pd.read_parquet(Path(fault_free_path))
    faulty_frame = pd.read_parquet(Path(faulty_path))

    if len(fault_free_frame) == 0:
        raise ValueError("FaultFree test split is empty")
    if len(faulty_frame) == 0:
        raise ValueError("Faulty test split is empty")

    target_faulty_count = max(1, int(round(len(fault_free_frame) * faulty_to_clean_ratio)))
    keep_faulty_count = min(target_faulty_count, len(faulty_frame))
    if keep_faulty_count < len(faulty_frame):
        selected_faulty_indices = _sample_indices_by_label(
            faulty_frame,
            target_count=keep_faulty_count,
            label_column=fault_label_column,
            random_state=random_state,
        )
        faulty_frame = faulty_frame.iloc[selected_faulty_indices].reset_index(drop=True)
    else:
        faulty_frame = faulty_frame.reset_index(drop=True)

    fault_free_frame = fault_free_frame.reset_index(drop=True)

    merged_frame = pd.concat([fault_free_frame, faulty_frame], ignore_index=True)
    merged_frame["is_fault"] = [0] * len(fault_free_frame) + [1] * len(faulty_frame)

    return merged_frame


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build either a capped benchmark test split or synthetic fault-onset sequences from the TEP data."
    )
    parser.add_argument(
        "--mode",
        choices=["benchmark", "onset"],
        default="benchmark",
        help="Choose between a merged benchmark split and a synthetic fault-onset dataset.",
    )
    parser.add_argument(
        "--data-dir",
        default="automl/data",
        help="Directory containing the TEP parquet files.",
    )
    parser.add_argument(
        "--output",
        default="automl/data/TEP_Testing_Balanced.parquet",
        help="Path of the merged parquet file.",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=0,
        help="Seed used when Faulty has to be downsampled.",
    )
    parser.add_argument(
        "--faulty-to-clean-ratio",
        type=float,
        default=1.0,
        help="Target number of faulty rows relative to clean rows. Example: 0.5 keeps half as many faulty rows as clean rows.",
    )
    parser.add_argument(
        "--fault-label-column",
        default="faultNumber",
        help="Column used to keep all fault classes in the faulty sample when available.",
    )
    parser.add_argument(
        "--fault-onset-fraction",
        type=float,
        default=0.5,
        help="Fraction of each synthetic run that stays clean before the fault starts. Used in onset mode.",
    )
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    if args.mode == "benchmark":
        merged_frame = build_balanced_test_split(
            data_dir / "TEP_FaultFree_Testing.parquet",
            data_dir / "TEP_Faulty_Testing.parquet",
            faulty_to_clean_ratio=args.faulty_to_clean_ratio,
            fault_label_column=args.fault_label_column,
            random_state=args.random_state,
        )
    else:
        merged_frame = build_fault_onset_dataset(
            data_dir / "TEP_FaultFree_Testing.parquet",
            data_dir / "TEP_Faulty_Testing.parquet",
            fault_onset_fraction=args.fault_onset_fraction,
            random_state=args.random_state,
        )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_frame.to_parquet(output_path, index=False)

    if args.mode == "benchmark":
        print(
            f"Wrote balanced TEP test split to {output_path} "
            f"(ratio={args.faulty_to_clean_ratio:g}, {len(merged_frame)} rows, "
            f"{int((merged_frame['is_fault'] == 1).sum())} faulty, "
            f"{int((merged_frame['is_fault'] == 0).sum())} fault-free)."
        )
    else:
        print(
            f"Wrote fault-onset TEP dataset to {output_path} "
            f"(onset={args.fault_onset_fraction:g}, {len(merged_frame)} rows, "
            f"{int((merged_frame['is_fault'] == 1).sum())} faulty, "
            f"{int((merged_frame['is_fault'] == 0).sum())} fault-free)."
        )


if __name__ == "__main__":
    main()