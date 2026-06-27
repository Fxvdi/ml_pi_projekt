"""Create a balanced TEP test split from the existing FaultFree and Faulty files."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def build_balanced_test_split(
    fault_free_path: str | Path,
    faulty_path: str | Path,
    *,
    faulty_to_clean_ratio: float = 1.0,
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
        random_generator = np.random.default_rng(random_state)
        selected_faulty_indices = np.sort(
            random_generator.choice(len(faulty_frame), size=keep_faulty_count, replace=False)
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
        description="Combine the TEP FaultFree and Faulty test splits with a capped faulty sample."
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
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_path = Path(args.output)

    merged_frame = build_balanced_test_split(
        data_dir / "TEP_FaultFree_Testing.parquet",
        data_dir / "TEP_Faulty_Testing.parquet",
        faulty_to_clean_ratio=args.faulty_to_clean_ratio,
        random_state=args.random_state,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    merged_frame.to_parquet(output_path, index=False)

    print(
        f"Wrote balanced TEP test split to {output_path} "
        f"(ratio={args.faulty_to_clean_ratio:g}, {len(merged_frame)} rows, "
        f"{int((merged_frame['is_fault'] == 1).sum())} faulty, "
        f"{int((merged_frame['is_fault'] == 0).sum())} fault-free)."
    )


if __name__ == "__main__":
    main()