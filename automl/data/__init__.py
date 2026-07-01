"""Data loading helpers for TEP and other anomaly-detection datasets."""

from .tep import TEPDataset, TEPSplits, load_tep_dataset, load_tep_splits

__all__ = ["TEPDataset", "TEPSplits", "load_tep_dataset", "load_tep_splits"]