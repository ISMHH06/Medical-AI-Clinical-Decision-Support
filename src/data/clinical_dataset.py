"""PyTorch Dataset for Phase 6 clinical-only tabular modeling.

Loads clinical features from the reconciled subset manifest, filtered by
split ("train", "val", or "test"), using train-derived statistics for
imputation and one-hot alignment to avoid leakage.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

import pandas as pd

from src.data.clinical_features import (
    build_clinical_features,
    get_train_feature_columns,
    get_train_standardization_stats,
)
from src.data.image_dataset import LABEL_COLUMNS


class ClinicalDataset(Dataset):
    """Clinical tabular dataset backed by the subset manifest parquet."""

    def __init__(
        self,
        manifest_path: str,
        split: str = "train",
        train_feature_columns: list[str] | None = None,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}.")

        frame = pd.read_parquet(manifest_path)
        required = {"split", *LABEL_COLUMNS}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        subset = frame.loc[frame["split"] == split].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"The manifest has no rows in the {split!r} split.")

        if split == "train":
            features_df, self._feature_columns = build_clinical_features(
                subset, train_frame=subset
            )
        else:
            if train_feature_columns is None:
                train_frame = frame.loc[frame["split"] == "train"]
                train_feature_columns = get_train_feature_columns(train_frame)
                age_mean, age_std, bmi_mean, bmi_std = get_train_standardization_stats(train_frame)
            features_df, _ = build_clinical_features(
                subset,
                train_frame=frame.loc[frame["split"] == "train"],
                feature_columns=train_feature_columns,
                age_mean=age_mean,
                age_std=age_std,
                bmi_mean=bmi_mean,
                bmi_std=bmi_std,
            )
            self._feature_columns = train_feature_columns

        self.features = torch.tensor(
            features_df.to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.targets = torch.tensor(
            subset[LABEL_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
        )
        self.split = split

    @property
    def feature_dim(self) -> int:
        return len(self._feature_columns)

    @property
    def feature_columns(self) -> list[str]:
        return self._feature_columns

    def __len__(self) -> int:
        return len(self.features)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.features[idx], self.targets[idx]