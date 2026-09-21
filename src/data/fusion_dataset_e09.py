"""FusionDatasetE09: raw images + clinical features for partial fine-tuning.

Unlike FusionDataset (which used precomputed .npy embeddings), this class
loads raw images from disk using the existing ChestXrayDataset transform
logic, and also computes clinical features on-the-fly with leakage-
prevention (train-derived medians/one-hot columns/standardization stats).

__getitem__ returns (image_tensor (3, 224, 224), clinical_features (29,),
target (5,)).
"""

from __future__ import annotations

import logging
import pandas as pd
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from src.data.clinical_features import build_clinical_features, get_train_feature_columns, get_train_standardization_stats
from src.data.image_dataset import ChestXrayDataset, LABEL_COLUMNS

LOGGER = logging.getLogger(__name__)


class FusionDatasetE09(Dataset):
    """Fusion dataset for E09: raw images + clinical features.

    Loads raw images via an internal ChestXrayDataset instance (same
    transform logic as Phase 4-7 vision baselines) and computes clinical
    features using train-derived statistics to avoid data leakage,
    mirroring the pattern from src/data/fusion_dataset.py and
    src/data/clinical_dataset.py.

    Unlike FusionDataset which used precomputed embeddings, this class
    performs a real image forward pass each training step, enabling
    partial vision fine-tuning.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
    ) -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}.")

        self.manifest_path = manifest_path
        self.split = split

        # Internal ChestXrayDataset handles raw image loading/transforms
        self.image_dataset = ChestXrayDataset(manifest_path, split=split)

        # Build clinical features using leakage-prevention pattern
        frame = pd.read_parquet(manifest_path)
        subset = frame.loc[frame["split"] == split].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"The manifest has no rows in the {split!r} split.")

        if split == "train":
            # Train: build_clinical_features with train_frame=subset
            features_df, self._feature_columns = build_clinical_features(
                subset, train_frame=subset
            )
        else:
            # Val: mirror ClinicalDataset pattern
            train_frame = frame.loc[frame["split"] == "train"]
            train_feature_columns = get_train_feature_columns(train_frame)
            age_mean, age_std, bmi_mean, bmi_std = get_train_standardization_stats(train_frame)

            features_df, _ = build_clinical_features(
                subset,
                train_frame=train_frame,
                feature_columns=train_feature_columns,
                age_mean=age_mean,
                age_std=age_std,
                bmi_mean=bmi_mean,
                bmi_std=bmi_std,
            )
            self._feature_columns = train_feature_columns

        self.features = torch.tensor(
            features_df.to_numpy(dtype=np.float32), dtype=torch.float32
        )  # shape: (N, 29)
        self.targets = torch.tensor(
            subset[LABEL_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
        )  # shape: (N, 5)

        # Sanity check: feature row count must match clinical row count
        if self.features.shape[0] != self.targets.shape[0]:
            raise RuntimeError(
                f"Row count mismatch for {split} split: "
                f"features={self.features.shape[0]}, targets={self.targets.shape[0]}"
            )
        self.num_samples = self.features.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Raw image from ChestXrayDataset (already transformed to 3, 224, 224)
        image, _ = self.image_dataset[idx]  # type: ignore[return-value]
        clinical_feat = self.features[idx]  # shape: (29,)
        target = self.targets[idx]  # shape: (5,)
        return image, clinical_feat, target