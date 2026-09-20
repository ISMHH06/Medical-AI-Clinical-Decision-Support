"""FusionDataset combining precomputed vision embeddings with clinical features.

This dataset class merges 1024-dim vision embeddings (frozen DenseNet-121)
with 29-dim clinical features (age, BMI, one-hot categoricals, missingness
flags) and the 5 target labels, all aligned by the manifest split.

The clinical feature pipeline exactly mirrors ClinicalDataset from Phase 6:
- Train split: build_clinical_features with train_frame=subset (computes stats internally)
- Val/test split: get_train_feature_columns + get_train_standardization_stats from train,
  then pass to build_clinical_features to avoid data leakage.
"""

from __future__ import annotations

import logging
from pathlib import Path

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

LOGGER = logging.getLogger(__name__)


class FusionDataset(Dataset):
    """Fusion dataset: vision embedding + clinical features + targets.

    The vision embeddings are loaded from precomputed .npy files generated
    by extract_vision_embeddings.py. The clinical features are computed on-
    the-fly using build_clinical_features with train-derived statistics to
    avoid data leakage, exactly mirroring the Phase 6 ClinicalDataset pattern.

    Rows are aligned by manifest index: the embedding array's row count
    must match the clinical dataset's row count for the same split.
    """

    def __init__(
        self,
        manifest_path: str,
        split: str,
        embedding_path: str,
    ) -> None:
        if split not in ("train", "val", "test"):
            raise ValueError(f"split must be 'train', 'val', or 'test', got {split!r}.")

        self.manifest_path = manifest_path
        self.split = split

        # Load precomputed vision embeddings
        embeddings_path = Path(embedding_path)
        if not embeddings_path.is_file():
            raise FileNotFoundError(f"Vision embeddings not found: {embeddings_path}")
        all_embeddings = np.load(embeddings_path)

        # Build clinical features using the same pattern as ClinicalDataset
        frame = pd.read_parquet(manifest_path)
        subset = frame.loc[frame["split"] == split].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"The manifest has no rows in the {split!r} split.")

        if split == "train":
            # Train split: build_clinical_features with train_frame=subset
            # (computes medians, one-hot, standardization internally)
            features_df, self._feature_columns = build_clinical_features(
                subset, train_frame=subset
            )
        else:
            # Val/test split: mirror ClinicalDataset pattern
            # Get train-derived feature columns and standardization stats
            train_frame = frame.loc[frame["split"] == "train"]
            train_feature_columns = get_train_feature_columns(train_frame)
            age_mean, age_std, bmi_mean, bmi_std = get_train_standardization_stats(train_frame)

            # Build clinical features with leakage-prevention stats
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
        )  # shape: (N, 29)
        self.targets = torch.tensor(
            subset[LABEL_COLUMNS].to_numpy(dtype=np.float32), dtype=torch.float32
        )  # shape: (N, 5)

        # Load embeddings aligned to this split
        self.embeddings = all_embeddings

        # Sanity check: embedding row count must match clinical row count
        if self.embeddings.shape[0] != self.features.shape[0]:
            raise RuntimeError(
                f"Row count mismatch for {split} split: "
                f"vision embeddings={self.embeddings.shape[0]}, "
                f"clinical features={self.features.shape[0]}"
            )
        self.num_samples = self.features.shape[0]

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        vision_emb = torch.tensor(
            self.embeddings[idx], dtype=torch.float32
        )  # shape: (1024,)
        clinical_feat = self.features[idx]  # shape: (29,)
        target = self.targets[idx]  # shape: (5,)
        return vision_emb, clinical_feat, target