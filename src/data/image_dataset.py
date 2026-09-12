"""PyTorch Dataset for the Phase 4 image-only vision baseline.

Loads chest X-ray images listed in the reconciled subset manifest,
filtered by split ("train" or "val").
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


LOGGER = logging.getLogger(__name__)

LABEL_COLUMNS = [
    "Cardiomegaly_label",
    "Edema_label",
    "Atelectasis_label",
    "Pleural Effusion_label",
    "Consolidation_label",
]
REQUIRED_COLUMNS = {"local_image_path", "split", *LABEL_COLUMNS}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
IMAGE_SIZE = (224, 224)


def build_transform(split: str) -> transforms.Compose:
    """Build the preprocessing pipeline for a split.

    Train gets RandomHorizontalFlip augmentation before normalization;
    val gets deterministic resize + normalize only.
    """
    normalize = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    if split == "train":
        return transforms.Compose(
            [
                transforms.Resize(IMAGE_SIZE),
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.ToTensor(),
            normalize,
        ]
    )


class ChestXrayDataset(Dataset):
    """Chest X-ray multi-label dataset backed by the subset manifest parquet."""

    def __init__(self, manifest_path: str | Path, split: str = "train") -> None:
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split!r}.")
        manifest_file = Path(manifest_path)
        if not manifest_file.is_file():
            raise FileNotFoundError(f"Manifest was not found: {manifest_file}")

        frame = pd.read_parquet(manifest_file)
        missing = sorted(REQUIRED_COLUMNS.difference(frame.columns))
        if missing:
            raise ValueError(f"Manifest is missing required columns: {missing}")

        subset = frame.loc[frame["split"] == split].reset_index(drop=True)
        if subset.empty:
            raise ValueError(f"The manifest has no rows in the {split!r} split.")
        if subset["local_image_path"].isna().any():
            raise ValueError("local_image_path contains missing values.")

        self.records = subset
        self.split = split
        self.transform = build_transform(split)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Load one sample, falling back to later rows on read failure.

        Image loading failures are logged and skipped by trying subsequent
        rows (wrapping around). Phase 3 found 0 corrupt files, so this path
        should be rare, but it keeps training from crashing on a bad file.
        """
        for attempt in range(len(self.records)):
            row_idx = (idx + attempt) % len(self.records)
            row = self.records.iloc[row_idx]
            try:
                with Image.open(row["local_image_path"]) as image:
                    image = image.convert("RGB")
                if self.transform is not None:
                    image = self.transform(image)
                target = torch.tensor(
                    row[LABEL_COLUMNS].to_numpy(dtype="float32"),
                    dtype=torch.float32,
                )
                return image, target
            except Exception as error:
                LOGGER.warning(
                    "Skipping unreadable image %s: %s",
                    row.get("local_image_path"),
                    error,
                )
        raise RuntimeError("No readable images found in the dataset.")
