"""Run E04 (vision-only) and E08 (fusion) inference over a requested split.

Supports --split {val,test} and saves per-label sigmoid probabilities
alongside ground truth for that split.

This is a diagnostic artifact for Phase 9 (calibration/ECE, temperature
scaling, abstention) -- no model weights are changed here, no training
happens here. It reuses the same E04Explainer / E08Explainer wrappers
built in Phase 8, just looped over every row in the chosen split.

Output:
    --split val  -> outputs/val_predictions.parquet
    --split test -> outputs/test_predictions.parquet

Columns:
    row_index, patient_id (if available)
    <label>_true   for each of the 5 labels  (ground truth, 0/1)
    <label>_e04    for each of the 5 labels  (E04 vision-only probability)
    <label>_e08    for each of the 5 labels  (E08 fusion probability)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fusion_dataset import FusionDataset
from src.data.image_dataset import LABEL_COLUMNS, build_transform
from src.explainability.gradcam import E04Explainer, E08Explainer

MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
VISION_EMBEDDINGS_BY_SPLIT = {
    "val": REPO_ROOT / "models" / "vision_embeddings_val.npy",
    "test": REPO_ROOT / "models" / "vision_embeddings_test.npy",
}
OUTPUT_PATH_BY_SPLIT = {
    "val": REPO_ROOT / "outputs" / "val_predictions.parquet",
    "test": REPO_ROOT / "outputs" / "test_predictions.parquet",
}

# Conservative default for a 6GB GPU doing forward-only inference at
# 224x224 through a DenseNet-121; raise if you have headroom, lower if
# you hit CUDA OOM.
BATCH_SIZE = 16
NUM_WORKERS = 2


class _SplitImageDataset(torch.utils.data.Dataset):
    """Loads raw split images (for E04/E08 forward passes) alongside their index."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.transform = build_transform("val")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int):
        from PIL import Image

        row = self.frame.iloc[idx]
        with Image.open(row["local_image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return idx, tensor


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate E04/E08 prediction parquet for val or test split.")
    parser.add_argument("--split", choices=["val", "test"], default="val", help="Dataset split to score.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = args.split
    output_path = OUTPUT_PATH_BY_SPLIT[split]
    embeddings_path = VISION_EMBEDDINGS_BY_SPLIT[split]

    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not embeddings_path.is_file():
        raise FileNotFoundError(f"Embeddings not found for split {split!r}: {embeddings_path}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    frame = pd.read_parquet(MANIFEST_PATH)
    split_frame = frame.loc[frame["split"] == split].reset_index(drop=True)
    n_rows = len(split_frame)
    print(f"{split.capitalize()} split size: {n_rows}")

    fusion_dataset = FusionDataset(str(MANIFEST_PATH), split, str(embeddings_path))
    image_dataset = _SplitImageDataset(split_frame)
    image_loader = DataLoader(
        image_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )

    e04 = E04Explainer().to(device)
    e04.eval()
    e08 = E08Explainer().to(device)
    e08.eval()

    n_labels = len(LABEL_COLUMNS)
    e04_probs = np.zeros((n_rows, n_labels), dtype=np.float32)
    e08_probs = np.zeros((n_rows, n_labels), dtype=np.float32)

    # Pre-load clinical feature tensors for every row in original split order
    # (cheap -- these are small vectors already produced by FusionDataset).
    clinical_feats = torch.stack([fusion_dataset[i][1] for i in range(n_rows)])

    start = time.time()
    seen = 0
    with torch.no_grad():
        for indices, images in image_loader:
            images = images.to(device, non_blocking=True)
            batch_clinical = clinical_feats[indices].to(device, non_blocking=True)

            e04_logits = e04(images)
            e04_batch_probs = torch.sigmoid(e04_logits).cpu().numpy()

            e08_logits = e08(images, batch_clinical)
            e08_batch_probs = torch.sigmoid(e08_logits).cpu().numpy()

            idx_np = indices.numpy()
            e04_probs[idx_np] = e04_batch_probs
            e08_probs[idx_np] = e08_batch_probs

            seen += len(idx_np)
            if seen % (BATCH_SIZE * 10) == 0 or seen == n_rows:
                elapsed = time.time() - start
                print(f"  {seen}/{n_rows} {split} rows scored ({elapsed:.1f}s elapsed)")

    ground_truth = split_frame[LABEL_COLUMNS].to_numpy(dtype=np.float32)

    out = pd.DataFrame({"row_index": np.arange(n_rows)})
    if "patient_id" in split_frame.columns:
        out["patient_id"] = split_frame["patient_id"].to_numpy()
    elif "deid_patient_id" in split_frame.columns:
        out["patient_id"] = split_frame["deid_patient_id"].to_numpy()

    for i, label in enumerate(LABEL_COLUMNS):
        out[f"{label}_true"] = ground_truth[:, i]
        out[f"{label}_e04"] = e04_probs[:, i]
        out[f"{label}_e08"] = e08_probs[:, i]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(output_path, index=False)

    elapsed_total = time.time() - start
    print(f"\nSaved {n_rows} rows x {n_labels} labels x 2 models to: {output_path}")
    print(f"Total inference time: {elapsed_total:.1f}s ({elapsed_total / max(n_rows, 1):.3f}s/row)")


if __name__ == "__main__":
    main()