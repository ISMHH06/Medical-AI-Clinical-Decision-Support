"""Phase 7 E08 Step 1: Extract vision embeddings from frozen DenseNet-121.

Loads the E04 champion checkpoint, replaces the classifier with Identity,
and runs a forward pass on train/val/test splits to produce 1024-dim
embeddings. Saves as .npy files for use in the fusion model.
"""

from __future__ import annotations

import logging
import numpy as np
import sys
from pathlib import Path

# Ensure repo root is on sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.image_dataset import ChestXrayDataset
from src.models.vision_baseline import build_vision_baseline_frozen

LOGGER = logging.getLogger(__name__)

VISION_CHECKPOINT = "models/vision_baseline_champion.pt"
EMBEDDING_DIR = REPO_ROOT / "models"
INPUT_SIZE = (224, 224)


def extract_embeddings(split: str, manifest_path: Path, embedding_dir: Path) -> np.ndarray:
    """Extract 1024-dim vision embeddings for a given split.

    Replaces model.classifier with nn.Identity() so forward pass returns
    the 1024-dim pooled feature vector directly.
    """
    model = build_vision_baseline_frozen(unfreeze_from_block="denseblock3")
    ckpt_path = REPO_ROOT / VISION_CHECKPOINT
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"Champion checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(torch.device("cuda"))

    # Replace classifier with Identity so forward returns the pooled vector
    model.classifier = torch.nn.Identity()

    # Build dataset for the split (no shuffling)
    dataset = ChestXrayDataset(str(manifest_path), split=split)
    loader = DataLoader(dataset, batch_size=32, shuffle=False, num_workers=0)

    embeddings = []
    model.eval()
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(torch.device("cuda"))
            with torch.set_grad_enabled(False):
                emb = model(images)
            embeddings.append(emb.cpu().numpy())

    embeddings = np.concatenate(embeddings, axis=0)
    return embeddings


def main() -> None:
    manifest_path = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"

    splits = ["train", "val", "test"]
    embedding_names = {
        "train": ("vision_embeddings_train.npy", 9813),
        "val": ("vision_embeddings_val.npy", 1307),
        "test": ("vision_embeddings_test.npy", None),  # test set size varies
    }

    for split in splits:
        print(f"Extracting {split} embeddings...")
        embeddings = extract_embeddings(split, manifest_path, REPO_ROOT / "models")
        name, expected_count = embedding_names[split]
        actual_count = embeddings.shape[0]
        print(f"  Shape: {embeddings.shape}")
        if expected_count is not None:
            print(f"  Expected count: {expected_count}, Got: {actual_count}")
            assert actual_count == expected_count, (
                f"Row count mismatch for {split}: expected {expected_count}, got {actual_count}"
            )
        # Sanity check: mean and std should not be degenerate
        mean = embeddings.mean()
        std = embeddings.std()
        print(f"  Mean: {mean:.6f}, Std: {std:.6f}")
        if np.isclose(mean, 0.0) and np.isclose(std, 1.0):
            print("  Sanity check: embeddings appear standardized (mean~0, std~1)")
        else:
            print("  Note: embeddings mean/std differ from 0/1 (expected for frozen features)")

        np.save(EMBEDDING_DIR / name, embeddings)
        print(f"  Saved to {name}")


if __name__ == "__main__":
    main()