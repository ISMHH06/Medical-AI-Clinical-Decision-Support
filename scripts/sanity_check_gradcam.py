"""Sanity-check Grad-CAM overlays for E08 on Cardiomegaly-positive validation rows."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fusion_dataset import FusionDataset
from src.data.image_dataset import build_transform
from src.explainability.gradcam import E08Explainer, generate_gradcam


MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
VISION_EMBEDDINGS_PATH = REPO_ROOT / "models" / "vision_embeddings_val.npy"
OUTPUT_DIR = REPO_ROOT / "outputs" / "gradcam_sanity_check"
LABEL_NAME = "Cardiomegaly_label"
LABEL_IDX = 0


def load_image_tensor(image_path: str) -> torch.Tensor:
    """Load the exact val image tensor used by Grad-CAM."""
    with Image.open(image_path) as image:
        return build_transform("val")(image.convert("RGB"))


def score_candidate_negative(
    explainer: E08Explainer,
    image_tensor: torch.Tensor,
    clinical_feat: torch.Tensor,
    device: torch.device,
) -> tuple[float, float]:
    """Return the cardiomegaly logit and probability for a single sample."""
    with torch.no_grad():
        logits = explainer(image_tensor, clinical_feat.unsqueeze(0).to(device))
    logit_value = float(logits[0, LABEL_IDX].item())
    probability = float(torch.sigmoid(logits[0, LABEL_IDX]).item())
    return logit_value, probability


def main() -> None:
    """Generate and save positive and negative Grad-CAM overlays."""
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not VISION_EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Validation embeddings not found: {VISION_EMBEDDINGS_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    frame = pd.read_parquet(MANIFEST_PATH)
    val_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
    positive_indices = val_frame.index[val_frame[LABEL_NAME] == 1].tolist()
    negative_indices = val_frame.index[val_frame[LABEL_NAME] == 0].tolist()
    if not positive_indices:
        raise ValueError(f"No validation rows found with {LABEL_NAME} == 1.")
    if not negative_indices:
        raise ValueError(f"No validation rows found with {LABEL_NAME} == 0.")

    rng = np.random.default_rng(42)
    positive_count = min(2, len(positive_indices))
    positive_selected = list(rng.choice(positive_indices, size=positive_count, replace=False))

    dataset = FusionDataset(str(MANIFEST_PATH), "val", str(VISION_EMBEDDINGS_PATH))
    explainer = E08Explainer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explainer = explainer.to(device)
    explainer.eval()

    print(f"Loaded {len(val_frame)} validation rows; {len(positive_indices)} are {LABEL_NAME} positive and {len(negative_indices)} are negative.")
    print(f"Positive selected indices: {positive_selected}")
    print(f"Saving overlays to: {OUTPUT_DIR}")

    case_rows: list[tuple[str, int, str, float, float, np.ndarray]] = []

    for index in positive_selected:
        row = val_frame.iloc[index]
        _, clinical_feat, _ = dataset[index]
        image_path = row["local_image_path"]
        image_tensor = load_image_tensor(image_path).unsqueeze(0).to(device)
        logit_value, probability = score_candidate_negative(explainer, image_tensor, clinical_feat, device)
        print(
            f"Row {index} | {Path(image_path).name} | {LABEL_NAME}=1 | "
            f"logit={logit_value:.4f} | probability={probability:.4f}"
        )

        heatmap, overlay_image = generate_gradcam(
            explainer=explainer,
            image_path=image_path,
            clinical_feat=clinical_feat,
            label_idx=LABEL_IDX,
            device=device,
        )
        output_path = OUTPUT_DIR / f"gradcam_{index:04d}_{Path(image_path).stem}.png"
        Image.fromarray(overlay_image).save(output_path)
        print(f"  Saved overlay: {output_path}")
        print(f"  Heatmap stats: mean={float(heatmap.mean()):.4f}, max={float(heatmap.max()):.4f}")
        case_rows.append(("positive", index, str(Path(image_path).name), logit_value, probability, heatmap))

    negative_pool = []
    for index in negative_indices:
        row = val_frame.iloc[index]
        _, clinical_feat, _ = dataset[index]
        image_path = row["local_image_path"]
        image_tensor = load_image_tensor(image_path).unsqueeze(0).to(device)
        logit_value, probability = score_candidate_negative(explainer, image_tensor, clinical_feat, device)
        if probability < 0.3:
            negative_pool.append((index, image_path, clinical_feat, logit_value, probability))

    if not negative_pool:
        candidate_order = list(rng.choice(negative_indices, size=min(20, len(negative_indices)), replace=False))
        for index in candidate_order:
            row = val_frame.iloc[index]
            _, clinical_feat, _ = dataset[index]
            image_path = row["local_image_path"]
            image_tensor = load_image_tensor(image_path).unsqueeze(0).to(device)
            logit_value, probability = score_candidate_negative(explainer, image_tensor, clinical_feat, device)
            if probability < 0.3:
                negative_pool.append((index, image_path, clinical_feat, logit_value, probability))

    if not negative_pool:
        raise RuntimeError("Could not find any sufficiently confident negative validation examples for the contrast check.")

    negative_selected = sorted(negative_pool, key=lambda item: item[3])[:2]
    print(f"Selected confident negative indices: {[idx for idx, _, _, _, _ in negative_selected]}")

    for index, image_path, clinical_feat, logit_value, probability in negative_selected:
        row = val_frame.iloc[index]
        print(
            f"Row {index} | {Path(image_path).name} | {LABEL_NAME}=0 | "
            f"logit={logit_value:.4f} | probability={probability:.4f}"
        )

        heatmap, overlay_image = generate_gradcam(
            explainer=explainer,
            image_path=image_path,
            clinical_feat=clinical_feat,
            label_idx=LABEL_IDX,
            device=device,
        )
        output_path = OUTPUT_DIR / f"gradcam_negative_{index:04d}_{Path(image_path).stem}.png"
        Image.fromarray(overlay_image).save(output_path)
        print(f"  Saved overlay: {output_path}")
        print(f"  Heatmap stats: mean={float(heatmap.mean()):.4f}, max={float(heatmap.max()):.4f}")
        case_rows.append(("negative", index, str(Path(image_path).name), logit_value, probability, heatmap))

    print("\nHeatmap intensity comparison:")
    for label_type, index, filename, logit_value, probability, heatmap in case_rows:
        print(
            f"  {label_type.upper()} row {index} | file={filename} | "
            f"logit={logit_value:.4f} | probability={probability:.4f} | "
            f"heatmap.mean={float(heatmap.mean()):.4f} | heatmap.max={float(heatmap.max()):.4f}"
        )

    print("\nGrad-CAM negative differential sanity check completed successfully.")


if __name__ == "__main__":
    main()