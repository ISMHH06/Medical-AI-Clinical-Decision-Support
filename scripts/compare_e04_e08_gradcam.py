"""Compare Grad-CAM attention for E04 vs E08 on the same Cardiomegaly case."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.image_dataset import build_transform
from src.explainability.gradcam import E04Explainer, E08Explainer, generate_gradcam, generate_gradcam_e04
from src.data.fusion_dataset import FusionDataset

MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
VISION_EMBEDDINGS_PATH = REPO_ROOT / "models" / "vision_embeddings_val.npy"
OUTPUT_PATH = REPO_ROOT / "outputs" / "e04_vs_e08_comparison" / "patient_1035_cardiomegaly.png"


def main() -> None:
    """Compare the E04 and E08 Grad-CAM overlays for patient 1035 (Cardiomegaly)."""
    frame = pd.read_parquet(MANIFEST_PATH)
    val_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
    dataset = FusionDataset(str(MANIFEST_PATH), "val", str(VISION_EMBEDDINGS_PATH))

    row = val_frame.iloc[1035]
    image_path = row["local_image_path"]
    vision_emb, clinical_feat, _ = dataset[1035]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    e04_explainer = E04Explainer().to(device)
    e04_explainer.eval()
    e04_heatmap, e04_overlay = generate_gradcam_e04(e04_explainer, image_path, label_idx=0, device=device)

    e08_explainer = E08Explainer().to(device)
    e08_explainer.eval()
    e08_heatmap, e08_overlay = generate_gradcam(e08_explainer, image_path, clinical_feat, label_idx=0, device=device)

    with Image.open(image_path) as image:
        real_image = build_transform("val")(image.convert("RGB"))
    real_image = real_image.unsqueeze(0).to(device)

    with torch.no_grad():
        e04_logits = e04_explainer(real_image)
    e04_prob = float(torch.sigmoid(e04_logits[0, 0]).detach().cpu().item())

    with torch.no_grad():
        e08_logits = e08_explainer(real_image, clinical_feat.unsqueeze(0).to(device))
    e08_prob = float(torch.sigmoid(e08_logits[0, 0]).detach().cpu().item())

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    axes[0].imshow(e04_overlay)
    axes[0].set_axis_off()
    axes[0].set_title(f"E04 (vision-only)\np={e04_prob:.2f}")

    axes[1].imshow(e08_overlay)
    axes[1].set_axis_off()
    axes[1].set_title(f"E08 (fusion)\np={e08_prob:.2f}")

    fig.suptitle("Patient (val row 1035) — Cardiomegaly")
    fig.tight_layout()
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=200)
    plt.close(fig)

    print(f"E04 Cardiomegaly probability: {e04_prob:.6f}")
    print(f"E08 Cardiomegaly probability: {e08_prob:.6f}")
    print(f"Saved comparison figure: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
