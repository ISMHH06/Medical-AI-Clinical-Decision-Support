"""Generate dense 2x5 explanation figures for selected E08 patients."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.clinical_features import build_clinical_features, get_train_feature_columns
from src.data.fusion_dataset import FusionDataset
from src.data.image_dataset import LABEL_COLUMNS
from src.explainability.combined_view import build_patient_figure
from src.explainability.gradcam import E08Explainer
from src.models.fusion_model import FusionModel
from scripts.sanity_check_shap import build_background_features


MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
VISION_EMBEDDINGS_PATH = REPO_ROOT / "models" / "vision_embeddings_val.npy"
OUTPUT_DIR = REPO_ROOT / "outputs" / "combined_explanations"
CASE_INDICES = [136, 1035, 403, 451]


def main() -> None:
    """Create one dense explanation figure per selected patient."""
    frame = pd.read_parquet(MANIFEST_PATH)
    val_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    feature_names = get_train_feature_columns(train_frame)
    label_names = LABEL_COLUMNS
    background_feats = build_background_features()

    dataset = FusionDataset(str(MANIFEST_PATH), "val", str(VISION_EMBEDDINGS_PATH))
    explainer = E08Explainer()
    fusion_model = FusionModel()
    state = torch.load(REPO_ROOT / "models" / "fusion_e08_best.pt", map_location="cpu", weights_only=True)
    fusion_model.load_state_dict(state["state_dict"])
    fusion_model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explainer = explainer.to(device)
    fusion_model = fusion_model.to(device)
    explainer.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for idx in CASE_INDICES:
        row = val_frame.iloc[idx]
        vision_emb, clinical_feat, target = dataset[idx]
        ground_truth_labels = row[LABEL_COLUMNS].to_numpy(dtype=np.int32)
        fig = build_patient_figure(
            row_index=idx,
            image_path=row["local_image_path"],
            explainer=explainer,
            fusion_model=fusion_model,
            vision_emb=vision_emb,
            clinical_feat=clinical_feat,
            background_feats=background_feats,
            feature_names=feature_names,
            label_names=label_names,
            ground_truth_labels=ground_truth_labels,
            device=device,
        )
        output_path = OUTPUT_DIR / f"patient_{idx}.png"
        fig.savefig(output_path, dpi=150)
        print(f"Saved combined explanation: {output_path}")

    print("Combined explanation generation completed successfully.")


if __name__ == "__main__":
    main()
