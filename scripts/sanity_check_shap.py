"""Sanity-check SHAP attribution for the E08 clinical branch."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.clinical_features import build_clinical_features, get_train_feature_columns
from src.data.fusion_dataset import FusionDataset
from src.explainability.shap_explainer import generate_shap_values
from src.models.fusion_model import FusionModel


MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
VISION_EMBEDDINGS_PATH = REPO_ROOT / "models" / "vision_embeddings_val.npy"
TRAIN_VISION_EMBEDDINGS_PATH = REPO_ROOT / "models" / "vision_embeddings_train.npy"
OUTPUT_DIR = REPO_ROOT / "outputs" / "shap_sanity_check"
LABEL_NAME = "Cardiomegaly_label"
LABEL_IDX = 0

CASE_INDICES = [136, 1035, 403, 451]


def load_row_case(dataset: FusionDataset, val_frame: pd.DataFrame, idx: int) -> tuple[str, str, torch.Tensor, torch.Tensor]:
    """Return the row image path plus its clinical feature vector for a val-indexed case."""
    row = val_frame.iloc[idx]
    vision_emb, clinical_feat, _ = dataset[idx]
    return row["local_image_path"], row[LABEL_NAME], vision_emb, clinical_feat


def build_background_features() -> np.ndarray:
    """Build a deterministic 50-row train-split clinical feature background."""
    frame = pd.read_parquet(MANIFEST_PATH)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    rng = np.random.default_rng(42)
    candidate_indices = rng.choice(len(train_frame), size=min(50, len(train_frame)), replace=False)
    sample = train_frame.iloc[candidate_indices].copy()
    feature_frame, _ = build_clinical_features(sample, train_frame=train_frame)
    return feature_frame.to_numpy(dtype=np.float32)


def save_bar_chart(
    values: np.ndarray,
    feature_names: list[str],
    output_path: Path | None = None,
    title: str = "",
    ax: plt.Axes | None = None,
) -> plt.Axes | None:
    """Render a signed SHAP bar chart for one explanation."""
    top_count = min(8, len(values))
    order = np.argsort(np.abs(values))[-top_count:][::-1]
    top_values = values[order]
    top_names = [feature_names[i] for i in order]

    colors = ["tab:red" if value > 0 else "tab:blue" for value in top_values]
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))
        should_save = True
    else:
        should_save = False

    ax.barh(top_names, top_values, color=colors)
    ax.axvline(0, color="black", linewidth=1)
    if title:
        ax.set_title(title)
    ax.set_xlabel("SHAP value")
    ax.invert_yaxis()

    if should_save and output_path is not None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(output_path, dpi=200)
        plt.close(fig)

    return ax


def print_top_features(label_name: str, values: np.ndarray, feature_names: list[str], limit: int = 5) -> None:
    """Print the highest-absolute SHAP contributions with signed values."""
    order = np.argsort(np.abs(values))[-limit:][::-1]
    print(f"  Top {limit} {label_name} features:")
    for idx in order:
        print(f"    {feature_names[idx]}: value={values[idx]:.6f}, abs={abs(values[idx]):.6f}")


def main() -> None:
    """Compute SHAP feature attributions for the specified val rows."""
    frame = pd.read_parquet(MANIFEST_PATH)
    val_frame = frame.loc[frame["split"] == "val"].reset_index(drop=True)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    feature_columns = get_train_feature_columns(train_frame)

    fusion_model = FusionModel()
    fusion_state = torch.load(REPO_ROOT / "models" / "fusion_e08_best.pt", map_location="cpu", weights_only=True)
    fusion_model.load_state_dict(fusion_state["state_dict"])
    fusion_model.eval()

    background = build_background_features()
    dataset = FusionDataset(str(MANIFEST_PATH), "val", str(VISION_EMBEDDINGS_PATH))

    print(f"Loaded val manifest rows: {len(val_frame)}")
    print(f"Train background rows: {background.shape[0]}")
    print(f"Clinical feature count: {len(feature_columns)}")
    print(f"Case indices: {CASE_INDICES}")

    for idx in CASE_INDICES:
        if idx not in val_frame.index:
            raise IndexError(f"Index {idx} not present in val split.")
        row_path, row_label, vision_emb, clinical_feat = load_row_case(dataset, val_frame, idx)
        shap_values = generate_shap_values(
            fusion_model,
            vision_emb,
            background,
            clinical_feat.numpy(),
            label_idx=0,
        )
        print(f"\nCASE row={idx}, label={row_label}, file={Path(row_path).name}")
        print_top_features("Cardiomegaly", shap_values, feature_columns)
        save_bar_chart(
            shap_values,
            feature_columns,
            OUTPUT_DIR / f"shap_{idx}_cardiomegaly.png",
            f"Row {idx} cardiomegaly SHAP",
        )
        print(f"  Saved chart: {OUTPUT_DIR / f'shap_{idx}_cardiomegaly.png'}")

    idx = 1035
    row_path, row_label, vision_emb, clinical_feat = load_row_case(dataset, val_frame, idx)
    shap_values_consol = generate_shap_values(
        fusion_model,
        vision_emb,
        background,
        clinical_feat.numpy(),
        label_idx=4,
    )
    print(f"\nCASE row={idx}, label={row_label}, file={Path(row_path).name}")
    print_top_features("Consolidation", shap_values_consol, feature_columns)
    save_bar_chart(
        shap_values_consol,
        feature_columns,
        OUTPUT_DIR / "shap_1035_consolidation.png",
        "Row 1035 consolidation SHAP",
    )
    print(f"  Saved chart: {OUTPUT_DIR / 'shap_1035_consolidation.png'}")

    print("\nSHAP sanity check completed successfully.")


if __name__ == "__main__":
    main()
