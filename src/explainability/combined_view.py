"""Combined per-patient explanation view for E08 Grad-CAM + SHAP."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from matplotlib.figure import Figure
from matplotlib.gridspec import GridSpec

from src.data.image_dataset import build_transform
from src.explainability.gradcam import E08Explainer, generate_gradcam
from src.explainability.shap_explainer import generate_shap_values
from src.models.fusion_model import FusionModel


def _top_shap_values(values: np.ndarray, feature_names: list[str], limit: int = 5) -> tuple[np.ndarray, list[str]]:
    """Return the strongest absolute SHAP values and corresponding feature names."""
    order = np.argsort(np.abs(values))[-limit:][::-1]
    return values[order], [feature_names[i] for i in order]


def build_patient_figure(
    row_index: int,
    image_path: str,
    explainer: E08Explainer,
    fusion_model: FusionModel,
    vision_emb: torch.Tensor,
    clinical_feat: torch.Tensor,
    background_feats: np.ndarray,
    feature_names: list[str],
    label_names: list[str],
    ground_truth_labels: np.ndarray,
    device: torch.device,
) -> Figure:
    """Build a combined 2x5 patient explanation figure across all 5 labels."""
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        real_image_tensor = build_transform("val")(image)
    real_image_tensor = real_image_tensor.unsqueeze(0).to(device)

    with torch.no_grad():
        logits = explainer(real_image_tensor, clinical_feat.unsqueeze(0).to(device))
    probabilities = torch.sigmoid(logits[0]).detach().cpu().numpy()

    fig = plt.figure(figsize=(18, 8))
    gs = GridSpec(2, 5, figure=fig)

    for label_idx in range(5):
        heatmap, overlay_image = generate_gradcam(
            explainer=explainer,
            image_path=image_path,
            clinical_feat=clinical_feat,
            label_idx=label_idx,
            device=device,
        )
        shap_values = generate_shap_values(
            fusion_model=fusion_model,
            vision_emb=vision_emb,
            background_clinical_feats=background_feats,
            sample_clinical_feat=clinical_feat.numpy(),
            label_idx=label_idx,
            device=device,
        )

        prob = float(probabilities[label_idx])
        ax_img = fig.add_subplot(gs[0, label_idx])
        ax_img.imshow(overlay_image)
        ax_img.set_axis_off()
        title = f"{label_names[label_idx]}\np={prob:.2f} (GT={int(ground_truth_labels[label_idx])})"
        ax_img.set_title(title, fontweight="bold" if ground_truth_labels[label_idx] == 1 else "normal")

        ax_shap = fig.add_subplot(gs[1, label_idx])
        top_values, top_names = _top_shap_values(shap_values, feature_names)
        colors = ["tab:red" if value > 0 else "tab:blue" for value in top_values]
        ax_shap.barh(top_names, top_values, color=colors)
        ax_shap.axvline(0.0, color="black", linewidth=1)
        ax_shap.set_xlabel("SHAP")
        ax_shap.invert_yaxis()
        ax_shap.set_title(f"{label_names[label_idx]} SHAP")

    fig.suptitle(f"Patient (val row {row_index})")
    fig.tight_layout()
    return fig
