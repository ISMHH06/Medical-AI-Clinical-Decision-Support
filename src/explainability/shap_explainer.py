"""SHAP-based explanations for the E08 clinical branch."""

from __future__ import annotations

from typing import Any

import numpy as np
import shap
import torch
import torch.nn as nn

from src.models.fusion_model import FusionModel


class ClinicalOnlyWrapper(nn.Module):
    """Freeze the vision embedding and expose only the clinical branch to SHAP."""

    def __init__(self, fusion_model: FusionModel, vision_emb: torch.Tensor) -> None:
        super().__init__()
        self.fusion_model = fusion_model
        self.register_buffer("vision_emb", vision_emb.float().view(1, -1))

    def forward(self, clinical_feat: torch.Tensor) -> torch.Tensor:
        """Expand the fixed vision embedding across the batch and route only clinical inputs."""
        clinical_feat = torch.as_tensor(clinical_feat, dtype=torch.float32)
        if clinical_feat.dim() == 1:
            clinical_feat = clinical_feat.unsqueeze(0)
        batch_size = clinical_feat.shape[0]
        vision_emb_expanded = self.vision_emb.expand(batch_size, -1)
        return self.fusion_model(vision_emb_expanded, clinical_feat)


def generate_shap_values(
    fusion_model: FusionModel,
    vision_emb: torch.Tensor,
    background_clinical_feats: np.ndarray | torch.Tensor,
    sample_clinical_feat: np.ndarray | torch.Tensor,
    label_idx: int,
    device: torch.device | None = None,
) -> np.ndarray:
    """Return SHAP values for one clinical feature vector and one output label."""
    if device is None:
        device = next(fusion_model.parameters()).device

    wrapper = ClinicalOnlyWrapper(fusion_model, torch.as_tensor(vision_emb, dtype=torch.float32))
    wrapper = wrapper.to(device)
    wrapper.eval()

    background = torch.as_tensor(background_clinical_feats, dtype=torch.float32, device=device)
    sample = torch.as_tensor(sample_clinical_feat, dtype=torch.float32, device=device)
    if background.dim() == 1:
        background = background.unsqueeze(0)
    if sample.dim() == 1:
        sample = sample.unsqueeze(0)

    explainer = shap.DeepExplainer(wrapper, background)
    shap_values = explainer.shap_values(sample, check_additivity=False)
    raw_shape = np.asarray(shap_values).shape
    print(f"[DEBUG] raw SHAP output shape: {raw_shape}")

    values: Any = shap_values
    if isinstance(values, list):
        values = np.asarray(values)
    if isinstance(values, np.ndarray):
        if values.ndim == 3:
            values = values[0, :, label_idx]
        elif values.ndim == 2:
            if values.shape[0] == 5 and values.shape[1] == sample.shape[1]:
                values = values[label_idx, :]
            elif values.shape[0] == sample.shape[0] and values.shape[1] == sample.shape[1]:
                values = values[0, :]
        elif values.ndim == 1:
            if values.shape[0] == 5:
                values = values[label_idx]
    values_array = np.asarray(values, dtype=np.float32)
    return values_array.reshape(-1)
