"""Fusion model: frozen DenseNet-121 vision embedding + trainable clinical encoder + fusion MLP."""

from __future__ import annotations

import torch
import torch.nn as nn


class FusionModel(nn.Module):
    """Fusion model: frozen vision embedding + trainable clinical encoder + fusion MLP.

    Takes vision_embedding (batch, 1024) and clinical_features (batch, 29) as
    two separate forward() inputs.

    Architecture:
    - Clinical encoder: Linear(29, 64) -> ReLU -> Dropout(0.3) -> Linear(64, 32) -> ReLU
      (produces a 32-dim clinical embedding).
    - Concatenates the 1024-dim vision embedding with the 32-dim clinical
      embedding (1056-dim total).
    - Fusion head: Linear(1056, 128) -> ReLU -> Dropout(0.3) -> Linear(128, 5)
      (outputs raw logits, no final activation).
    """

    def __init__(
        self,
        num_features: int = 29,
        vision_dim: int = 1024,
        hidden_dim: int = 128,
        num_labels: int = 5,
    ) -> None:
        super().__init__()
        # Clinical encoder (trainable)
        self.clinical_encoder = nn.Sequential(
            nn.Linear(num_features, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 32),
            nn.ReLU(),
        )
        # Fusion head
        self.fusion_head = nn.Sequential(
            nn.Linear(vision_dim + 32, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, num_labels),
        )
        self.vision_dim = vision_dim
        self.num_features = num_features

    def forward(self, vision_emb: torch.Tensor, clinical_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        vision_emb: (batch, vision_dim)   -- frozen DenseNet embedding
        clinical_feat: (batch, 29)       -- clinical features
        returns: (batch, 5) logits
        """
        clinical_embedding = self.clinical_encoder(clinical_feat)
        fused_input = torch.cat([vision_emb, clinical_embedding], dim=1)
        logits = self.fusion_head(fused_input)
        return logits