"""FusionModelE09: early fusion with partial vision fine-tuning.

Unlike E08 (fully frozen vision) and E04 (fully trainable vision),
this model unfreezes only denseblock4 and norm5 while keeping
denseblock3 frozen. This gives a middle ground where vision features
can be fine-tuned during fusion training without the full compute
cost of E04.

Architecture:
- Vision: build_vision_baseline_frozen(unfreeze_from_block="denseblock3")
  with further unfreezing of denseblock4 + norm5.
  Classifier replaced with nn.Identity() -> returns 1024-dim embedding.
- Clinical encoder: Linear(29,64)->ReLU->Dropout(0.3)->Linear(64,32)->ReLU
- Fusion: concat [1024-dim vision embedding + 32-dim clinical embedding] -> 1056-dim
- Fusion head: Linear(1056,128)->ReLU->Dropout(0.3)->Linear(128,5) logits.
"""

from __future__ import annotations

import logging
import torch
import torch.nn as nn

from torchvision import models

from src.models.vision_baseline import build_vision_baseline_frozen


LOGGER = logging.getLogger(__name__)


def _unfurther_unfreeze(model: nn.Module) -> None:
    """Further unfreeze only denseblock4 and norm5, keeping denseblock3 frozen.

    After build_vision_baseline_frozen(unfreeze_from_block="denseblock3"),
    denseblock3 + denseblock4 + norm5 + classifier are trainable.
    This function sets requires_grad=False for denseblock3 specifically,
    leaving only denseblock4 + norm5 + classifier trainable.
    """
    for name, parameter in model.features.named_parameters():
        # denseblock3 is the first block; zero-out its requires_grad
        if name.startswith("denseblock3."):
            parameter.requires_grad = False
        # denseblock4 and norm5 stay trainable (already true from frozen baseline)


class FusionModelE09(nn.Module):
    """Fusion model with partial vision fine-tuning (denseblock4+norm5 unfrozen).

    Unlike E08 (fully frozen vision encoder), this model unfreezes
    denseblock4 and norm5 so the vision features can be updated during
    fusion training. denseblock3 remains frozen.

    Takes a RAW IMAGE (3, 224, 224) as input, runs it through the
    partially-unfrozen vision backbone to get a 1024-dim embedding,
    then proceeds through the clinical encoder + fusion head.
    """

    def __init__(
        self,
        num_features: int = 29,
        vision_dim: int = 1024,
        hidden_dim: int = 128,
        num_labels: int = 5,
    ) -> None:
        super().__init__()

        # Build vision backbone with E03-style freeze (from denseblock3 onwards trainable)
        self.vision = build_vision_baseline_frozen(unfreeze_from_block="denseblock3")

        # Further unfreeze only denseblock4 and norm5; denseblock3 stays frozen
        _unfurther_unfreeze(self.vision)

        # Replace classifier with Identity so forward returns 1024-dim embedding
        self.vision.classifier = nn.Identity()

        # Clinical encoder (always trainable)
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

        # Count and print trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        trainable_percent = 100.0 * trainable_params / total_params if total_params else 0.0

        # Count vision-specific trainable params
        vision_trainable = sum(
            p.numel() for p in self.vision.parameters() if p.requires_grad
        )
        vision_total = sum(p.numel() for p in self.vision.parameters())
        vision_percent = 100.0 * vision_trainable / vision_total if vision_total else 0.0

        LOGGER.info(
            f"FusionModelE09: {trainable_params:,}/{total_params} params trainable ({trainable_percent:.2f}%). "
            f"Vision trainable: {vision_trainable:,}/{vision_total} ({vision_percent:.2f}%). "
            f"(denseblock4+norm5 unfrozen, denseblock3 frozen)"
        )

    def forward(self, image: torch.Tensor, clinical_feat: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        image: (batch, 3, 224, 224) raw input image
        clinical_feat: (batch, 29) clinical features
        returns: (batch, 5) logits
        """
        # Vision backbone: image -> 1024-dim embedding (classifier is Identity)
        vision_emb = self.vision(image)  # shape: (batch, 1024)

        # Clinical encoding
        clinical_embedding = self.clinical_encoder(clinical_feat)  # shape: (batch, 32)

        # Concatenate [vision + clinical] -> 1056-dim
        fused_input = torch.cat([vision_emb, clinical_embedding], dim=1)

        # Fusion head -> logits
        logits = self.fusion_head(fused_input)  # shape: (batch, 5)
        return logits