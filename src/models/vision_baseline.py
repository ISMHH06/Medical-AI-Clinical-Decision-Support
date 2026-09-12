"""DenseNet-121 image-only baseline model for Phase 4."""

from __future__ import annotations

import torch.nn as nn
from torchvision import models


def build_vision_baseline(num_labels: int = 5) -> nn.Module:
    """Return an ImageNet-pretrained DenseNet-121 with a 5-logit classifier.

    The final classifier is replaced with ``nn.Linear(in_features, 5)``
    where ``in_features`` is read from the original classifier layer so it
    is never hardcoded. No sigmoid is applied: the model outputs raw logits
    for use with ``BCEWithLogitsLoss``.
    """
    model = models.densenet121(weights="IMAGENET1K_V1")
    in_features = model.classifier.in_features
    model.classifier = nn.Linear(in_features, num_labels)
    return model
