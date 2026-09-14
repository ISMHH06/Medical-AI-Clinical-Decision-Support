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


DENSE_BLOCK_ORDER = ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]


def build_vision_baseline_frozen(
    num_labels: int = 5, unfreeze_from_block: str = "denseblock3"
) -> nn.Module:
    """Return the baseline with early DenseNet blocks frozen for fine-tuning.

    Builds the same ImageNet-pretrained DenseNet-121 as
    :func:`build_vision_baseline`, then sets ``requires_grad=False`` for all
    parameters in ``model.features`` except those in ``unfreeze_from_block``,
    later dense blocks, and the final ``norm5`` layer. With the default
    ``unfreeze_from_block="denseblock3"``, only denseblock3, denseblock4,
    norm5, and the new classifier stay trainable. The classifier is always
    left trainable. Prints total/trainable parameter counts for confirmation.
    """
    if unfreeze_from_block not in DENSE_BLOCK_ORDER:
        raise ValueError(
            f"unfreeze_from_block must be one of {DENSE_BLOCK_ORDER}, "
            f"got {unfreeze_from_block!r}."
        )
    model = build_vision_baseline(num_labels=num_labels)
    first_trainable = DENSE_BLOCK_ORDER.index(unfreeze_from_block)
    trainable_blocks = set(DENSE_BLOCK_ORDER[first_trainable:]) | {"norm5"}
    for name, parameter in model.features.named_parameters():
        top_level = name.split(".")[0]
        parameter.requires_grad = top_level in trainable_blocks

    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    percent = 100 * trainable / total if total else 0.0
    print(
        f"Frozen baseline: {trainable:,}/{total:,} parameters trainable ({percent:.2f}%). "
        f"Trainable feature blocks: {sorted(trainable_blocks)} + classifier."
    )
    return model
