"""Clinical MLP baseline model for Phase 6 tabular modeling."""

from __future__ import annotations

import torch.nn as nn


def build_clinical_baseline(input_dim: int, num_labels: int = 5) -> nn.Module:
    """Return a simple MLP for clinical tabular features.

    Architecture: Linear(input_dim, 64) -> ReLU -> Dropout(0.3)
    -> Linear(64, 32) -> ReLU -> Dropout(0.3) -> Linear(32, num_labels).

    No sigmoid applied: outputs raw logits for use with BCEWithLogitsLoss.
    """
    return nn.Sequential(
        nn.Linear(input_dim, 64),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(64, 32),
        nn.ReLU(),
        nn.Dropout(0.3),
        nn.Linear(32, num_labels),
    )