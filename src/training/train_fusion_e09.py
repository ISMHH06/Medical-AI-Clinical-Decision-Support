"""Training loop for Phase 7 E09: Early fusion with partial vision fine-tuning.

Unfreezes denseblock4+norm5 of DenseNet-121 while keeping denseblock3 frozen,
enabling vision feature fine-tuning during fusion training (unlike E08 which kept
the vision encoder fully frozen).

Since this involves real CNN forward/backward passes (not just MLP over precomputed
embeddings), uses batch size 16 and LR 0.0001 matching E04's vision training settings.

Mirrored from train_fusion_e08.py: uses evaluate()'s 3-value return
(mean_auroc, mean_auprc, sensitivity_specificity dict), no manual
per-label recomputation after training.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.fusion_dataset_e09 import FusionDatasetE09
from src.models.fusion_model_e09 import FusionModelE09
from src.training.train_utils import (
    compute_pos_weight_from_targets,
    train_one_epoch,
    evaluate,
    save_checkpoint,
    load_checkpoint,
    write_experiment_log_entry,
)

LOGGER = logging.getLogger(__name__)

MANIFEST_PATH = "data/processed/image_subset_manifest.parquet"


def main() -> None:
    # Configuration
    batch_size = 16
    learning_rate = 0.0001
    max_epochs = 20
    early_stopping_patience = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    LOGGER.info("=" * 70)
    LOGGER.info("Phase 7 E09: Early fusion with partial vision fine-tuning")
    LOGGER.info("  Vision: denseblock4+norm5 unfrozen, denseblock3 frozen")
    LOGGER.info("  Batch size: %d (E04-style vision batch)", batch_size)
    LOGGER.info("  LR: %s (E04-style fine-tuning LR)", learning_rate)
    LOGGER.info("  Max epochs: %d (E04-style)", max_epochs)
    LOGGER.info("  NOTE: This will be noticeably slower than E08 due to real CNN")
    LOGGER.info("        forward/backward passes over raw images (not just MLP over")
    LOGGER.info("        precomputed embeddings). This is expected.")
    LOGGER.info("=" * 70)

    # Setup
    torch.manual_seed(42)

    # Create datasets (raw images + clinical features, no precomputed embeddings)
    train_dataset = FusionDatasetE09(MANIFEST_PATH, "train")
    val_dataset = FusionDatasetE09(MANIFEST_PATH, "val")

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Compute pos_weight from training targets
    train_targets = train_dataset.targets  # (N, 5) torch.Tensor
    pos_weight = compute_pos_weight_from_targets(train_targets)
    pos_weight = pos_weight.to(device)

    # Initialize fusion model with partial vision fine-tuning
    model = FusionModelE09().to(device)

    # Optimizer: Adam over ONLY parameters with requires_grad=True
    # (E09 has frozen vision layers (denseblock3), unlike E08's fully trainable clinical portion)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=learning_rate)

    # Loss function (BCEWithLogitsLoss with pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Training loop
    best_val_auc = -1.0
    best_epoch = -1
    epochs_without_improvement = 0

    val_aucs = []

    for epoch in range(1, max_epochs + 1):
        # Train one epoch
        train_loss, _, _ = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )   

        # Evaluate on validation (3-value return: mean_auroc, mean_auprc, sensitivity_specificity dict)
        val_auc, val_ap, val_sens_spec = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_auc)

        val_aucs.append(val_auc)
        LOGGER.info(
            "Epoch %d/%d | Train Loss: %.4f | Val AUROC: %.4f (AP: %.4f) | LR: %.6f",
            epoch,
            max_epochs,
            train_loss,
            val_auc,
            optimizer.param_groups[0]["lr"],
        )

        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                optimizer,
                f"models/fusion_e09_best.pt",
                f"models/fusion_e09_best_metrics.json",
            )
            LOGGER.info("  -> Saved best checkpoint (AUROC: %.4f)", best_val_auc)
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            LOGGER.info(
                "Early stopping at epoch %d (no improvement for %d epochs)",
                epoch,
                early_stopping_patience,
            )
            break

    # Load best checkpoint and evaluate on val set
    model, _ = load_checkpoint(model, "models/fusion_e09_best.pt")
    val_auc, val_ap, val_sens_spec = evaluate(model, val_loader, criterion, device)
    LOGGER.info(
        "\nBest validation AUROC: %.4f (at epoch %d)",
        best_val_auc,
        best_epoch,
    )
    LOGGER.info("Val AUROC: %.4f (AP: %.4f)", val_auc, val_ap)

    # Log per-label sensitivity/specificity from the built-in evaluate() dict
    LOGGER.info("Per-label sensitivity/specificity at 0.5 threshold:")
    for label, metrics in val_sens_spec.items():
        LOGGER.info(
            "  %s: sensitivity=%.4f specificity=%.4f",
            label,
            metrics["sensitivity"],
            metrics["specificity"],
        )

    # Compare against E04 and E08
    e04_val_auc = 0.7267  # E04 vision-only val AUROC
    e08_val_auc = 0.7306  # E08 late fusion val AUROC (from experiment log)
    diff_vs_e04 = val_auc - e04_val_auc
    diff_vs_e08 = val_auc - e08_val_auc

    LOGGER.info(
        "Comparisons: Val AUROC vs E04 (%.4f): %.4f vs vs E08 (%.4f): %.4f",
        e04_val_auc,
        diff_vs_e04,
        e08_val_auc,
        diff_vs_e08,
    )

    # Append to experiment log
    write_experiment_log_entry(
        experiment_id="E09",
        phase="7",
        model_name="Early fusion with partial vision fine-tuning (denseblock4+norm5 unfrozen, frozen denseblock3, trainable clinical encoder + fusion MLP)",
        changes="extends E08 by unfreezing denseblock4+norm5 during fusion training instead of keeping the vision encoder fully frozen; enables vision fine-tuning with raw images",
        auroc=val_auc,
        auprc=val_ap,
        notes=(
            f"Best val AUROC: {best_val_auc:.4f} (epoch {best_epoch}); "
            f"Val AUROC: {val_auc:.4f} vs E04 val {e04_val_auc:.4f} {diff_vs_e04:+.4f} vs E08 val {e08_val_auc:.4f} {diff_vs_e08:+.4f}. "
            f"Partial fine-tuning: denseblock4+norm5 unfrozen, denseblock3 frozen. "
            f"Batch size 16, LR 0.0001 (fine-tuning settings)."
        ),
    )
    LOGGER.info("Experiment log entry E09 appended.")

    LOGGER.info("=" * 70)
    LOGGER.info("E09 training complete.")
    LOGGER.info("=" * 70)


if __name__ == "__main__":
    main()