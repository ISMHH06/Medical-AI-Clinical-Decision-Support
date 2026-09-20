"""Training loop for Phase 7 E08: Early fusion with frozen vision + trainable clinical encoder."""

from __future__ import annotations

import logging
import os
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.fusion_dataset import FusionDataset
from src.models.fusion_model import FusionModel
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
    batch_size = 32
    learning_rate = 0.001
    max_epochs = 30
    early_stopping_patience = 5
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Setup
    torch.manual_seed(42)

    # Load precomputed vision embeddings for all splits
    vision_embeddings = {
        "train": np.load("models/vision_embeddings_train.npy"),
        "val": np.load("models/vision_embeddings_val.npy"),
    }

    # Create datasets
    train_dataset = FusionDataset(
        str(MANIFEST_PATH), "train", "models/vision_embeddings_train.npy"
    )
    val_dataset = FusionDataset(
        str(MANIFEST_PATH), "val", "models/vision_embeddings_val.npy"
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Compute pos_weight from training targets
    train_targets = train_dataset.targets  # (N, 5) torch.Tensor
    pos_weight = compute_pos_weight_from_targets(train_targets)
    pos_weight = pos_weight.to(device)

    # Initialize fusion model
    model = FusionModel().to(device)

    # Optimizer and scheduler
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=2)

    # Loss function (BCEWithLogitsLoss with pos_weight)
    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

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

        # Evaluate on validation
        val_auc, val_ap, val_sens_spec = evaluate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_auc)

        val_aucs.append(val_auc)
        print(
            f"Epoch {epoch:02d} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val AUROC: {val_auc:.4f} (AP: {val_ap:.4f}) | "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        # Check for improvement
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(
                model,
                optimizer,
                f"models/fusion_e08_best.pt",
                f"models/fusion_e08_best_metrics.json",
            )
            print(f"  -> Saved best checkpoint (AUROC: {best_val_auc:.4f})")
        else:
            epochs_without_improvement += 1

        # Early stopping
        if epochs_without_improvement >= early_stopping_patience:
            print(f"Early stopping at epoch {epoch} (no improvement for {early_stopping_patience} epochs)")
            break

    # Load best checkpoint and evaluate on val set
    model, _ = load_checkpoint(model, "models/fusion_e08_best.pt")
    val_auc, val_ap, val_sens_spec = evaluate(model, val_loader, criterion, device)
    print(f"\nBest validation AUROC: {best_val_auc:.4f} (at epoch {best_epoch})")
    print(f"Val AUROC: {val_auc:.4f} (AP: {val_ap:.4f})")
    for label, metrics in val_sens_spec.items(): 
        print(f"  {label}: sensitivity={metrics['sensitivity']:.4f} specificity={metrics['specificity']:.4f}")

    # Compute signed difference against vision-only E04 (val AUROC)
    vision_e04_val_auc = 0.7267
    diff = val_auc - vision_e04_val_auc
    print(f"Diff vs vision-only E04 val (0.7267): {diff:+.4f}")

    # Append to experiment log (val AUROC/AUPRC, consistent with E01-E07)
    write_experiment_log_entry(
        experiment_id="E08",
        phase="7",
        model_name="Early fusion (frozen DenseNet-121 embeddings + trainable clinical encoder + fusion MLP)",
        changes="concatenation-based early fusion: 1024-dim frozen vision embedding + 32-dim learned clinical embedding -> 128-dim fusion layer -> 5 logits, pos_weight class balancing",
        auroc=val_auc,
        auprc=val_ap,
        notes=f"Best val AUROC: {best_val_auc:.4f} (epoch {best_epoch}); val AUROC: {val_auc:.4f} vs vision-only E04 val 0.7267 {diff:+.4f}. Early fusion with frozen vision embeddings and trainable clinical encoder.",
    )
    print("\nExperiment log entry E08 appended.")


if __name__ == "__main__":
    main()