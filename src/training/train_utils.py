"""Shared training utilities for Phase 4-7 experiments.

Provides common functions: train_one_epoch, evaluate, compute_pos_weight_from_targets,
save_checkpoint, load_checkpoint, write_experiment_log_entry.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

from src.data.image_dataset import LABEL_COLUMNS

LOGGER = logging.getLogger(__name__)


def train_one_epoch(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one training epoch.

    Returns:
        average_loss, y_true, y_prob (probabilities via sigmoid of logits).
    """
    model.train()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []
    total_batches = len(loader)

    for batch_idx, (vision_emb, clinical_feat, targets) in enumerate(loader, start=1):
        vision_emb = vision_emb.to(device)
        clinical_feat = clinical_feat.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits = model(vision_emb, clinical_feat)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = vision_emb.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        if batch_idx % 50 == 0:
            running_loss = total_loss / total_samples if total_samples else float("nan")
            print(f"  Batch {batch_idx}/{total_batches} | running loss: {running_loss:.4f}")
        all_targets.append(targets.detach().cpu().numpy())
        all_logits.append(logits.detach().cpu().numpy())

    average_loss = total_loss / total_samples if total_samples else float("nan")
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 5))
    y_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 5))
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))  # sigmoid
    return average_loss, y_true, y_prob


def evaluate(
    model: torch.nn.Module,
    loader: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> tuple[float, float, float, float]:
    """Evaluate model on validation/test data.

    Returns:
        mean_auroc, mean_auprc, sensitivity, specificity (at 0.5 threshold).
    """
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_logits: list[np.ndarray] = []

    with torch.no_grad():
        for vision_emb, clinical_feat, targets in loader:
            vision_emb = vision_emb.to(device)
            clinical_feat = clinical_feat.to(device)
            targets = targets.to(device)

            logits = model(vision_emb, clinical_feat)
            loss = criterion(logits, targets)

            batch_size = vision_emb.size(0)
            total_loss += float(loss.item()) * batch_size
            total_samples += batch_size
            all_targets.append(targets.detach().cpu().numpy())
            all_logits.append(logits.detach().cpu().numpy())

    average_loss = total_loss / total_samples if total_samples else float("nan")
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, 5))
    y_logits = np.concatenate(all_logits, axis=0) if all_logits else np.zeros((0, 5))
    y_prob = 1.0 / (1.0 + np.exp(-y_logits))  # sigmoid

    # Compute mean AUROC and AUPRC across labels
    valid_aurocs: list[float] = []
    valid_auprcs: list[float] = []
    for label_idx in range(y_true.shape[1]):
        try:
            from sklearn.metrics import roc_auc_score, average_precision_score

            auroc = float(roc_auc_score(y_true[:, label_idx], y_prob[:, label_idx]))
            auprc = float(average_precision_score(y_true[:, label_idx], y_prob[:, label_idx]))
            valid_aurocs.append(auroc)
            valid_auprcs.append(auprc)
        except Exception as e:
            LOGGER.warning("Could not compute metrics for label %d: %s", label_idx, e)

    mean_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
    mean_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")

    # Compute sensitivity and specificity at 0.5 threshold for ALL labels
    y_pred = (y_prob >= 0.5).astype(int)
    sensitivity_specificity: dict[str, dict[str, float]] = {}
    for index, label in enumerate(LABEL_COLUMNS):
        true_positive = int(((y_pred[:, index] == 1) & (y_true[:, index] == 1)).sum())
        true_negative = int(((y_pred[:, index] == 0) & (y_true[:, index] == 0)).sum())
        false_positive = int(((y_pred[:, index] == 1) & (y_true[:, index] == 0)).sum())
        false_negative = int(((y_pred[:, index] == 0) & (y_true[:, index] == 1)).sum())
        sensitivity = (
            true_positive / (true_positive + false_negative)
            if (true_positive + false_negative) > 0
            else float("nan")
        )
        specificity = (
            true_negative / (true_negative + false_positive)
            if (true_negative + false_positive) > 0
            else float("nan")
        )
        sensitivity_specificity[label] = {
            "sensitivity": float(sensitivity),
            "specificity": float(specificity),
        }

    return mean_auroc, mean_auprc, sensitivity_specificity


def compute_pos_weight_from_targets(train_targets: torch.Tensor) -> torch.Tensor:
    """Compute pos_weight for BCEWithLogitsLoss from training multi-label targets.

    pos_weight is a 1-element per-label tensor computed as
    (number of negative samples) / (number of positive samples) for each label.
    Returns a tensor of shape (5,) for the 5 disease labels.
    """
    # train_targets shape: (N, 5), binary
    num_positive = train_targets.sum(dim=0).float()  # shape (5,)
    num_negative = train_targets.size(0) - num_positive
    # Avoid division by zero
    eps = 1e-6
    pos_weight = (num_negative + eps) / (num_positive + eps)
    # Cap at 20 to avoid extremely large weights
    pos_weight = torch.min(pos_weight, torch.tensor(20.0))
    return pos_weight


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer | None,
    checkpoint_path: str,
    metrics_path: str,
) -> None:
    """Save model checkpoint and associated metrics."""
    state: dict[str, Any] = {
        "state_dict": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer else None,
    }
    ckpt_dir = Path(checkpoint_path).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, checkpoint_path)
    LOGGER.info("Saved checkpoint to %s", checkpoint_path)


def load_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[torch.nn.Module, torch.optim.Optimizer | None]:
    """Load model checkpoint. Returns (model, optimizer) after loading."""
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    LOGGER.info("Loaded checkpoint from %s", checkpoint_path)
    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
        LOGGER.info("Loaded optimizer state")
    return model, optimizer


def write_experiment_log_entry(
    experiment_id: str,
    phase: str,
    model_name: str,
    changes: str,
    auroc: float,
    auprc: float,
    notes: str,
) -> None:
    """Append one experiment row to docs/experiment_log.md.

    The row format follows the existing table conventions. If the log file
    does not exist, it is created with a header.
    """
    import pathlib

    log_path = pathlib.Path("docs/experiment_log.md")
    row = (
        f"| {experiment_id} | 2026-09-17 | {phase} | {model_name} | {changes} | "
        f"{auroc:.4f} | {auprc:.4f} | {notes} |"
    )

    # Ensure the header exists
    if not log_path.is_file():
        log_path.write_text(
            "# Experiment Log\n\n"
            "Track every modeling experiment here. One row per experiment.\n\n"
            "| ID | Date | Phase | Model | Changes | AUROC | AUPRC | Notes |\n"
            "| -- | ---- | ----- | ----- | ------- | ----- | ----- | ----- |\n",
            encoding="utf-8",
        )

    # Append the row
    content = log_path.read_text(encoding="utf-8")
    # Check if a row with this experiment_id already exists
    if f"| {experiment_id} |" in content:
        LOGGER.warning("Experiment %s already in log; skipping duplicate write.", experiment_id)
        return
    if content and not content.endswith("\n"):
        content += "\n"
    content += row + "\n"
    log_path.write_text(content, encoding="utf-8")
    LOGGER.info("Appended experiment %s to %s", experiment_id, log_path)