"""Phase 4 training loop for the image-only DenseNet-121 baseline."""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import nn
from torch.utils.data import DataLoader

from src.data.image_dataset import LABEL_COLUMNS, ChestXrayDataset
from src.models.vision_baseline import build_vision_baseline


LOGGER = logging.getLogger(__name__)

DEFAULTS: dict[str, Any] = {
    "batch_size": 16,
    "learning_rate": 0.0001,
    "max_epochs": 15,
    "early_stopping_patience": 4,
    "manifest_path": "data/processed/image_subset_manifest.parquet",
    "checkpoint_path": "models/vision_baseline_best.pt",
    "metrics_output_path": "models/vision_baseline_best_metrics.json",
}
EXPERIMENT_LOG_RELATIVE_PATH = Path("docs") / "experiment_log.md"


def configure_logging() -> None:
    """Configure concise progress logging for command-line execution."""
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def load_config(config_path: str | Path) -> tuple[dict[str, Any], Path]:
    """Load the YAML config with defaults, returning the repository root.

    Relative paths in the YAML are resolved from the repository root (the
    parent of ``configs/``), so the command works from any directory.
    """
    config_file = Path(config_path).resolve()
    if not config_file.is_file():
        raise FileNotFoundError(f"Configuration file was not found: {config_file}")
    with config_file.open("r", encoding="utf-8") as stream:
        user_config = yaml.safe_load(stream) or {}
    if not isinstance(user_config, dict):
        raise ValueError("The YAML configuration must contain a key/value mapping.")
    config = {**DEFAULTS, **user_config}
    return config, config_file.parent.parent


def resolve_config_path(path_value: str, repository_root: Path) -> Path:
    """Resolve an absolute or repository-relative path from the YAML config."""
    path = Path(path_value)
    return path if path.is_absolute() else repository_root / path


def compute_label_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[dict[str, float | None], dict[str, float | None], float, float]:
    """Compute per-label AUROC/AUPRC, skipping labels missing a class.

    A val split with only one class present for a label breaks AUROC, so
    each label is wrapped in try/except: a warning is logged and that
    label's metric is recorded as None rather than crashing training.
    """
    aurocs: dict[str, float | None] = {}
    auprcs: dict[str, float | None] = {}
    for index, label in enumerate(LABEL_COLUMNS):
        try:
            aurocs[label] = float(roc_auc_score(y_true[:, index], y_prob[:, index]))
        except Exception as error:
            LOGGER.warning("Skipping AUROC for %s: %s", label, error)
            aurocs[label] = None
        try:
            auprcs[label] = float(average_precision_score(y_true[:, index], y_prob[:, index]))
        except Exception as error:
            LOGGER.warning("Skipping AUPRC for %s: %s", label, error)
            auprcs[label] = None
    valid_auroc = [value for value in aurocs.values() if value is not None]
    valid_auprc = [value for value in auprcs.values() if value is not None]
    mean_auroc = float(np.mean(valid_auroc)) if valid_auroc else float("nan")
    mean_auprc = float(np.mean(valid_auprc)) if valid_auprc else float("nan")
    return aurocs, auprcs, mean_auroc, mean_auprc


def compute_sensitivity_specificity(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, dict[str, float]]:
    """Compute sensitivity and specificity per label at a probability threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    results: dict[str, dict[str, float]] = {}
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
        results[label] = {"sensitivity": float(sensitivity), "specificity": float(specificity)}
    return results


def append_experiment_log(repository_root: Path, best_record: dict[str, Any]) -> None:
    """Append one E01 row to the markdown experiment log table.

    Reads the existing ``docs/experiment_log.md``, appends a single table
    row, and writes it back. Never raises: a missing log file or any other
    failure only logs a warning so training results are never lost.
    """
    try:
        log_path = repository_root / EXPERIMENT_LOG_RELATIVE_PATH
        if not log_path.is_file():
            LOGGER.warning(
                "Experiment log not found at %s; skipping log update.", log_path
            )
            return

        per_label_auroc = best_record.get("per_label_auroc", {}) or {}
        valid_aurocs = {
            label: value
            for label, value in per_label_auroc.items()
            if value is not None
        }
        if valid_aurocs:
            best_label = max(valid_aurocs, key=lambda label: valid_aurocs[label])
            worst_label = min(valid_aurocs, key=lambda label: valid_aurocs[label])
            best_name = best_label.removesuffix("_label")
            worst_name = worst_label.removesuffix("_label")
            notes = (
                f"Best: {best_name} ({valid_aurocs[best_label]:.4f}), "
                f"Worst: {worst_name} ({valid_aurocs[worst_label]:.4f})."
            )
        else:
            notes = "No per-label AUROC available."

        row = (
            f"| E01 | {date.today().isoformat()} | 4 | "
            f"DenseNet-121 (ImageNet pretrained) | "
            f"baseline, no augmentation beyond h-flip, no class weighting | "
            f"{float(best_record['mean_auroc']):.4f} | "
            f"{float(best_record['mean_auprc']):.4f} | {notes} |"
        )
        content = log_path.read_text(encoding="utf-8")
        if content and not content.endswith("\n"):
            content += "\n"
        content += row + "\n"
        log_path.write_text(content, encoding="utf-8")
        LOGGER.info("Appended experiment E01 to %s", log_path)
    except Exception as error:
        LOGGER.warning("Failed to update experiment log: %s", error)


def run_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run one train (with optimizer) or eval (without) epoch over a loader."""
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    all_targets: list[np.ndarray] = []
    all_probabilities: list[np.ndarray] = []

    for images, targets in loader:
        images = images.to(device)
        targets = targets.to(device)
        if training:
            optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, targets)
        if training:
            loss.backward()
            optimizer.step()
        batch_size = images.size(0)
        total_loss += float(loss.item()) * batch_size
        total_samples += batch_size
        all_targets.append(targets.detach().cpu().numpy())
        all_probabilities.append(torch.sigmoid(logits).detach().cpu().numpy())

    average_loss = total_loss / total_samples if total_samples else float("nan")
    y_true = np.concatenate(all_targets, axis=0) if all_targets else np.zeros((0, len(LABEL_COLUMNS)))
    y_prob = (
        np.concatenate(all_probabilities, axis=0)
        if all_probabilities
        else np.zeros((0, len(LABEL_COLUMNS)))
    )
    return average_loss, y_true, y_prob


def train_model(config_path: str | Path) -> dict[str, Any]:
    """Train the baseline, checkpoint on mean val AUROC, and report operating points."""
    config, repository_root = load_config(config_path)
    batch_size = int(config["batch_size"])
    learning_rate = float(config["learning_rate"])
    max_epochs = int(config["max_epochs"])
    patience = int(config["early_stopping_patience"])
    manifest_path = resolve_config_path(str(config["manifest_path"]), repository_root)
    checkpoint_path = resolve_config_path(str(config["checkpoint_path"]), repository_root)
    metrics_output_path = resolve_config_path(str(config["metrics_output_path"]), repository_root)

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is not available. This baseline must run on GPU: reinstall "
            "PyTorch with CUDA support (not the CPU-only build) using the "
            "install selector at https://pytorch.org/ before training."
        )
    device = torch.device("cuda")
    LOGGER.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0))

    train_dataset = ChestXrayDataset(manifest_path, split="train")
    val_dataset = ChestXrayDataset(manifest_path, split="val")
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=2
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, num_workers=2
    )
    LOGGER.info(
        "Loaded datasets: train=%d images, val=%d images, batch_size=%d",
        len(train_dataset),
        len(val_dataset),
        batch_size,
    )

    model = build_vision_baseline().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_mean_auroc = float("-inf")
    best_record: dict[str, Any] = {}
    epochs_without_improvement = 0

    for epoch in range(1, max_epochs + 1):
        train_loss, _, _ = run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, y_true, y_prob = run_epoch(model, val_loader, criterion, device)
        aurocs, auprcs, mean_auroc, mean_auprc = compute_label_metrics(y_true, y_prob)

        is_best = bool(aurocs and mean_auroc > best_mean_auroc)
        if is_best:
            best_mean_auroc = mean_auroc
            epochs_without_improvement = 0
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            best_record = {
                "epoch": epoch,
                "mean_auroc": mean_auroc,
                "mean_auprc": mean_auprc,
                "per_label_auroc": aurocs,
                "per_label_auprc": auprcs,
                "val_loss": val_loss,
                "train_loss": train_loss,
            }
            metrics_output_path.parent.mkdir(parents=True, exist_ok=True)
            with metrics_output_path.open("w", encoding="utf-8") as stream:
                json.dump(best_record, stream, indent=2)
        else:
            epochs_without_improvement += 1

        LOGGER.info(
            "Epoch %d/%d | train_loss=%.4f val_loss=%.4f mean_val_auroc=%.4f%s",
            epoch,
            max_epochs,
            train_loss,
            val_loss,
            mean_auroc,
            " <- NEW BEST" if is_best else "",
        )

        if epochs_without_improvement >= patience:
            LOGGER.info(
                "Early stopping: mean val AUROC did not improve for %d consecutive "
                "epochs. Stopping training.",
                patience,
            )
            break

    if not best_record:
        raise RuntimeError("Training produced no checkpoint; mean val AUROC never improved.")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    _, y_true, y_prob = run_epoch(model, val_loader, criterion, device)
    operating_points = compute_sensitivity_specificity(y_true, y_prob, threshold=0.5)
    LOGGER.info("Val sensitivity/specificity at 0.5 threshold (best checkpoint):")
    for label in LABEL_COLUMNS:
        LOGGER.info(
            "  %s: sensitivity=%.4f specificity=%.4f",
            label,
            operating_points[label]["sensitivity"],
            operating_points[label]["specificity"],
        )
    best_record["sensitivity_specificity_at_0_5"] = operating_points
    with metrics_output_path.open("w", encoding="utf-8") as stream:
        json.dump(best_record, stream, indent=2)
    try:
        append_experiment_log(repository_root, best_record)
    except Exception as error:
        LOGGER.warning("Failed to update experiment log: %s", error)
    return best_record


def main(argv: list[str] | None = None) -> None:
    """Parse CLI arguments and train the configured vision baseline."""
    parser = argparse.ArgumentParser(description="Train the Phase 4 vision baseline.")
    parser.add_argument(
        "--config",
        default=str(Path(__file__).resolve().parents[2] / "configs" / "vision_baseline_config.yaml"),
        help="Path to YAML training configuration (default: configs/vision_baseline_config.yaml).",
    )
    args = parser.parse_args(argv)
    configure_logging()
    train_model(args.config)


if __name__ == "__main__":
    main()
