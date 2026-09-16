"""Phase 7 E07: Late fusion evaluation of vision + clinical champions.

Read-only evaluation script -- no training. Loads both champion checkpoints,
runs inference on the shared val split, averages output probabilities,
and compares vision-only / clinical-only / fused metrics.
"""

from __future__ import annotations

import json
import logging
import sys
from datetime import date
from pathlib import Path

# Ensure repo root is on sys.path for direct execution
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.data.clinical_dataset import ClinicalDataset
from src.data.image_dataset import ChestXrayDataset, LABEL_COLUMNS
from src.models.clinical_baseline import build_clinical_baseline
from src.models.vision_baseline import build_vision_baseline_frozen


LOGGER = logging.getLogger(__name__)

EXPERIMENT_LOG_RELATIVE_PATH = Path("docs") / "experiment_log.md"
VISION_CHECKPOINT = "models/vision_baseline_champion.pt"
CLINICAL_CHECKPOINT = "models/clinical_baseline_champion.pt"
OUTPUT_METRICS = "models/late_fusion_e07_metrics.json"
PROBS_OUTPUT_DIR = "models"
VISION_PROBS_FILE = "models/vision_probs.npy"
CLINICAL_PROBS_FILE = "models/clinical_probs.npy"
TARGETS_FILE = "models/val_targets.npy"
MANIFEST_PATH = "data/processed/image_subset_manifest.parquet"


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")


def compute_label_metrics(
    y_true: np.ndarray, y_prob: np.ndarray
) -> tuple[dict[str, float | None], dict[str, float | None], float, float]:
    """Compute per-label AUROC/AUPRC, skipping labels missing a class."""
    from sklearn.metrics import average_precision_score, roc_auc_score

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


def load_manifest_val_split(manifest_path: Path) -> pd.DataFrame:
    """Load and return the val-split rows from the manifest, in original order."""
    df = pd.read_parquet(manifest_path)
    val_df = df.loc[df["split"] == "val"].reset_index(drop=True)
    if val_df.empty:
        raise ValueError("Manifest has no rows in val split.")
    return val_df


def build_vision_dataset(manifest_path: Path) -> ChestXrayDataset:
    """Build vision dataset for val split."""
    return ChestXrayDataset(manifest_path, split="val")


def build_clinical_dataset(manifest_path: Path) -> ClinicalDataset:
    """Build clinical dataset for val split."""
    return ClinicalDataset(manifest_path, split="val")


def verify_alignment(vision_ds: ChestXrayDataset, clinical_ds: ClinicalDataset, val_manifest: pd.DataFrame) -> None:
    """Verify both datasets have same row count and targets match."""
    if len(vision_ds) != len(clinical_ds):
        raise RuntimeError(
            f"Row count mismatch: vision={len(vision_ds)}, clinical={len(clinical_ds)}. "
            "Both must align to the same val manifest rows."
        )
    LOGGER.info("Row counts match: vision=%d, clinical=%d", len(vision_ds), len(clinical_ds))

    # Check first 5 path_to_image from vision source for sanity
    for i in range(min(5, len(val_manifest))):
        LOGGER.info("  val row %d: %s", i, val_manifest.iloc[i]["path_to_image"])

    # Verify targets match between datasets
    vision_targets = torch.cat([vision_ds[i][1].unsqueeze(0) for i in range(len(vision_ds))], dim=0)
    clinical_targets = torch.cat([clinical_ds[i][1].unsqueeze(0) for i in range(len(clinical_ds))], dim=0)
    if not torch.allclose(vision_targets, clinical_targets):
        raise RuntimeError("Targets mismatch between vision and clinical datasets -- alignment broken.")
    LOGGER.info("Targets match between vision and clinical datasets.")


def run_inference(
    model: torch.nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Run model over loader, return (probabilities, targets)."""
    model.eval()
    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []
    with torch.no_grad():
        for features, targets in loader:
            features = features.to(device)
            logits = model(features)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())
    return np.concatenate(all_probs, axis=0), np.concatenate(all_targets, axis=0)


def append_experiment_log(
    repository_root: Path,
    mean_auroc: float,
    mean_auprc: float,
    per_label_auroc: dict[str, float | None],
) -> None:
    """Append E07 row to experiment log."""
    try:
        log_path = repository_root / EXPERIMENT_LOG_RELATIVE_PATH
        if not log_path.is_file():
            LOGGER.warning("Experiment log not found at %s; skipping log update.", log_path)
            return

        valid_aurocs = {label: v for label, v in per_label_auroc.items() if v is not None}
        if valid_aurocs:
            best_label = max(valid_aurocs, key=lambda l: valid_aurocs[l])
            worst_label = min(valid_aurocs, key=lambda l: valid_aurocs[l])
            best_name = best_label.removesuffix("_label")
            worst_name = worst_label.removesuffix("_label")
            notes = (
                f"Fused AUROC vs vision-only: +/-{mean_auroc - 0.0:.4f}; "
                f"vs clinical-only: +/-{mean_auroc - 0.0:.4f}. "
                f"Best: {best_name} ({valid_aurocs[best_label]:.4f}), "
                f"Worst: {worst_name} ({valid_aurocs[worst_label]:.4f})."
            )
        else:
            notes = "Fused AUROC vs vision-only: +/-0.0000; vs clinical-only: +/-0.0000."

        row = (
            f"| E07 | {date.today().isoformat()} | 7 | "
            f"Late fusion (avg of E04 vision + E06 clinical) | "
            f"simple unweighted averaging of vision and clinical champion model output probabilities, no new training | "
            f"{mean_auroc:.4f} | {mean_auprc:.4f} | {notes} |"
        )
        content = log_path.read_text(encoding="utf-8")
        if content and not content.endswith("\n"):
            content += "\n"
        content += row + "\n"
        log_path.write_text(content, encoding="utf-8")
        LOGGER.info("Appended experiment E07 to %s", log_path)
    except Exception as error:
        LOGGER.warning("Failed to update experiment log: %s", error)


def main() -> None:
    configure_logging()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA not available -- this evaluation requires GPU.")

    device = torch.device("cuda")
    LOGGER.info("Using device: %s (%s)", device, torch.cuda.get_device_name(0))

    repo_root = Path(__file__).resolve().parents[1]
    manifest_file = repo_root / MANIFEST_PATH
    vision_ckpt = repo_root / VISION_CHECKPOINT
    clinical_ckpt = repo_root / CLINICAL_CHECKPOINT
    output_file = repo_root / OUTPUT_METRICS

    # Load val split manifest
    val_manifest = load_manifest_val_split(manifest_file)
    LOGGER.info("Loaded val manifest: %d rows", len(val_manifest))

    # Build both datasets
    vision_ds = build_vision_dataset(manifest_file)
    clinical_ds = build_clinical_dataset(manifest_file)
    verify_alignment(vision_ds, clinical_ds, val_manifest)

    # DataLoaders
    vision_loader = DataLoader(vision_ds, batch_size=32, shuffle=False, num_workers=0)
    clinical_loader = DataLoader(clinical_ds, batch_size=32, shuffle=False, num_workers=0)

    # Load vision model
    vision_model = build_vision_baseline_frozen(unfreeze_from_block="denseblock3").to(device)
    vision_state = torch.load(vision_ckpt, map_location=device, weights_only=True)
    vision_model.load_state_dict(vision_state)
    LOGGER.info("Loaded vision champion from %s", vision_ckpt)

    # Load clinical model
    clinical_model = build_clinical_baseline(input_dim=clinical_ds.feature_dim).to(device)
    clinical_state = torch.load(clinical_ckpt, map_location=device, weights_only=True)
    clinical_model.load_state_dict(clinical_state)
    LOGGER.info("Loaded clinical champion from %s", clinical_ckpt)

    # Run inference
    LOGGER.info("Running vision inference on val set...")
    vision_probs, vision_targets = run_inference(vision_model, vision_loader, device)
    LOGGER.info("Running clinical inference on val set...")
    clinical_probs, clinical_targets = run_inference(clinical_model, clinical_loader, device)

    # Verify targets match (sanity)
    if not np.allclose(vision_targets, clinical_targets):
        raise RuntimeError("Targets mismatch after inference -- alignment broken.")
    targets = vision_targets
    LOGGER.info("Targets verified identical across both models.")

    # Save raw probability arrays for downstream sweep scripts
    np.save(repo_root / VISION_PROBS_FILE, vision_probs)
    np.save(repo_root / CLINICAL_PROBS_FILE, clinical_probs)
    np.save(repo_root / TARGETS_FILE, targets)
    LOGGER.info("Saved raw prob arrays: %s, %s, %s", VISION_PROBS_FILE, CLINICAL_PROBS_FILE, TARGETS_FILE)

    # Fused probabilities
    fused_probs = (vision_probs + clinical_probs) / 2.0

    # Compute metrics for all three
    results = {}
    for name, probs in [
        ("vision_only", vision_probs),
        ("clinical_only", clinical_probs),
        ("fused", fused_probs),
    ]:
        aurocs, auprcs, mean_auroc, mean_auprc = compute_label_metrics(targets, probs)
        sens_spec = compute_sensitivity_specificity(targets, probs, threshold=0.5)
        results[name] = {
            "per_label_auroc": aurocs,
            "per_label_auprc": auprcs,
            "mean_auroc": mean_auroc,
            "mean_auprc": mean_auprc,
            "sensitivity_specificity_at_0_5": sens_spec,
        }
        LOGGER.info(
            "%s: mean_auroc=%.4f mean_auprc=%.4f",
            name, mean_auroc, mean_auprc,
        )

    # Print side-by-side comparison
    print("\n" + "=" * 80)
    print("LATE FUSION E07 -- VAL METRICS COMPARISON")
    print("=" * 80)
    print(f"{'Label':<25} {'Vision AUROC':>12} {'Clinical AUROC':>14} {'Fused AUROC':>12} {'Vision AUPRC':>12} {'Clinical AUPRC':>14} {'Fused AUPRC':>12}")
    print("-" * 80)
    for label in LABEL_COLUMNS:
        v_auroc = results["vision_only"]["per_label_auroc"][label]
        c_auroc = results["clinical_only"]["per_label_auroc"][label]
        f_auroc = results["fused"]["per_label_auroc"][label]
        v_auprc = results["vision_only"]["per_label_auprc"][label]
        c_auprc = results["clinical_only"]["per_label_auprc"][label]
        f_auprc = results["fused"]["per_label_auprc"][label]
        print(
            f"{label:<25} "
            f"{v_auroc:>12.4f} " if v_auroc is not None else f"{'N/A':>12} "
            f"{c_auroc:>14.4f} " if c_auroc is not None else f"{'N/A':>14} "
            f"{f_auroc:>12.4f} " if f_auroc is not None else f"{'N/A':>12} "
            f"{v_auprc:>12.4f} " if v_auprc is not None else f"{'N/A':>12} "
            f"{c_auprc:>14.4f} " if c_auprc is not None else f"{'N/A':>14} "
            f"{f_auprc:>12.4f} " if f_auprc is not None else f"{'N/A':>12}"
        )
    print("-" * 80)
    print(
        f"{'MEAN':<25} "
        f"{results['vision_only']['mean_auroc']:>12.4f} "
        f"{results['clinical_only']['mean_auroc']:>14.4f} "
        f"{results['fused']['mean_auroc']:>12.4f} "
        f"{results['vision_only']['mean_auprc']:>12.4f} "
        f"{results['clinical_only']['mean_auprc']:>14.4f} "
        f"{results['fused']['mean_auprc']:>12.4f}"
    )
    print("=" * 80)

    # Save metrics
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    LOGGER.info("Saved late fusion metrics to %s", output_file)

    # Update experiment log with fused metrics (compute diffs for notes)
    fused_auroc = results["fused"]["mean_auroc"]
    fused_auprc = results["fused"]["mean_auprc"]
    vision_auroc = results["vision_only"]["mean_auroc"]
    clinical_auroc = results["clinical_only"]["mean_auroc"]
    diff_vision = fused_auroc - vision_auroc
    diff_clinical = fused_auroc - clinical_auroc

    # Override notes with actual diffs
    try:
        log_path = repo_root / EXPERIMENT_LOG_RELATIVE_PATH
        if log_path.is_file():
            valid_aurocs = {l: v for l, v in results["fused"]["per_label_auroc"].items() if v is not None}
            if valid_aurocs:
                best_label = max(valid_aurocs, key=lambda l: valid_aurocs[l])
                worst_label = min(valid_aurocs, key=lambda l: valid_aurocs[l])
                best_name = best_label.removesuffix("_label")
                worst_name = worst_label.removesuffix("_label")
                notes = (
                    f"Fused AUROC vs vision-only: {diff_vision:+.4f}; "
                    f"vs clinical-only: {diff_clinical:+.4f}. "
                    f"Best: {best_name} ({valid_aurocs[best_label]:.4f}), "
                    f"Worst: {worst_name} ({valid_aurocs[worst_label]:.4f})."
                )
            else:
                notes = f"Fused AUROC vs vision-only: {diff_vision:+.4f}; vs clinical-only: {diff_clinical:+.4f}."

            row = (
                f"| E07 | {date.today().isoformat()} | 7 | "
                f"Late fusion (avg of E04 vision + E06 clinical) | "
                f"simple unweighted averaging of vision and clinical champion model output probabilities, no new training | "
                f"{fused_auroc:.4f} | {fused_auprc:.4f} | {notes} |"
            )
            content = log_path.read_text(encoding="utf-8")
            if content and not content.endswith("\n"):
                content += "\n"
            content += row + "\n"
            log_path.write_text(content, encoding="utf-8")
            LOGGER.info("Appended experiment E07 to %s", log_path)
    except Exception as error:
        LOGGER.warning("Failed to update experiment log: %s", error)


if __name__ == "__main__":
    main()