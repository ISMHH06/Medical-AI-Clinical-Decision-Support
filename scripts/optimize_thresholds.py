"""Phase 7/8: Per-label decision threshold optimization on val sets.

Loads E04 vision-only champion and E08 fusion champion, runs inference on
val split, and sweeps per-label thresholds to maximize Youden's J statistic
(sensitivity + specificity - 1). Saves results and compares optimal thresholds.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.data.image_dataset import ChestXrayDataset, LABEL_COLUMNS
from src.data.fusion_dataset import FusionDataset
from src.models.fusion_model import FusionModel
from src.models.vision_baseline import build_vision_baseline_frozen


LOGGER = logging.getLogger(__name__)

MANIFEST_PATH = "data/processed/image_subset_manifest.parquet"
E04_CHECKPOINT = "models/vision_baseline_champion.pt"
E08_CHECKPOINT = "models/fusion_e08_champion.pt"


def run_e04_vision_only() -> dict[str, np.ndarray]:
    """Run E04 vision-only champion on val split, return probs and targets."""
    model = build_vision_baseline_frozen(unfreeze_from_block="denseblock3")
    ckpt_path = Path(E04_CHECKPOINT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"E04 champion checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(state)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.load_state_dict(state)
    model.to(device)

    val_dataset = ChestXrayDataset(str(MANIFEST_PATH), split="val")
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for images, targets in val_loader:
            images = images.to(device)
            with torch.set_grad_enabled(False):
                logits = model(images)
            probs = (1.0 / (1.0 + torch.exp(-logits))).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    y_prob = np.concatenate(all_probs, axis=0)  # (N, 5)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 5)
    return {"probs": y_prob, "targets": y_true}


def run_e08_fusion() -> dict[str, np.ndarray]:
    """Run E08 fusion champion on val split, return probs and targets."""
    val_embeddings = np.load("models/vision_embeddings_val.npy")

    val_dataset = FusionDataset(
        MANIFEST_PATH, "val", "models/vision_embeddings_val.npy"
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=0)

    model = FusionModel()
    ckpt_path = Path(E08_CHECKPOINT)
    if not ckpt_path.is_file():
        raise FileNotFoundError(f"E08 champion checkpoint not found: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    all_probs: list[np.ndarray] = []
    all_targets: list[np.ndarray] = []

    with torch.no_grad():
        for vision_emb, clinical_feat, targets in val_loader:
            vision_emb = vision_emb.to(device)
            clinical_feat = clinical_feat.to(device)
            with torch.set_grad_enabled(False):
                logits = model(vision_emb, clinical_feat)
            probs = (1.0 / (1.0 + torch.exp(-logits))).cpu().numpy()
            all_probs.append(probs)
            all_targets.append(targets.numpy())

    y_prob = np.concatenate(all_probs, axis=0)  # (N, 5)
    y_true = np.concatenate(all_targets, axis=0)  # (N, 5)
    return {"probs": y_prob, "targets": y_true}


def sweep_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, dict]:
    """Per-label threshold sweep using Youden's J = sensitivity + specificity - 1.

    For each of the 5 labels independently, sweeps thresholds 0.05..0.95
    step 0.05 and picks the threshold maximizing Youden's J.

    Returns dict with keys per label name, each value dict containing:
        optimal_threshold, sensitivity_at_optimal, specificity_at_optimal,
        sensitivity_at_0.5, specificity_at_0.5
    """
    results: dict[str, dict] = {}
    for label_idx, label_name in enumerate(LABEL_COLUMNS):
        true = y_true[:, label_idx]
        probs = y_prob[:, label_idx]

        # Compute sensitivity/specificity at fixed 0.5 threshold once,
        # outside the sweep loop, so they are not overwritten by the
        # best-threshold values.
        pred_05 = (probs >= 0.5).astype(int)
        tp_05 = int(((pred_05 == 1) & (true == 1)).sum())
        tn_05 = int(((pred_05 == 0) & (true == 0)).sum())
        fp_05 = int(((pred_05 == 1) & (true == 0)).sum())
        fn_05 = int(((pred_05 == 0) & (true == 1)).sum())
        sens_05 = float(tp_05 / (tp_05 + fn_05)) if (tp_05 + fn_05) > 0 else 0.0
        spec_05 = float(tn_05 / (tn_05 + fp_05)) if (tn_05 + fp_05) > 0 else 0.0

        best_j = -1.0
        best_threshold = 0.5
        best_sens = 0.0
        best_spec = 0.0

        for threshold in np.arange(0.05, 1.0, 0.05):
            pred = (probs >= threshold).astype(int)
            tp = int(((pred == 1) & (true == 1)).sum())
            tn = int(((pred == 0) & (true == 0)).sum())
            fp = int(((pred == 1) & (true == 0)).sum())
            fn = int(((pred == 0) & (true == 1)).sum())

            sensitivity = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            specificity = float(tn / (tn + fp)) if (tn + fp) > 0 else 0.0
            youden_j = sensitivity + specificity - 1.0

            if youden_j > best_j:
                best_j = youden_j
                best_threshold = float(threshold)
                best_sens = sensitivity
                best_spec = specificity

        results[label_name] = {
            "optimal_threshold": best_threshold,
            "sensitivity_at_optimal": best_sens,
            "specificity_at_optimal": best_spec,
            "sensitivity_at_0.5": sens_05,
            "specificity_at_0.5": spec_05,
        }
    return results


def main() -> None:
    print("=" * 70)
    print("PER-LABEL THRESHOLD OPTIMIZATION: E04 Vision vs E08 Fusion")
    print("=" * 70)

    # Run E04 vision-only
    print("\n[E04] Loading vision-only champion and evaluating on val...")
    e04_data = run_e04_vision_only()
    e04_results = sweep_thresholds(e04_data["targets"], e04_data["probs"])

    # Run E08 fusion
    print("[E08] Loading fusion champion and evaluating on val...")
    e08_data = run_e08_fusion()
    e08_results = sweep_thresholds(e08_data["targets"], e08_data["probs"])

    # Print E04 table
    print("\n" + "-" * 70)
    print("E04 VISION-ONLY: per-label optimal thresholds")
    print("-" * 70)
    header = (
        f"{'Label':20s} {'Opt thresh':>10s} {'Sens @ opt':>11s} "
        f"{'Spec @ opt':>11s} {'Sens @ 0.5':>11s} {'Spec @ 0.5':>11s}"
    )
    print(header)
    for label in LABEL_COLUMNS:
        r = e04_results[label]
        row = (
            f"{label:20s} {r['optimal_threshold']:>10.2f} "
            f"{r['sensitivity_at_optimal']:>11.4f} "
            f"{r['specificity_at_optimal']:>11.4f} "
            f"{r['sensitivity_at_0.5']:>11.4f} "
            f"{r['specificity_at_0.5']:>11.4f}"
        )
        print(row)

    # Print E08 table
    print("\n" + "-" * 70)
    print("E08 FUSION: per-label optimal thresholds")
    print("-" * 70)
    header2 = (
        f"{'Label':20s} {'Opt thresh':>10s} {'Sens @ opt':>11s} "
        f"{'Spec @ opt':>11s} {'Sens @ 0.5':>11s} {'Spec @ 0.5':>11s}"
    )
    print(header2)
    for label in LABEL_COLUMNS:
        r = e08_results[label]
        row = (
            f"{label:20s} {r['optimal_threshold']:>10.2f} "
            f"{r['sensitivity_at_optimal']:>11.4f} "
            f"{r['specificity_at_optimal']:>11.4f} "
            f"{r['sensitivity_at_0.5']:>11.4f} "
            f"{r['specificity_at_0.5']:>11.4f}"
        )
        print(row)

    # Comparison summary, especially Atelectasis and Consolidation
    print("\n" + "=" * 70)
    print("COMPARISON: E04 vs E08 at optimal thresholds (Atelectasis & Consolidation)")
    print("=" * 70)

    for label in ["Atelectasis_label", "Consolidation_label"]:
        e04 = e04_results[label]
        e08 = e08_results[label]
        print(f"\n{label}:")
        print(
            f"  E04:    threshold={e04['optimal_threshold']:.2f}, "
            f"sens={e04['sensitivity_at_optimal']:.4f}, spec={e04['specificity_at_optimal']:.4f}"
        )
        print(
            f"  E08:    threshold={e08['optimal_threshold']:.2f}, "
            f"sens={e08['sensitivity_at_optimal']:.4f}, spec={e08['specificity_at_optimal']:.4f}"
        )
        e04_better_sens = e04["sensitivity_at_optimal"] > e08["sensitivity_at_optimal"]
        e04_better_spec = e04["specificity_at_optimal"] > e08["specificity_at_optimal"]
        if e04_better_sens and e04_better_spec:
            print("  -> E04 better at both sensitivity and specificity at optimal threshold")
        elif e04_better_sens:
            print("  -> E04 better at sensitivity at optimal threshold")
        elif e04_better_spec:
            print("  -> E04 better at specificity at optimal threshold")
        else:
            print("  -> E08 better at both sensitivity and specificity at optimal threshold")

    # Save results
    output_path = Path("models/threshold_optimization_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = {"vision_e04": e04_results, "fusion_e08": e08_results}
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")

    print("\n" + "=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":
    main()