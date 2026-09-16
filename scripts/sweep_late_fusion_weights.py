"""Phase 7 E07b: Weighted late fusion sweep.

Loads precomputed vision_probs.npy, clinical_probs.npy, and val_targets.npy
from the E07 evaluation, then sweeps vision_weight in [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
(where clinical_weight = 1 - vision_weight) and reports mean AUROC/AUPRC.

No new inference is run -- reuses the arrays already saved by evaluate_late_fusion.py.
"""

from __future__ import annotations

import json
import numpy as np
import sys
from pathlib import Path
from datetime import date

# Compute repo root from script location (same pattern as other scripts in repo)
SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.image_dataset import LABEL_COLUMNS
from sklearn.metrics import roc_auc_score, average_precision_score


def main() -> None:
    # Load precomputed arrays from E07 inference
    vision_probs = np.load(REPO_ROOT / "models" / "vision_probs.npy")
    clinical_probs = np.load(REPO_ROOT / "models" / "clinical_probs.npy")
    val_targets = np.load(REPO_ROOT / "models" / "val_targets.npy")

    print(f"Loaded arrays: vision_probs={vision_probs.shape}, clinical_probs={clinical_probs.shape}, val_targets={val_targets.shape}")

    # Sweep configuration
    vision_weights = [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]
    results = []

    print("\n--- Late Fusion Weight Sweep ---")
    for vw in vision_weights:
        cw = 1.0 - vw
        fused_probs = vw * vision_probs + cw * clinical_probs

        # Compute AUROC and AUPRC for this weight
        aurocs = {}
        auprcs = {}
        for idx, label in enumerate(LABEL_COLUMNS):
            try:
                au = float(roc_auc_score(val_targets[:, idx], fused_probs[:, idx]))
                aupro = float(average_precision_score(val_targets[:, idx], fused_probs[:, idx]))
                aurocs[label] = au
                auprcs[label] = aupro
            except Exception:
                aurocs[label] = None
                auprcs[label] = None

        valid_aurocs = [v for v in aurocs.values() if v is not None]
        valid_auprcs = [v for v in auprcs.values() if v is not None]
        mean_auroc = float(np.mean(valid_aurocs)) if valid_aurocs else float("nan")
        mean_auprc = float(np.mean(valid_auprcs)) if valid_auprcs else float("nan")

        results.append(
            {
                "vision_weight": vw,
                "clinical_weight": 1.0 - vw,
                "mean_auroc": mean_auroc,
                "mean_auprc": mean_auprc,
            }
        )
        print(f"vision_weight={vw:.2f} -> mean_auroc={mean_auroc:.4f}, mean_auprc={mean_auprc:.4f}")

    # Identify best weight
    best = max(results, key=lambda r: r["mean_auroc"])
    pure_vision = [r for r in results if r["vision_weight"] == 1.0][0]

    print("\n--- Sweep Summary ---")
    print(f"Best vision_weight: {best['vision_weight']:.2f} (mean AUROC={best['mean_auroc']:.4f})")
    print(f"Pure vision-only (weight=1.0): mean AUROC={pure_vision['mean_auroc']:.4f}")
    diff = best["mean_auroc"] - pure_vision["mean_auroc"]
    if diff > 0.002:
        print(f"Best weight beats pure vision-only by {diff:.4f} (> 0.002 threshold) -- fusion helps!")
    elif diff > 0:
        print(f"Best weight beats pure vision-only by {diff:.4f} (small improvement)")
    else:
        print(f"Best weight does NOT beat pure vision-only (difference: {diff:.4f})")

    # Save sweep results
    sweep_output = REPO_ROOT / "models" / "late_fusion_weight_sweep.json"
    with open(sweep_output, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved sweep results to {sweep_output}")

    # Check if any weight beats pure vision-only by > 0.002
    best_diff = best["mean_auroc"] - pure_vision["mean_auroc"]
    if best_diff > 0.002:
        # Append to experiment log
        clinical_best = max(
            [r for r in results if r["vision_weight"] != 1.0], key=lambda r: r["mean_auroc"]
        )
        clinical_diff = best["mean_auroc"] - clinical_best["mean_auroc"]

        log_path = REPO_ROOT / "docs" / "experiment_log.md"
        if log_path.is_file():
            notes = (
                f"Weighted late fusion sweep over vision_weights [0.6, 0.7, 0.8, 0.9, 0.95, 1.0]; "
                f"best weight={best['vision_weight']:.2f} (mean AUROC={best['mean_auroc']:.4f}); "
                f"margin over vision-only: {best_diff:+.4f}; "
                f"margin over clinical-only: {clinical_diff:+.4f}."
            )
            row = (
                f"| E07b | {date.today().isoformat()} | 7 | "
                f"Weighted late fusion (best weight from sweep) | "
                f"weighted average of vision + clinical probabilities, weight found via sweep over [0.6, 0.7, 0.8, 0.9, 0.95, 1.0] | "
                f"{best['mean_auroc']:.4f} | {best['mean_auprc']:.4f} | {notes} |"
            )
            content = log_path.read_text(encoding="utf-8")
            if content and not content.endswith("\n"):
                content += "\n"
            content += row + "\n"
            log_path.write_text(content, encoding="utf-8")
            print(f"Appended experiment E07b to experiment log.")
    else:
        # No meaningful improvement -- just print conclusion
        print("\n--- Conclusion ---")
        print("No late-fusion weighting improved over vision-only by more than 0.002 AUROC; "
              "no experiment log entry was appended (negative result).")
        print(f"Best weight {best['vision_weight']:.2f} gave mean AUROC={best['mean_auroc']:.4f} vs "
              f"vision-only {pure_vision['mean_auroc']:.4f} (diff={best_diff:+.4f}).")


if __name__ == "__main__":
    main()