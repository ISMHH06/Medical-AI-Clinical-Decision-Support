"""Phase 11 error-structure analysis for E08 on the test split.

This script operates only on saved prediction artifacts and threshold files.
It reports confusion statistics, error co-occurrence structure, and error
confidence structure for each label.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
PREDICTIONS_PATH = REPO_ROOT / "outputs" / "test_predictions.parquet"
TEMPERATURES_PATH = REPO_ROOT / "outputs" / "calibration" / "deployed_temperatures.json"
THRESHOLDS_PATH = REPO_ROOT / "models" / "threshold_optimization_results.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "error_analysis"
CONFUSION_PATH = OUTPUT_DIR / "confusion_matrix.csv"
CO_OCCURRENCE_PATH = OUTPUT_DIR / "error_cooccurrence.csv"
CONFIDENCE_PATH = OUTPUT_DIR / "error_confidence.csv"

LABELS = [
    "Cardiomegaly_label",
    "Edema_label",
    "Atelectasis_label",
    "Pleural Effusion_label",
    "Consolidation_label",
]

PROB_EPS = 1e-7
NEAR_MISS_DISTANCE = 0.10
CONFIDENT_DISTANCE = 0.25


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def calibrate_probabilities(raw_probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(raw_probabilities.astype(np.float64), PROB_EPS, 1.0 - PROB_EPS)
    logits = np.log(clipped / (1.0 - clipped))
    return sigmoid(logits / temperature)


def load_deployed_temperatures() -> dict[str, float]:
    if not TEMPERATURES_PATH.is_file():
        raise FileNotFoundError(f"Missing deployed temperatures: {TEMPERATURES_PATH}")
    with TEMPERATURES_PATH.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)
    temperatures = {row["label"]: float(row["deployed_temperature"]) for row in rows if row["model"] == "e08"}
    missing = [label for label in LABELS if label not in temperatures]
    if missing:
        raise KeyError(f"Missing E08 deployed temperatures for labels: {missing}")
    return temperatures


def load_thresholds() -> dict[str, float]:
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(f"Missing threshold optimization results: {THRESHOLDS_PATH}")
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as fp:
        stored = json.load(fp)
    try:
        fusion_thresholds = stored["fusion_e08"]
    except KeyError as error:
        raise KeyError("Missing 'fusion_e08' thresholds in threshold_optimization_results.json") from error

    thresholds: dict[str, float] = {}
    for label in LABELS:
        try:
            thresholds[label] = float(fusion_thresholds[label]["optimal_threshold"])
        except KeyError as error:
            raise KeyError(f"Missing E08 threshold for {label}") from error
    return thresholds


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def build_confusion_table(frame: pd.DataFrame, calibrated_probs: dict[str, np.ndarray], thresholds: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in LABELS:
        y_true = frame[f"{label}_true"].to_numpy(dtype=int)
        y_prob = calibrated_probs[label]
        y_pred = (y_prob >= thresholds[label]).astype(int)

        tp = int(np.sum((y_true == 1) & (y_pred == 1)))
        tn = int(np.sum((y_true == 0) & (y_pred == 0)))
        fp = int(np.sum((y_true == 0) & (y_pred == 1)))
        fn = int(np.sum((y_true == 1) & (y_pred == 0)))

        rows.append(
            {
                "label": label,
                "TP": tp,
                "TN": tn,
                "FP": fp,
                "FN": fn,
                "FPR": _rate(fp, fp + tn),
                "FNR": _rate(fn, fn + tp),
            }
        )

    return pd.DataFrame(rows)


def build_cooccurrence_table(frame: pd.DataFrame, calibrated_probs: dict[str, np.ndarray], thresholds: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    true_matrix = frame[[f"{label}_true" for label in LABELS]].to_numpy(dtype=int)

    baseline_rates: dict[str, float] = {}
    for label_idx, label in enumerate(LABELS):
        other_true = (true_matrix[:, np.arange(len(LABELS)) != label_idx].sum(axis=1) > 0)
        baseline_rates[label] = _rate(int(other_true.sum()), len(true_matrix))

    for label_idx, label in enumerate(LABELS):
        y_true = true_matrix[:, label_idx]
        y_prob = calibrated_probs[label]
        y_pred = (y_prob >= thresholds[label]).astype(int)

        for error_type, mask in {
            "FN": (y_true == 1) & (y_pred == 0),
            "FP": (y_true == 0) & (y_pred == 1),
        }.items():
            error_indices = np.where(mask)[0]
            n_errors = int(error_indices.size)
            if n_errors == 0:
                pct_with_other = float("nan")
                pct_isolated = float("nan")
            else:
                other_true = true_matrix[error_indices][:, np.arange(len(LABELS)) != label_idx].sum(axis=1) > 0
                pct_with_other = float(other_true.mean())
                pct_isolated = float((~other_true).mean())

            rows.append(
                {
                    "label": label,
                    "error_type": error_type,
                    "n_errors": n_errors,
                    "pct_with_other_true_label": pct_with_other,
                    "pct_isolated": pct_isolated,
                    "baseline_pct_with_other_true_label": baseline_rates[label],
                }
            )

    return pd.DataFrame(rows)


def build_confidence_table(frame: pd.DataFrame, calibrated_probs: dict[str, np.ndarray], thresholds: dict[str, float]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for label in LABELS:
        y_true = frame[f"{label}_true"].to_numpy(dtype=int)
        y_prob = calibrated_probs[label]
        y_pred = (y_prob >= thresholds[label]).astype(int)
        threshold = thresholds[label]

        for error_type, mask in {
            "FN": (y_true == 1) & (y_pred == 0),
            "FP": (y_true == 0) & (y_pred == 1),
        }.items():
            if error_type == "FN":
                distances = threshold - y_prob[mask]
                near_miss = distances <= NEAR_MISS_DISTANCE
                confident = distances > CONFIDENT_DISTANCE
            else:
                distances = y_prob[mask] - threshold
                near_miss = distances <= NEAR_MISS_DISTANCE
                confident = distances > CONFIDENT_DISTANCE

            n_errors = int(distances.size)
            rows.append(
                {
                    "label": label,
                    "error_type": error_type,
                    "n_errors": n_errors,
                    "mean_distance_from_threshold": float(np.mean(distances)) if n_errors else float("nan"),
                    "median_distance_from_threshold": float(np.median(distances)) if n_errors else float("nan"),
                    "pct_near_miss": float(np.mean(near_miss)) if n_errors else float("nan"),
                    "pct_confidently_wrong": float(np.mean(confident)) if n_errors else float("nan"),
                }
            )

    return pd.DataFrame(rows)


def print_table(title: str, table: pd.DataFrame) -> None:
    print(f"\n{title}")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    for path in (PREDICTIONS_PATH, TEMPERATURES_PATH, THRESHOLDS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input was not found: {path}")

    predictions = pd.read_parquet(PREDICTIONS_PATH)
    temperatures = load_deployed_temperatures()
    thresholds = load_thresholds()

    calibrated_probs: dict[str, np.ndarray] = {}
    for label in LABELS:
        raw_probs = predictions[f"{label}_e08"].to_numpy(dtype=np.float64)
        calibrated_probs[label] = calibrate_probabilities(raw_probs, temperatures[label])

    confusion_df = build_confusion_table(predictions, calibrated_probs, thresholds)
    cooccurrence_df = build_cooccurrence_table(predictions, calibrated_probs, thresholds)
    confidence_df = build_confidence_table(predictions, calibrated_probs, thresholds)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    confusion_df.to_csv(CONFUSION_PATH, index=False)
    cooccurrence_df.to_csv(CO_OCCURRENCE_PATH, index=False)
    confidence_df.to_csv(CONFIDENCE_PATH, index=False)

    print_table("CONFUSION MATRIX", confusion_df)
    print_table("ERROR CO-OCCURRENCE", cooccurrence_df)
    print_table("ERROR CONFIDENCE", confidence_df)

    highest_fnr = confusion_df.loc[confusion_df["FNR"].idxmax()]
    highest_fpr = confusion_df.loc[confusion_df["FPR"].idxmax()]
    combined_confidence = (
        confidence_df.groupby("label", as_index=False)[["n_errors", "pct_confidently_wrong"]]
        .apply(lambda group: pd.Series({
            "combined_pct_confidently_wrong": float(np.average(group["pct_confidently_wrong"], weights=group["n_errors"])) if group["n_errors"].sum() else float("nan")
        }))
        .reset_index()
    )
    combined_confidence = confidence_df.groupby("label", as_index=False).apply(
        lambda group: pd.Series(
            {
                "combined_pct_confidently_wrong": float(
                    np.average(group["pct_confidently_wrong"], weights=group["n_errors"])
                )
                if group["n_errors"].sum()
                else float("nan")
            }
        )
    ).reset_index(drop=True)
    highest_confident = combined_confidence.loc[combined_confidence["combined_pct_confidently_wrong"].idxmax()]

    print("\nCLOSING SUMMARY")
    print(f"Highest FNR: {highest_fnr['label']} ({highest_fnr['FNR']:.4f})")
    print(f"Highest FPR: {highest_fpr['label']} ({highest_fpr['FPR']:.4f})")
    print(
        f"Highest pct_confidently_wrong (FN+FP combined): {highest_confident['label']} "
        f"({highest_confident['combined_pct_confidently_wrong']:.4f})"
    )
    print(f"\nSaved confusion matrix: {CONFUSION_PATH}")
    print(f"Saved error co-occurrence: {CO_OCCURRENCE_PATH}")
    print(f"Saved error confidence: {CONFIDENCE_PATH}")


if __name__ == "__main__":
    main()