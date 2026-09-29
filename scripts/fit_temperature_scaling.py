"""Fit per-label, per-model temperature scaling on val and evaluate on test.

This script reads precomputed probabilities from:
- outputs/val_predictions.parquet
- outputs/test_predictions.parquet

For each (label, model) combination, it:
1) Recovers logits from probabilities on val.
2) Fits scalar temperature T > 0 minimizing val BCE.
3) Applies T to held-out test logits.
4) Reports before/after BCE, equal-width ECE, and quantile ECE.

Outputs:
- outputs/calibration/temperatures.json
- outputs/calibration/temperature_scaling_results.csv
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize_scalar

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plot_reliability_diagrams import compute_calibration_curve, compute_calibration_curve_quantile

VAL_PREDICTIONS_PATH = REPO_ROOT / "outputs" / "val_predictions.parquet"
TEST_PREDICTIONS_PATH = REPO_ROOT / "outputs" / "test_predictions.parquet"
OUTPUT_DIR = REPO_ROOT / "outputs" / "calibration"
TEMPERATURES_PATH = OUTPUT_DIR / "temperatures.json"
RESULTS_PATH = OUTPUT_DIR / "temperature_scaling_results.csv"

LABELS = [
    "Cardiomegaly_label",
    "Edema_label",
    "Atelectasis_label",
    "Pleural Effusion_label",
    "Consolidation_label",
]
MODELS = ["e04", "e08"]

PROB_EPS = 1e-7
T_MIN = 0.05
T_MAX = 10.0


def _clip_probs(probs: np.ndarray) -> np.ndarray:
    return np.clip(probs, PROB_EPS, 1.0 - PROB_EPS)


def probs_to_logits(probs: np.ndarray) -> np.ndarray:
    p = _clip_probs(probs)
    return np.log(p / (1.0 - p))


def sigmoid(x: np.ndarray) -> np.ndarray:
    # Numerically stable enough for logits in this workflow.
    return 1.0 / (1.0 + np.exp(-x))


def binary_cross_entropy(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    p = _clip_probs(y_prob)
    y = y_true.astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def expected_calibration_error_from_curve(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    *,
    use_quantile: bool,
) -> float:
    if use_quantile:
        _, mean_pred, frac_pos, counts = compute_calibration_curve_quantile(y_true, y_prob, n_bins=8)
    else:
        _, mean_pred, frac_pos, counts = compute_calibration_curve(y_true, y_prob, n_bins=10)

    if counts.size == 0:
        return 0.0

    gaps = np.abs(mean_pred - frac_pos)
    n_total = counts.sum()
    return float(np.sum((counts / n_total) * gaps))


def fit_temperature_for_val(y_true: np.ndarray, val_probs: np.ndarray) -> tuple[float, float, float]:
    val_logits = probs_to_logits(val_probs)

    def objective(temp: float) -> float:
        scaled_probs = sigmoid(val_logits / temp)
        return binary_cross_entropy(y_true, scaled_probs)

    result = minimize_scalar(objective, bounds=(T_MIN, T_MAX), method="bounded")
    fitted_t = float(result.x) if result.success else 1.0

    bce_before = objective(1.0)
    bce_after = objective(fitted_t)
    return fitted_t, bce_before, bce_after


def main() -> None:
    if not VAL_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(f"Missing val predictions: {VAL_PREDICTIONS_PATH}")
    if not TEST_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(f"Missing test predictions: {TEST_PREDICTIONS_PATH}")

    val_df = pd.read_parquet(VAL_PREDICTIONS_PATH)
    test_df = pd.read_parquet(TEST_PREDICTIONS_PATH)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fitted_rows: list[dict[str, float | str]] = []

    for label in LABELS:
        y_val = val_df[f"{label}_true"].to_numpy(dtype=np.float64)
        for model in MODELS:
            val_probs = val_df[f"{label}_{model}"].to_numpy(dtype=np.float64)
            temperature, bce_before, bce_after = fit_temperature_for_val(y_val, val_probs)
            fitted_rows.append(
                {
                    "label": label,
                    "model": model,
                    "temperature": temperature,
                    "val_bce_before": bce_before,
                    "val_bce_after": bce_after,
                }
            )

    with TEMPERATURES_PATH.open("w", encoding="utf-8") as fp:
        json.dump(fitted_rows, fp, indent=2)

    fitted_table = pd.DataFrame(fitted_rows)
    print("Fitted temperatures (val):")
    print(fitted_table.to_string(index=False))

    fitted_lookup = {
        (row["label"], row["model"]): float(row["temperature"])
        for row in fitted_rows
    }

    result_rows: list[dict[str, float | str]] = []

    for label in LABELS:
        y_test = test_df[f"{label}_true"].to_numpy(dtype=np.float64)
        for model in MODELS:
            temperature = fitted_lookup[(label, model)]
            raw_probs = test_df[f"{label}_{model}"].to_numpy(dtype=np.float64)
            logits = probs_to_logits(raw_probs)
            scaled_probs = sigmoid(logits / temperature)

            bce_before = binary_cross_entropy(y_test, raw_probs)
            bce_after = binary_cross_entropy(y_test, scaled_probs)

            ece_width_before = expected_calibration_error_from_curve(y_test, raw_probs, use_quantile=False)
            ece_width_after = expected_calibration_error_from_curve(y_test, scaled_probs, use_quantile=False)

            ece_quantile_before = expected_calibration_error_from_curve(y_test, raw_probs, use_quantile=True)
            ece_quantile_after = expected_calibration_error_from_curve(y_test, scaled_probs, use_quantile=True)

            result_rows.append(
                {
                    "label": label,
                    "model": model,
                    "temperature": temperature,
                    "bce_before": bce_before,
                    "bce_after": bce_after,
                    "ece_width_before": ece_width_before,
                    "ece_width_after": ece_width_after,
                    "ece_quantile_before": ece_quantile_before,
                    "ece_quantile_after": ece_quantile_after,
                }
            )

    results_table = pd.DataFrame(result_rows)
    results_table.to_csv(RESULTS_PATH, index=False)

    print("\nTemperature scaling results on held-out test:")
    print(results_table.to_string(index=False))

    improved_width = int((results_table["ece_width_after"] < results_table["ece_width_before"]).sum())
    improved_quantile = int(
        (results_table["ece_quantile_after"] < results_table["ece_quantile_before"]).sum()
    )

    print("\nImprovement summary:")
    print(f"Equal-width ECE improved in {improved_width}/10 label-model combinations.")
    print(f"Quantile ECE improved in {improved_quantile}/10 label-model combinations.")

    print(f"\nSaved temperatures: {TEMPERATURES_PATH}")
    print(f"Saved results: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
