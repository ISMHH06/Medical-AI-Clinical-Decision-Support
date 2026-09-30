"""Evaluate post-calibration selective prediction for E04 and E08."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
TEMPERATURES_PATH = REPO_ROOT / "outputs" / "calibration" / "deployed_temperatures.json"
PREDICTIONS_PATH = REPO_ROOT / "outputs" / "test_predictions.parquet"
THRESHOLDS_PATH = REPO_ROOT / "models" / "threshold_optimization_results.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "calibration"
FIGURE_PATH = OUTPUT_DIR / "abstention_coverage_curves.png"
RESULTS_PATH = OUTPUT_DIR / "abstention_coverage_results.csv"

LABELS = ["Cardiomegaly_label", "Edema_label", "Atelectasis_label", "Pleural Effusion_label", "Consolidation_label"]
MODELS = {"e04": "vision_e04", "e08": "fusion_e08"}
MODEL_COLORS = {"e04": "#4C78A8", "e08": "#F58518"}
COVERAGES = np.round(np.arange(100, 49, -5) / 100, 2)
EPSILON = 1e-7


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute sigmoid values from logits."""
    return 1.0 / (1.0 + np.exp(-values))


def calibrate_probabilities(raw_probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Recover logits from raw probabilities and apply the deployed temperature."""
    clipped = np.clip(raw_probabilities, EPSILON, 1.0 - EPSILON)
    logits = np.log(clipped / (1.0 - clipped))
    return sigmoid(logits / temperature)


def load_deployed_temperatures() -> dict[tuple[str, str], float]:
    """Load exactly one deployed temperature for each label/model pair."""
    with TEMPERATURES_PATH.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    temperatures = {(row["label"], row["model"]): float(row["deployed_temperature"]) for row in rows}
    required = {(label, model) for label in LABELS for model in MODELS}
    missing = required.difference(temperatures)
    if missing:
        raise ValueError(f"Deployed temperatures are missing pairs: {sorted(missing)}")
    return temperatures


def load_phase7_thresholds() -> dict[tuple[str, str], float]:
    """Load model-specific validation thresholds from the Phase 7 Youden's J sweep."""
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as stream:
        stored = json.load(stream)
    thresholds: dict[tuple[str, str], float] = {}
    for model, result_key in MODELS.items():
        for label in LABELS:
            try:
                thresholds[(label, model)] = float(stored[result_key][label]["optimal_threshold"])
            except KeyError as error:
                raise ValueError(f"Missing Phase 7 threshold for {label}/{model}") from error
    return thresholds


def metrics_at_coverage(y_true: np.ndarray, probabilities: np.ndarray, threshold: float, coverage: float) -> dict[str, float | int]:
    """Measure retained-set classification performance after abstaining by confidence."""
    confidence = np.abs(probabilities - threshold)
    n_retained = max(1, int(np.ceil(coverage * len(y_true))))
    retained = np.argsort(-confidence)[:n_retained]
    true = y_true[retained]
    predicted = (probabilities[retained] >= threshold).astype(int)
    positive = true == 1
    negative = true == 0
    sensitivity = float((predicted[positive] == 1).mean()) if positive.any() else float("nan")
    specificity = float((predicted[negative] == 0).mean()) if negative.any() else float("nan")
    balanced_accuracy = (sensitivity + specificity) / 2 if not np.isnan(sensitivity) and not np.isnan(specificity) else float("nan")
    return {"n_retained": n_retained, "sensitivity": sensitivity, "specificity": specificity, "balanced_accuracy": balanced_accuracy}


def build_results(predictions: pd.DataFrame) -> pd.DataFrame:
    """Calibrate every label/model pair and compute its confidence coverage curve."""
    temperatures = load_deployed_temperatures()
    thresholds = load_phase7_thresholds()
    rows: list[dict[str, object]] = []
    for label in LABELS:
        true_column = f"{label}_true"
        if true_column not in predictions:
            raise ValueError(f"Missing test-target column: {true_column}")
        y_true = predictions[true_column].to_numpy(dtype=int)
        for model in MODELS:
            probability_column = f"{label}_{model}"
            if probability_column not in predictions:
                raise ValueError(f"Missing test-probability column: {probability_column}")
            threshold = thresholds[(label, model)]
            temperature = temperatures[(label, model)]
            calibrated = calibrate_probabilities(predictions[probability_column].to_numpy(dtype=float), temperature)
            print(f"{label} / {model}: threshold={threshold:.2f}, deployed_temperature={temperature:.4f}")
            for coverage in COVERAGES:
                metrics = metrics_at_coverage(y_true, calibrated, threshold, float(coverage))
                rows.append({"label": label, "model": model, "coverage": float(coverage), **metrics})
    return pd.DataFrame(rows)


def save_figure(results: pd.DataFrame) -> None:
    """Save the five-panel sensitivity/specificity versus coverage figure."""
    figure, axes = plt.subplots(2, 3, figsize=(16, 9), sharex=True, sharey=True)
    flat_axes = list(axes.flat)
    for axis, label in zip(flat_axes, LABELS):
        for model in MODELS:
            data = results.loc[(results["label"] == label) & (results["model"] == model)].sort_values("coverage", ascending=False)
            axis.plot(data["coverage"], data["sensitivity"], color=MODEL_COLORS[model], linestyle="-", marker="o", label=f"{model.upper()} sensitivity")
            axis.plot(data["coverage"], data["specificity"], color=MODEL_COLORS[model], linestyle="--", marker="o", label=f"{model.upper()} specificity")
        axis.set_title(label.removesuffix("_label"))
        axis.set(xlabel="Coverage (retained fraction)", ylabel="Metric", xlim=(1.0, 0.5), ylim=(0.0, 1.05))
        axis.grid(alpha=0.3)
    flat_axes[-1].axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.suptitle("Post-Calibration Abstention: Sensitivity and Specificity vs Coverage", y=0.98)
    figure.tight_layout(rect=(0, 0.07, 1, 0.95))
    figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_summary(results: pd.DataFrame) -> None:
    """Print balanced accuracy at full coverage versus 70% coverage."""
    summary = (results.loc[results["coverage"].isin([1.0, 0.7])].pivot(index=["label", "model"], columns="coverage", values="balanced_accuracy").rename(columns={1.0: "balanced_accuracy_coverage_1.0", 0.7: "balanced_accuracy_coverage_0.7"}).reset_index())
    print("\nBALANCED ACCURACY SUMMARY")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    """Run Phase 9 post-calibration abstention analysis on the test predictions."""
    for path in (TEMPERATURES_PATH, PREDICTIONS_PATH, THRESHOLDS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input was not found: {path}")
    predictions = pd.read_parquet(PREDICTIONS_PATH)
    print(f"Loaded {len(predictions):,} test predictions from {PREDICTIONS_PATH}")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    results = build_results(predictions)
    results.to_csv(RESULTS_PATH, index=False)
    save_figure(results)
    print_summary(results)
    print(f"\nSaved abstention coverage figure: {FIGURE_PATH}")
    print(f"Saved abstention coverage data: {RESULTS_PATH}")


if __name__ == "__main__":
    main()
