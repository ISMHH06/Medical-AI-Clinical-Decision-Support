"""Phase 10 subgroup fairness testing for E04 and E08 on the test split.

This script evaluates whether performance differs meaningfully across
categorical demographic/clinical subgroups in the test set.

Workflow:
1) Discover raw categorical subgroup columns in the processed manifest.
2) Count each value in the test split and drop groups with fewer than 50 rows.
3) For each remaining subgroup/value, compute AUROC and ECE for each label and
   for both E04 and E08.
4) Apply the deployed E08 temperature scaling policy from
   outputs/calibration/deployed_temperatures.json when present.
5) Save subgroup_metrics.csv and fairness_gaps_top15.csv under outputs/fairness.

This is diagnostic-only analysis; it does not retrain or alter the models.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPTS_DIR = Path(__file__).resolve().parent
if str(SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_DIR))

from plot_reliability_diagrams import expected_calibration_error

MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
TEST_PREDICTIONS_PATH = REPO_ROOT / "outputs" / "test_predictions.parquet"
DEPLOYED_TEMPERATURES_PATH = REPO_ROOT / "outputs" / "calibration" / "deployed_temperatures.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "fairness"
SUBGROUP_METRICS_PATH = OUTPUT_DIR / "subgroup_metrics.csv"
FAIRNESS_GAPS_PATH = OUTPUT_DIR / "fairness_gaps_top15.csv"
SPREAD_PATH = OUTPUT_DIR / "subgroup_performance_spread.csv"

LABELS = [
    "Cardiomegaly_label",
    "Edema_label",
    "Atelectasis_label",
    "Pleural Effusion_label",
    "Consolidation_label",
]
MODEL_NAMES = ["e04", "e08"]
SUBGROUP_COLUMNS = ["sex", "race", "ethnicity", "insurance_type"]
MIN_GROUP_SIZE = 50
PROB_EPS = 1e-7


def _load_deployed_temperatures() -> dict[tuple[str, str], float]:
    if not DEPLOYED_TEMPERATURES_PATH.is_file():
        return {}
    with DEPLOYED_TEMPERATURES_PATH.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)

    deployed: dict[tuple[str, str], float] = {}
    for row in rows:
        label = row.get("label")
        model = row.get("model")
        if label is None or model is None:
            continue
        temp = row.get("deployed_temperature", 1.0)
        try:
            deployed[(str(label), str(model))] = float(temp)
        except (TypeError, ValueError):
            deployed[(str(label), str(model))] = 1.0
    return deployed


def _probs_to_logits(probs: np.ndarray) -> np.ndarray:
    p = np.clip(probs.astype(np.float64), PROB_EPS, 1.0 - PROB_EPS)
    return np.log(p / (1.0 - p))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _apply_temperature(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    if temperature is None or np.isnan(temperature):
        return probabilities
    temperature = float(temperature)
    if temperature <= 0:
        raise ValueError(f"Temperature must be > 0; got {temperature}")
    if np.isclose(temperature, 1.0):
        return probabilities
    return _sigmoid(_probs_to_logits(probabilities) / temperature)


def _safe_auroc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    if y_true.size == 0:
        return float("nan")
    if np.unique(y_true).size < 2:
        return float("nan")
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return float("nan")


def _summarize_group_counts(frame: pd.DataFrame) -> list[tuple[str, str, int]]:
    rows: list[tuple[str, str, int]] = []
    for subgroup_col in SUBGROUP_COLUMNS:
        if subgroup_col not in frame.columns:
            continue
        counts = frame[subgroup_col].fillna("Unknown").astype(str).value_counts(dropna=False)
        for group_value, count in counts.items():
            rows.append((subgroup_col, str(group_value), int(count)))
    return rows


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not TEST_PREDICTIONS_PATH.is_file():
        raise FileNotFoundError(f"Test predictions not found: {TEST_PREDICTIONS_PATH}")

    manifest = pd.read_parquet(MANIFEST_PATH, engine='fastparquet')
    test_frame = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    pred_df = pd.read_parquet(TEST_PREDICTIONS_PATH, engine='fastparquet').reset_index(drop=True)

    if len(test_frame) != len(pred_df):
        print(
            f"Warning: manifest test rows ({len(test_frame)}) and prediction rows ({len(pred_df)}) differ. "
            "Assuming same row order for the test split."
        )

    deployed_temps = _load_deployed_temperatures()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Discovered raw categorical subgroup columns:")
    print(f"  {SUBGROUP_COLUMNS}")
    print("\nTest-split subgroup counts (groups with <50 examples will be excluded):")

    excluded_total = 0
    valid_values_by_column: dict[str, list[str]] = {}

    for subgroup_col in SUBGROUP_COLUMNS:
        if subgroup_col not in test_frame.columns:
            print(f"  {subgroup_col}: column not found in manifest; skipping")
            continue

        counts = test_frame[subgroup_col].fillna("Unknown").astype(str).value_counts(dropna=False)
        valid_values = [str(value) for value, count in counts.items() if int(count) >= MIN_GROUP_SIZE]
        valid_values_by_column[subgroup_col] = valid_values
        excluded_total += int((counts < MIN_GROUP_SIZE).sum())

        print(f"\n{subgroup_col}:")
        for value, count in counts.sort_values(ascending=False).items():
            status = "KEEP" if int(count) >= MIN_GROUP_SIZE else "EXCLUDE"
            print(f"  {value}: n={int(count)} [{status}]")

    rows: list[dict[str, object]] = []
    for subgroup_col in SUBGROUP_COLUMNS:
        if subgroup_col not in test_frame.columns:
            continue

        counts = test_frame[subgroup_col].fillna("Unknown").astype(str).value_counts(dropna=False)
        valid_values = [str(value) for value, count in counts.items() if int(count) >= MIN_GROUP_SIZE]
        if not valid_values:
            continue

        for group_value in valid_values:
            mask = test_frame[subgroup_col].fillna("Unknown").astype(str).eq(group_value).to_numpy()
            group_manifest = test_frame.loc[mask].reset_index(drop=True)
            group_predictions = pred_df.loc[mask].reset_index(drop=True)
            n_group = int(len(group_manifest))
            if n_group == 0:
                continue

            for label in LABELS:
                y_true = group_predictions[f"{label}_true"].to_numpy(dtype=np.float64)
                if y_true.size == 0:
                    continue

                for model in MODEL_NAMES:
                    raw_probs = group_predictions[f"{label}_{model}"].to_numpy(dtype=np.float64)
                    temperature = 1.0
                    if model == "e08":
                        temperature = float(deployed_temps.get((label, model), 1.0))
                    probs = _apply_temperature(raw_probs, temperature)

                    auroc = _safe_auroc(y_true, probs)
                    ece, _, _, _, _ = expected_calibration_error(y_true, probs, n_bins=10)

                    rows.append(
                        {
                            "subgroup_column": subgroup_col,
                            "subgroup_value": group_value,
                            "label": label,
                            "model": model,
                            "n": n_group,
                            "positive_rate": float(y_true.mean()),
                            "auroc": auroc,
                            "ece": float(ece),
                            "deployed_temperature": float(temperature),
                        }
                    )

    metrics_df = pd.DataFrame(rows)
    if metrics_df.empty:
        metrics_df = pd.DataFrame(
            columns=[
                "subgroup_column",
                "subgroup_value",
                "label",
                "model",
                "n",
                "positive_rate",
                "auroc",
                "ece",
                "deployed_temperature",
            ]
        )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    metrics_df.to_csv(SUBGROUP_METRICS_PATH, index=False)

    pivot = (
        metrics_df.pivot_table(
            index=["subgroup_column", "subgroup_value", "label"],
            columns="model",
            values="auroc",
            aggfunc="first",
        )
        .reset_index()
    )
    if not pivot.empty:
        pivot["gap_abs"] = (pivot["e08"] - pivot["e04"]).abs()
        gap_df = pivot[["subgroup_column", "subgroup_value", "label", "e04", "e08", "gap_abs"]].sort_values(
            "gap_abs", ascending=False
        )
        gap_df = gap_df.head(15).reset_index(drop=True)
        gap_df.rename(
            columns={
                "subgroup_column": "subgroup",
                "subgroup_value": "value",
                "e04": "e04_auroc",
                "e08": "e08_auroc",
                "gap_abs": "abs_gap",
            },
            inplace=True,
        )
    else:
        gap_df = pd.DataFrame(columns=["subgroup", "value", "label", "e04_auroc", "e08_auroc", "abs_gap"])

    gap_df.to_csv(FAIRNESS_GAPS_PATH, index=False)

    spread_rows: list[dict[str, object]] = []
    if not metrics_df.empty:
        valid_metrics = metrics_df.dropna(subset=["subgroup_column", "subgroup_value", "label", "model", "auroc", "ece"]).copy()
        for (subgroup_column, label, model), group_df in valid_metrics.groupby(["subgroup_column", "label", "model"]):
            subgroup_values = group_df["subgroup_value"].nunique()
            if subgroup_values < 2:
                continue

            auroc_min = float(group_df["auroc"].min())
            auroc_max = float(group_df["auroc"].max())
            auroc_min_row = group_df.loc[group_df["auroc"].idxmin()]
            auroc_max_row = group_df.loc[group_df["auroc"].idxmax()]

            ece_min = float(group_df["ece"].min())
            ece_max = float(group_df["ece"].max())
            ece_min_row = group_df.loc[group_df["ece"].idxmin()]
            ece_max_row = group_df.loc[group_df["ece"].idxmax()]

            spread_rows.append(
                {
                    "subgroup_column": subgroup_column,
                    "label": label,
                    "model": model,
                    "n_groups_compared": int(subgroup_values),
                    "auroc_min": auroc_min,
                    "auroc_min_group": str(auroc_min_row["subgroup_value"]),
                    "auroc_max": auroc_max,
                    "auroc_max_group": str(auroc_max_row["subgroup_value"]),
                    "auroc_spread": float(auroc_max - auroc_min),
                    "ece_min": ece_min,
                    "ece_min_group": str(ece_min_row["subgroup_value"]),
                    "ece_max": ece_max,
                    "ece_max_group": str(ece_max_row["subgroup_value"]),
                    "ece_spread": float(ece_max - ece_min),
                }
            )

    spread_df = pd.DataFrame(spread_rows)
    if spread_df.empty:
        spread_df = pd.DataFrame(
            columns=[
                "subgroup_column",
                "label",
                "model",
                "n_groups_compared",
                "auroc_min",
                "auroc_min_group",
                "auroc_max",
                "auroc_max_group",
                "auroc_spread",
                "ece_min",
                "ece_min_group",
                "ece_max",
                "ece_max_group",
                "ece_spread",
            ]
        )
    spread_df = spread_df.sort_values(["subgroup_column", "label", "model"]).reset_index(drop=True)
    spread_df.to_csv(SPREAD_PATH, index=False)

    print(f"\nExcluded groups with fewer than {MIN_GROUP_SIZE} examples: {excluded_total}")
    print(f"Saved subgroup metrics to: {SUBGROUP_METRICS_PATH}")
    print(f"Saved top fairness gaps to: {FAIRNESS_GAPS_PATH}")

    print("\nTop 15 fairness gaps by absolute AUROC difference (E08 - E04):")
    if gap_df.empty:
        print("  No valid subgroup/label combinations met the minimum group size.")
    else:
        print(gap_df.to_string(index=False))

    print("\nACROSS-SUBGROUP performance spread (top 15 by AUROC spread):")
    if spread_df.empty:
        print("  No valid (subgroup_column, label, model) combinations had at least 2 non-excluded groups.")
    else:
        print(spread_df.sort_values("auroc_spread", ascending=False).head(15).to_string(index=False))

    print("\nACROSS-SUBGROUP performance spread (top 15 by ECE spread):")
    if spread_df.empty:
        print("  No valid (subgroup_column, label, model) combinations had at least 2 non-excluded groups.")
    else:
        print(spread_df.sort_values("ece_spread", ascending=False).head(15).to_string(index=False))

    print(f"\nSaved across-subgroup spread table: {SPREAD_PATH}")


if __name__ == "__main__":
    main()
