"""Phase 10 clinical feature perturbation stress-testing for E08 on the test split.

This script evaluates robustness to seven perturbation conditions plus a clean
baseline. It perturbs only the 29-dim clinical feature vectors, keeps the
precomputed test vision embeddings fixed, and scores the E08 fusion model.

Outputs:
- outputs/robustness/clinical_perturbation_results.csv
- outputs/robustness/clinical_perturbation_curves.png
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.clinical_features import build_clinical_features
from src.data.fusion_dataset import FusionDataset
from src.data.image_dataset import LABEL_COLUMNS
from src.models.fusion_model import FusionModel

MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
TEST_EMBEDDINGS_PATH = REPO_ROOT / "models" / "vision_embeddings_test.npy"
FUSION_CHECKPOINT_PATH = REPO_ROOT / "models" / "fusion_e08_best.pt"
TEMPERATURES_PATH = REPO_ROOT / "outputs" / "calibration" / "deployed_temperatures.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "robustness"
RESULTS_PATH = OUTPUT_DIR / "clinical_perturbation_results.csv"
FIGURE_PATH = OUTPUT_DIR / "clinical_perturbation_curves.png"

CONDITION_ORDER = [
    "clean",
    "missing_10",
    "missing_30",
    "missing_50",
    "noise_025x",
    "noise_05x",
    "noise_1x",
    "noise_2x",
]

CONTINUOUS_FEATURES = ["age", "recent_bmi"]
MISSING_FRACTIONS = {
    "missing_10": 0.10,
    "missing_30": 0.30,
    "missing_50": 0.50,
}
NOISE_SCALES = {
    "noise_025x": 0.25,
    "noise_05x": 0.50,
    "noise_1x": 1.00,
    "noise_2x": 2.00,
}

CHUNK_SIZE = 512
PROB_EPS = 1e-7


def _extract_state_dict(checkpoint: object) -> dict[str, torch.Tensor]:
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if isinstance(state_dict, dict):
            return state_dict
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def read_deployed_temperatures() -> dict[str, float]:
    if not TEMPERATURES_PATH.is_file():
        raise FileNotFoundError(f"Missing deployed temperatures: {TEMPERATURES_PATH}")
    with TEMPERATURES_PATH.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)
    return {
        row["label"]: float(row["deployed_temperature"])
        for row in rows
        if row["model"] == "e08"
    }


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(raw_probs: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(raw_probs.astype(np.float64), PROB_EPS, 1.0 - PROB_EPS)
    logits = np.log(clipped / (1.0 - clipped))
    return sigmoid(logits / temperature)


def load_e08_model(device: torch.device) -> nn.Module:
    if not FUSION_CHECKPOINT_PATH.is_file():
        raise FileNotFoundError(f"Fusion checkpoint not found: {FUSION_CHECKPOINT_PATH}")

    model = FusionModel()
    checkpoint = torch.load(FUSION_CHECKPOINT_PATH, map_location="cpu", weights_only=True)
    model.load_state_dict(_extract_state_dict(checkpoint))
    model.to(device)
    model.eval()
    return model


def build_train_feature_stats(train_frame: pd.DataFrame) -> tuple[list[str], dict[str, float]]:
    train_features_df, feature_columns = build_clinical_features(train_frame, train_frame=train_frame)
    stds = {feature: float(train_features_df[feature].std()) for feature in CONTINUOUS_FEATURES}
    return feature_columns, stds


def perturb_clinical_features(
    base_features: np.ndarray,
    condition: str,
    rng: np.random.Generator,
    continuous_indices: list[int],
    continuous_scales: dict[int, float],
) -> np.ndarray:
    perturbed = base_features.copy()
    n_rows, n_features = perturbed.shape

    if condition in MISSING_FRACTIONS:
        fraction = MISSING_FRACTIONS[condition]
        num_zero = max(1, int(round(fraction * n_features)))
        for row_idx in range(n_rows):
            zero_indices = rng.choice(n_features, size=num_zero, replace=False)
            perturbed[row_idx, zero_indices] = 0.0
        return perturbed

    if condition in NOISE_SCALES:
        scale = NOISE_SCALES[condition]
        for feature_idx in continuous_indices:
            noise_std = continuous_scales[feature_idx] * scale
            perturbed[:, feature_idx] += rng.normal(0.0, noise_std, size=n_rows).astype(np.float32)
        return perturbed

    if condition != "clean":
        raise ValueError(f"Unsupported condition: {condition!r}")

    return perturbed


def run_condition(
    model: nn.Module,
    vision_embeddings: np.ndarray,
    clinical_features: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    n_rows = clinical_features.shape[0]
    probabilities = np.zeros((n_rows, len(LABEL_COLUMNS)), dtype=np.float32)
    vision_tensor = torch.from_numpy(vision_embeddings).to(device)

    with torch.inference_mode():
        for start_idx in range(0, n_rows, CHUNK_SIZE):
            end_idx = min(start_idx + CHUNK_SIZE, n_rows)
            vision_chunk = vision_tensor[start_idx:end_idx]
            clinical_chunk = torch.from_numpy(clinical_features[start_idx:end_idx]).to(device)
            logits = model(vision_chunk, clinical_chunk)
            probabilities[start_idx:end_idx] = torch.sigmoid(logits).detach().cpu().numpy()

    return probabilities


def make_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = results_df.groupby("condition", as_index=False)["auroc"].mean().rename(columns={"auroc": "mean_auroc"})
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITION_ORDER, ordered=True)
    return summary.sort_values("condition").reset_index(drop=True)


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not TEST_EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Test embeddings not found: {TEST_EMBEDDINGS_PATH}")

    rng = np.random.default_rng(42)
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(42)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    frame = pd.read_parquet(MANIFEST_PATH)
    train_frame = frame.loc[frame["split"] == "train"].reset_index(drop=True)
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    if test_frame.empty:
        raise ValueError("The manifest has no rows in the test split.")
    print(f"Test split size: {len(test_frame)}")

    feature_columns, train_stds = build_train_feature_stats(train_frame)
    continuous_indices = [feature_columns.index(feature) for feature in CONTINUOUS_FEATURES]
    continuous_scales = {feature_columns.index(feature): train_stds[feature] for feature in CONTINUOUS_FEATURES}

    test_dataset = FusionDataset(str(MANIFEST_PATH), "test", str(TEST_EMBEDDINGS_PATH))
    base_clinical = test_dataset.features.cpu().numpy().astype(np.float32)
    vision_embeddings = np.asarray(test_dataset.embeddings, dtype=np.float32)
    true_labels = test_dataset.targets.cpu().numpy().astype(np.float32)

    temperatures = read_deployed_temperatures()
    missing_temperatures = [label for label in LABEL_COLUMNS if label not in temperatures]
    if missing_temperatures:
        raise KeyError(f"Missing deployed E08 temperatures for labels: {missing_temperatures}")

    model = load_e08_model(device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    clean_auroc_lookup: dict[str, float] = {}
    start_time = time.time()

    for condition_idx, condition in enumerate(CONDITION_ORDER, start=1):
        print(f"\n=== Condition {condition_idx}/{len(CONDITION_ORDER)}: {condition} ===", flush=True)
        perturbed_clinical = perturb_clinical_features(
            base_clinical,
            condition,
            rng,
            continuous_indices,
            continuous_scales,
        )
        raw_probs = run_condition(model, vision_embeddings, perturbed_clinical, device)

        for label_idx, label in enumerate(LABEL_COLUMNS):
            probs = apply_temperature(raw_probs[:, label_idx], temperatures[label])
            auroc = float(roc_auc_score(true_labels[:, label_idx], probs))
            all_rows.append(
                {
                    "condition": condition,
                    "label": label,
                    "auroc": auroc,
                }
            )
            if condition == "clean":
                clean_auroc_lookup[label] = auroc

        print(f"Finished condition {condition}; computing AUROCs...", flush=True)

    results_df = pd.DataFrame(all_rows)
    results_df["condition"] = pd.Categorical(results_df["condition"], categories=CONDITION_ORDER, ordered=True)
    results_df = results_df.sort_values(["condition", "label"]).reset_index(drop=True)
    results_df["auroc_drop_from_clean"] = results_df.apply(
        lambda row: 0.0 if row["condition"] == "clean" else clean_auroc_lookup[row["label"]] - float(row["auroc"]),
        axis=1,
    )
    results_df = results_df[["condition", "label", "auroc", "auroc_drop_from_clean"]]
    results_df.to_csv(RESULTS_PATH, index=False)

    summary_df = make_summary_table(results_df)
    print("\nCondition summary (mean AUROC across all 5 labels):", flush=True)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    drop_df = (
        results_df.loc[results_df["condition"] != "clean"]
        .sort_values("auroc_drop_from_clean", ascending=False)
        .reset_index(drop=True)
    )
    print("\nAUROC drops from clean (descending):", flush=True)
    print(drop_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    condition_to_index = {condition: idx for idx, condition in enumerate(CONDITION_ORDER)}
    figure, axes = plt.subplots(1, len(LABEL_COLUMNS), figsize=(4.4 * len(LABEL_COLUMNS), 4.8), sharey=True)
    if len(LABEL_COLUMNS) == 1:
        axes = [axes]

    for ax, label in zip(axes, LABEL_COLUMNS):
        label_rows = results_df.loc[results_df["label"] == label].copy()
        label_rows["condition_index"] = label_rows["condition"].map(condition_to_index)
        label_rows = label_rows.sort_values("condition_index")
        ax.plot(
            label_rows["condition_index"].to_numpy(),
            label_rows["auroc"].to_numpy(),
            color="tab:orange",
            marker="o",
            linewidth=2,
        )
        ax.set_title(label.removesuffix("_label"))
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels(CONDITION_ORDER, rotation=30, ha="right")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.35, 1.02)
        ax.grid(alpha=0.25)

    figure.suptitle("E08 robustness to clinical feature perturbations (test split)")
    figure.tight_layout(rect=(0, 0.03, 1, 0.95))
    figure.savefig(FIGURE_PATH, dpi=250, bbox_inches="tight")
    plt.close(figure)

    elapsed = time.time() - start_time
    print(f"\nSaved robustness CSV: {RESULTS_PATH}", flush=True)
    print(f"Saved robustness figure: {FIGURE_PATH}", flush=True)
    print(f"Total runtime: {elapsed:.1f}s", flush=True)


if __name__ == "__main__":
    main()