"""Phase 11 error-case visualization for selected E08 test-set failures.

This script selects a small, deterministic set of clinically meaningful
error cases and renders their Grad-CAM overlays to diagnose what the model
is focusing on when it makes a false negative or false positive decision.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
PREDICTIONS_PATH = REPO_ROOT / "outputs" / "test_predictions.parquet"
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
TEMPERATURES_PATH = REPO_ROOT / "outputs" / "calibration" / "deployed_temperatures.json"
THRESHOLDS_PATH = REPO_ROOT / "models" / "threshold_optimization_results.json"
VISION_EMBEDDINGS_PATH = REPO_ROOT / "models" / "vision_embeddings_test.npy"
OUTPUT_PATH = REPO_ROOT / "outputs" / "error_analysis" / "error_case_gradcam.png"

LABELS = [
    "Cardiomegaly_label",
    "Edema_label",
    "Atelectasis_label",
    "Pleural Effusion_label",
    "Consolidation_label",
]
LABEL_TO_INDEX = {label: idx for idx, label in enumerate(LABELS)}
PROB_EPS = 1e-7

from src.data.fusion_dataset import FusionDataset
from src.explainability.gradcam import E08Explainer, generate_gradcam


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


def build_calibrated_probabilities(predictions: pd.DataFrame, temperatures: dict[str, float]) -> dict[str, np.ndarray]:
    calibrated: dict[str, np.ndarray] = {}
    for label in LABELS:
        raw_probs = predictions[f"{label}_e08"].to_numpy(dtype=np.float64)
        calibrated[label] = calibrate_probabilities(raw_probs, temperatures[label])
    return calibrated


def select_category_cases(
    predictions: pd.DataFrame,
    calibrated_probs: dict[str, np.ndarray],
    thresholds: dict[str, float],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []

    def add_case(category: str, label: str, row_index: int, has_other_tp: bool | None = None) -> None:
        row = predictions.iloc[row_index]
        selected.append(
            {
                "row_index": int(row_index),
                "category": category,
                "label": label,
                "calibrated_probability": float(calibrated_probs[label][row_index]),
                "threshold": float(thresholds[label]),
                "ground_truth": int(row[f"{label}_true"]),
                "has_other_true_positive": has_other_tp,
            }
        )

    atelectasis_true = predictions["Atelectasis_label_true"].to_numpy(dtype=int)
    atelectasis_prob = calibrated_probs["Atelectasis_label"]
    atelectasis_pred = (atelectasis_prob >= thresholds["Atelectasis_label"]).astype(int)
    fn_mask = (atelectasis_true == 1) & (atelectasis_pred == 0) & (thresholds["Atelectasis_label"] - atelectasis_prob <= 0.10)
    fp_mask = (atelectasis_true == 0) & (atelectasis_pred == 1) & (atelectasis_prob - thresholds["Atelectasis_label"] <= 0.10)

    fn_indices = np.where(fn_mask)[0].tolist()
    fp_indices = np.where(fp_mask)[0].tolist()
    other_atelectasis_true_cols = [f"{label}_true" for label in LABELS if label != "Atelectasis_label"]
    for idx in fn_indices[:2]:
        row_has_other_tp = bool((predictions.iloc[idx][other_atelectasis_true_cols].to_numpy(dtype=int) > 0).any())
        add_case("Atelectasis near-miss FN", "Atelectasis_label", int(idx), row_has_other_tp)
    for idx in fp_indices[:2]:
        row_has_other_tp = bool((predictions.iloc[idx][other_atelectasis_true_cols].to_numpy(dtype=int) > 0).any())
        add_case("Atelectasis near-miss FP", "Atelectasis_label", int(idx), row_has_other_tp)

    edema_label = "Edema_label"
    edema_true = predictions[f"{edema_label}_true"].to_numpy(dtype=int)
    edema_prob = calibrated_probs[edema_label]
    edema_pred = (edema_prob >= thresholds[edema_label]).astype(int)
    fn_confident = (edema_true == 1) & (edema_pred == 0) & (thresholds[edema_label] - edema_prob > 0.25)
    fp_confident = (edema_true == 0) & (edema_pred == 1) & (edema_prob - thresholds[edema_label] > 0.25)

    fn_indices = np.where(fn_confident)[0].tolist()
    fp_indices = np.where(fp_confident)[0].tolist()
    if fn_indices and fp_indices:
        ordered = [*fn_indices[:1], *fp_indices[:1]]
    else:
        ordered = sorted(set(fn_indices) | set(fp_indices))[:2]
    for idx in ordered:
        add_case("Edema confidently wrong", edema_label, int(idx), None)

    pleural_label = "Pleural Effusion_label"
    pleural_true = predictions[f"{pleural_label}_true"].to_numpy(dtype=int)
    pleural_prob = calibrated_probs[pleural_label]
    pleural_pred = (pleural_prob >= thresholds[pleural_label]).astype(int)
    other_pleural_true_cols = [f"{label}_true" for label in LABELS if label != pleural_label]
    has_other_true_positive = (
        predictions[other_pleural_true_cols].to_numpy(dtype=int).sum(axis=1) > 0
    )
    pleural_fp_mask = (pleural_true == 0) & (pleural_pred == 1) & (~has_other_true_positive)
    pleural_indices = np.where(pleural_fp_mask)[0].tolist()
    for idx in pleural_indices[:2]:
        row_has_other_tp = bool((predictions.iloc[idx][other_pleural_true_cols].to_numpy(dtype=int) > 0).any())
        add_case("Pleural Effusion isolated FP", pleural_label, int(idx), row_has_other_tp)

    # Keep all category-specific selections. A patient may legitimately appear in
    # multiple requested pools (for example, an Atelectasis false positive can also
    # be a Pleural Effusion isolated false positive), so deduplicating by row index
    # would discard valid cases from the target categories.
    return selected


def select_cases(predictions: pd.DataFrame, calibrated_probs: dict[str, np.ndarray], thresholds: dict[str, float]) -> list[dict[str, object]]:
    cases = select_category_cases(predictions, calibrated_probs, thresholds)
    if len(cases) < 8:
        raise RuntimeError(f"Expected at least 8 selected cases, got {len(cases)}.")
    return cases[:8]


def load_case_image_data(row_index: int) -> tuple[str, torch.Tensor]:
    manifest = pd.read_parquet(MANIFEST_PATH, engine='fastparquet')
    test_frame = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    if row_index >= len(test_frame):
        raise IndexError(f"row_index={row_index} is outside the test split size {len(test_frame)}")
    row = test_frame.iloc[row_index]
    image_path = str(row["local_image_path"])
    dataset = FusionDataset(str(MANIFEST_PATH), "test", str(VISION_EMBEDDINGS_PATH))
    _, clinical_feat, _ = dataset[row_index]
    return image_path, clinical_feat


def render_case_panel(ax, overlay_image: np.ndarray, title: str) -> None:
    ax.imshow(overlay_image)
    ax.set_title(title, fontsize=10, pad=8)
    ax.axis("off")


def main() -> None:
    for path in (PREDICTIONS_PATH, MANIFEST_PATH, TEMPERATURES_PATH, THRESHOLDS_PATH, VISION_EMBEDDINGS_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Required input was not found: {path}")

    predictions = pd.read_parquet(PREDICTIONS_PATH, engine='fastparquet').reset_index(drop=True)
    temperatures = load_deployed_temperatures()
    thresholds = load_thresholds()
    calibrated_probs = build_calibrated_probabilities(predictions, temperatures)
    selected_cases = select_cases(predictions, calibrated_probs, thresholds)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    explainer = E08Explainer().to(device)
    explainer.eval()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    axes = axes.flatten()

    print("SELECTED ERROR CASES")
    for case_idx, case in enumerate(selected_cases, start=1):
        row_index = int(case["row_index"])
        label = str(case["label"])
        category = str(case["category"])
        image_path, clinical_feat = load_case_image_data(row_index)
        label_idx = LABEL_TO_INDEX[label]
        _, overlay_image = generate_gradcam(
            explainer=explainer,
            image_path=image_path,
            clinical_feat=clinical_feat,
            label_idx=label_idx,
            device=device,
        )

        title = (
            f"{category} | p={case['calibrated_probability']:.2f} "
            f"(thresh={case['threshold']:.2f}) | GT={int(case['ground_truth'])}"
        )
        render_case_panel(axes[case_idx - 1], overlay_image, title)

        has_other_tp = case["has_other_true_positive"]
        if case["category"] in {"Edema confidently wrong"}:
            has_other_tp = "NA"
        print(
            f"case={case_idx} | row_index={row_index} | category={category} | label={label} | "
            f"calibrated_probability={float(case['calibrated_probability']):.4f} | "
            f"threshold={float(case['threshold']):.4f} | ground_truth={int(case['ground_truth'])} | "
            f"has_other_true_positive={has_other_tp}"
        )

    fig.tight_layout()
    fig.savefig(OUTPUT_PATH, dpi=200, bbox_inches="tight")
    print(f"\nSaved Grad-CAM figure: {OUTPUT_PATH}")

    plt.close(fig)


if __name__ == "__main__":
    main()
