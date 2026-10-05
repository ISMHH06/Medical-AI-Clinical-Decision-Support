"""Phase 10 image perturbation stress-testing for E04 and E08 on the test split.

This script evaluates robustness to eight image perturbation conditions plus a
clean baseline. It applies each condition to the raw test images before the
normalization/resizing pipeline and scores the same E04/E08 models used in the
project.

Outputs:
- outputs/robustness/image_perturbation_results.csv
- outputs/robustness/image_perturbation_curves.png
"""

from __future__ import annotations

import io
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
from PIL import Image
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fusion_dataset import FusionDataset
from src.data.image_dataset import LABEL_COLUMNS, build_transform
from src.explainability.gradcam import E04Explainer, E08Explainer

MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
TEST_EMBEDDINGS_PATH = REPO_ROOT / "models" / "vision_embeddings_test.npy"
TEMPERATURES_PATH = REPO_ROOT / "outputs" / "calibration" / "deployed_temperatures.json"
OUTPUT_DIR = REPO_ROOT / "outputs" / "robustness"
RESULTS_PATH = OUTPUT_DIR / "image_perturbation_results.csv"
FIGURE_PATH = OUTPUT_DIR / "image_perturbation_curves.png"

CONDITION_ORDER = [
    "clean",
    "brightness_dark",
    "brightness_bright",
    "noise_low",
    "noise_mid",
    "noise_high",
    "jpeg_q75",
    "jpeg_q50",
    "jpeg_q25",
]
MODEL_ORDER = ["e04", "e08"]
PROB_EPS = 1e-7
BATCH_SIZE = 16
NUM_WORKERS = 0


def read_deployed_temperatures() -> dict[tuple[str, str], float]:
    if not TEMPERATURES_PATH.is_file():
        raise FileNotFoundError(f"Missing deployed temperatures: {TEMPERATURES_PATH}")
    with TEMPERATURES_PATH.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)
    return {(row["label"], row["model"]): float(row["deployed_temperature"]) for row in rows}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def apply_temperature(raw_probs: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(raw_probs.astype(np.float64), PROB_EPS, 1.0 - PROB_EPS)
    logits = np.log(clipped / (1.0 - clipped))
    return sigmoid(logits / temperature)


def apply_perturbation(pil_image: Image.Image, perturbation_name: str) -> Image.Image:
    """Apply one named perturbation to an RGB PIL image in pixel space."""
    image = pil_image.convert("RGB")
    if perturbation_name == "clean":
        return image

    arr = np.asarray(image, dtype=np.float32) / 255.0

    if perturbation_name == "brightness_dark":
        arr = np.clip(arr * 0.7, 0.0, 1.0)
    elif perturbation_name == "brightness_bright":
        arr = np.clip(arr * 1.3, 0.0, 1.0)
    elif perturbation_name == "noise_low":
        arr = np.clip(arr + np.random.normal(0.0, 0.02, size=arr.shape), 0.0, 1.0)
    elif perturbation_name == "noise_mid":
        arr = np.clip(arr + np.random.normal(0.0, 0.05, size=arr.shape), 0.0, 1.0)
    elif perturbation_name == "noise_high":
        arr = np.clip(arr + np.random.normal(0.0, 0.10, size=arr.shape), 0.0, 1.0)
    elif perturbation_name == "jpeg_q75":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=75)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    elif perturbation_name == "jpeg_q50":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=50)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    elif perturbation_name == "jpeg_q25":
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=25)
        buffer.seek(0)
        return Image.open(buffer).convert("RGB")
    else:
        raise ValueError(f"Unsupported perturbation: {perturbation_name!r}")

    arr = np.clip(arr, 0.0, 1.0)
    return Image.fromarray(np.round(arr * 255.0).astype(np.uint8), mode="RGB")


class _PerturbedImageDataset(Dataset):
    """Load one test image and apply a requested perturbation before val transform."""

    def __init__(self, frame: pd.DataFrame, perturbation_name: str) -> None:
        self.frame = frame.reset_index(drop=True)
        self.perturbation_name = perturbation_name
        self.transform = build_transform("val")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, idx: int) -> tuple[int, torch.Tensor]:
        row = self.frame.iloc[idx]
        with Image.open(row["local_image_path"]) as image:
            image = image.convert("RGB")
            perturbed = apply_perturbation(image, self.perturbation_name)
            tensor = self.transform(perturbed)
        return idx, tensor


def _make_summary_table(results_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        results_df.groupby(["condition", "model"], as_index=False)["auroc"].mean().rename(columns={"auroc": "mean_auroc"})
    )
    summary["condition"] = pd.Categorical(summary["condition"], categories=CONDITION_ORDER, ordered=True)
    summary = summary.sort_values(["condition", "model"]).reset_index(drop=True)
    return summary


def main() -> None:
    if not MANIFEST_PATH.is_file():
        raise FileNotFoundError(f"Manifest not found: {MANIFEST_PATH}")
    if not TEST_EMBEDDINGS_PATH.is_file():
        raise FileNotFoundError(f"Test embeddings not found: {TEST_EMBEDDINGS_PATH}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    frame = pd.read_parquet(MANIFEST_PATH, engine='fastparquet')
    test_frame = frame.loc[frame["split"] == "test"].reset_index(drop=True)
    n_rows = len(test_frame)
    print(f"Test split size: {n_rows}")

    fusion_dataset = FusionDataset(str(MANIFEST_PATH), "test", str(TEST_EMBEDDINGS_PATH))
    clinical_feats = torch.stack([fusion_dataset[i][1] for i in range(n_rows)])

    temperatures = read_deployed_temperatures()
    true_labels = test_frame[LABEL_COLUMNS].to_numpy(dtype=np.float64)
    e04 = E04Explainer().to(device)
    e08 = E08Explainer().to(device)
    e04.eval()
    e08.eval()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, object]] = []
    clean_auroc_lookup: dict[tuple[str, str], float] = {}
    total_pairs = len(CONDITION_ORDER) * n_rows
    processed_pairs = 0
    start_time = time.time()

    for condition_idx, condition in enumerate(CONDITION_ORDER, start=1):
        print(f"\n=== Condition {condition_idx}/{len(CONDITION_ORDER)}: {condition} ===", flush=True)
        dataset = _PerturbedImageDataset(test_frame, condition)
        loader = DataLoader(
            dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=NUM_WORKERS,
            pin_memory=(device.type == "cuda"),
        )

        e04_probs = np.zeros((n_rows, len(LABEL_COLUMNS)), dtype=np.float32)
        e08_raw_probs = np.zeros((n_rows, len(LABEL_COLUMNS)), dtype=np.float32)

        with torch.no_grad():
            for indices, images in loader:
                images = images.to(device, non_blocking=True)
                batch_clinical = clinical_feats[indices].to(device, non_blocking=True)

                e04_logits = e04(images)
                e04_batch_probs = torch.sigmoid(e04_logits).cpu().numpy()
                e08_logits = e08(images, batch_clinical)
                e08_batch_probs = torch.sigmoid(e08_logits).cpu().numpy()

                idx_np = np.asarray(indices, dtype=np.int64)
                e04_probs[idx_np] = e04_batch_probs
                e08_raw_probs[idx_np] = e08_batch_probs

                processed_pairs += len(idx_np)
                if processed_pairs % 500 == 0 or processed_pairs == total_pairs:
                    elapsed = time.time() - start_time
                    print(f"  Processed {processed_pairs}/{total_pairs} image-condition pairs ({elapsed:.1f}s elapsed)", flush=True)

        print(f"Finished condition {condition}; computing AUROCs...", flush=True)

        for model_name, probs in [("e04", e04_probs), ("e08", e08_raw_probs)]:
            for label_idx, label in enumerate(LABEL_COLUMNS):
                if model_name == "e08":
                    temperature = temperatures.get((label, "e08"), 1.0)
                    probs_for_label = apply_temperature(probs[:, label_idx], temperature)
                else:
                    probs_for_label = probs[:, label_idx]

                auroc = float(roc_auc_score(true_labels[:, label_idx], probs_for_label))
                row = {
                    "condition": condition,
                    "model": model_name,
                    "label": label,
                    "auroc": auroc,
                }
                all_rows.append(row)

                if condition == "clean":
                    clean_auroc_lookup[(model_name, label)] = auroc

    results_df = pd.DataFrame(all_rows)
    results_df["condition"] = pd.Categorical(results_df["condition"], categories=CONDITION_ORDER, ordered=True)
    results_df = results_df.sort_values(["condition", "model", "label"]).reset_index(drop=True)

    for row_idx, row in results_df.iterrows():
        if row["condition"] == "clean":
            results_df.at[row_idx, "auroc_drop_from_clean"] = 0.0
        else:
            clean_auc = clean_auroc_lookup[(row["model"], row["label"])]
            results_df.at[row_idx, "auroc_drop_from_clean"] = float(clean_auc - float(row["auroc"]))

    results_df = results_df[["condition", "model", "label", "auroc", "auroc_drop_from_clean"]]
    results_df.to_csv(RESULTS_PATH, index=False)

    print("\nAll conditions complete. Preparing summary tables...", flush=True)
    summary_df = _make_summary_table(results_df)
    print("\nCondition x model summary (mean AUROC across all 5 labels):", flush=True)
    print(summary_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    drop_df = (
        results_df.loc[results_df["condition"] != "clean"]
        .sort_values("auroc_drop_from_clean", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )
    print("\nTop 10 AUROC drops from clean (descending):", flush=True)
    print(drop_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"), flush=True)

    figure, axes = plt.subplots(1, len(LABEL_COLUMNS), figsize=(4.5 * len(LABEL_COLUMNS), 4.8), sharey=True)
    if len(LABEL_COLUMNS) == 1:
        axes = [axes]

    for ax, label in zip(axes, LABEL_COLUMNS):
        ax.plot([0, len(CONDITION_ORDER) - 1], [0.5, 0.5], linestyle="--", color="gray", linewidth=1, alpha=0.5)
        for model_name in MODEL_ORDER:
            model_rows = results_df.loc[(results_df["model"] == model_name) & (results_df["label"] == label)].copy()
            model_rows = model_rows.sort_values("condition", key=lambda values: values.map({c: i for i, c in enumerate(CONDITION_ORDER)}))
            ax.plot(
                range(len(CONDITION_ORDER)),
                model_rows["auroc"].to_numpy(),
                color={"e04": "tab:blue", "e08": "tab:orange"}[model_name],
                marker="o",
                linewidth=2,
                label=f"{model_name.upper()}" if label == LABEL_COLUMNS[0] else None,
            )
        ax.set_title(label.removesuffix("_label"))
        ax.set_xticks(range(len(CONDITION_ORDER)))
        ax.set_xticklabels(CONDITION_ORDER, rotation=30, ha="right")
        ax.set_ylabel("AUROC")
        ax.set_ylim(0.35, 1.02)
        ax.grid(alpha=0.25)

    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("E04 vs E08 robustness to image perturbations (test split)")
    figure.tight_layout(rect=(0, 0.05, 1, 0.96))
    figure.savefig(FIGURE_PATH, dpi=250, bbox_inches="tight")
    plt.close(figure)

    print(f"\nSaved robustness CSV: {RESULTS_PATH}", flush=True)
    print(f"Saved robustness figure: {FIGURE_PATH}", flush=True)
    print(f"Total runtime: {time.time() - start_time:.1f}s", flush=True)


if __name__ == "__main__":
    main()
