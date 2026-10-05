"""Compare E08 MC-dropout selective prediction with calibrated abstention.

The DenseNet backbone stays in evaluation mode.  Only the two trained dropout
layers in E08's fusion MLP are switched to training mode, so BatchNorm and all
other inference behavior remain deterministic.
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
from PIL import Image
from torch.utils.data import DataLoader, Dataset

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.fusion_dataset import FusionDataset
from src.data.image_dataset import LABEL_COLUMNS, build_transform
from src.explainability.gradcam import E08Explainer


MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
EMBEDDINGS_PATH = REPO_ROOT / "models" / "vision_embeddings_test.npy"
PREDICTIONS_PATH = REPO_ROOT / "outputs" / "test_predictions.parquet"
TEMPERATURES_PATH = REPO_ROOT / "outputs" / "calibration" / "deployed_temperatures.json"
THRESHOLDS_PATH = REPO_ROOT / "models" / "threshold_optimization_results.json"
CALIBRATED_RESULTS_PATH = REPO_ROOT / "outputs" / "calibration" / "abstention_coverage_results.csv"
OUTPUT_DIR = REPO_ROOT / "outputs" / "calibration"
MC_SAMPLES_PATH = OUTPUT_DIR / "mc_dropout_e08.parquet"
MC_RESULTS_PATH = OUTPUT_DIR / "mc_dropout_coverage_results.csv"
FIGURE_PATH = OUTPUT_DIR / "mc_dropout_vs_calibration_coverage.png"

N_MC_SAMPLES = 50
IMAGE_BATCH_SIZE = 16
FUSION_BATCH_SIZE = 256
NUM_WORKERS = 0
RANDOM_SEED = 42
EPSILON = 1e-7
MAD_WARNING_THRESHOLD = 0.05
COVERAGES = np.round(np.arange(100, 49, -5) / 100, 2)
E08_RESULT_KEY = "fusion_e08"


class TestImageDataset(Dataset):
    """Load test images in manifest order with the deterministic val transform."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self.transform = build_transform("val")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int) -> tuple[int, torch.Tensor]:
        row = self.frame.iloc[index]
        with Image.open(row["local_image_path"]) as image:
            tensor = self.transform(image.convert("RGB"))
        return index, tensor


def sigmoid(values: np.ndarray) -> np.ndarray:
    """Compute sigmoid values from NumPy logits."""
    return 1.0 / (1.0 + np.exp(-values))


def calibrate_probabilities(probabilities: np.ndarray, temperature: float) -> np.ndarray:
    """Apply the deployed temperature to raw saved probabilities."""
    clipped = np.clip(probabilities, EPSILON, 1.0 - EPSILON)
    logits = np.log(clipped / (1.0 - clipped))
    return sigmoid(logits / temperature)


def load_temperatures() -> dict[str, float]:
    """Load deployed E08 temperatures, one for each target label."""
    with TEMPERATURES_PATH.open("r", encoding="utf-8") as stream:
        rows = json.load(stream)
    temperatures = {
        row["label"]: float(row["deployed_temperature"])
        for row in rows
        if row["model"] == "e08"
    }
    missing = set(LABEL_COLUMNS).difference(temperatures)
    if missing:
        raise ValueError(f"Missing deployed E08 temperatures for: {sorted(missing)}")
    return temperatures


def load_thresholds() -> dict[str, float]:
    """Load the existing Phase 7 E08 Youden-J thresholds."""
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as stream:
        stored = json.load(stream)
    try:
        return {
            label: float(stored[E08_RESULT_KEY][label]["optimal_threshold"])
            for label in LABEL_COLUMNS
        }
    except KeyError as error:
        raise ValueError("Missing an E08 Phase 7 optimal threshold.") from error


def enable_e08_dropout(explainer: E08Explainer) -> list[tuple[str, nn.Dropout]]:
    """Leave E08 in eval mode except for its verified fusion dropout modules."""
    explainer.eval()
    dropout_modules = [
        (name, module)
        for name, module in explainer.fusion_model.named_modules()
        if isinstance(module, nn.Dropout)
    ]
    expected_names = {"clinical_encoder.2", "fusion_head.2"}
    actual_names = {name for name, _ in dropout_modules}
    if actual_names != expected_names:
        raise RuntimeError(
            "Expected exactly E08's clinical_encoder.2 and fusion_head.2 dropout "
            f"modules, found: {sorted(actual_names)}"
        )
    for _, module in dropout_modules:
        module.train()

    if explainer.densenet.training or explainer.fusion_model.training:
        raise RuntimeError("E08 parent modules must remain in eval mode for MC dropout.")
    if any(not module.training for _, module in dropout_modules):
        raise RuntimeError("Failed to enable an E08 dropout module for MC sampling.")
    return dropout_modules


def compute_embeddings(
    explainer: E08Explainer,
    frame: pd.DataFrame,
    device: torch.device,
) -> torch.Tensor:
    """Run each test image through the deterministic DenseNet exactly once."""
    loader = DataLoader(
        TestImageDataset(frame),
        batch_size=IMAGE_BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=(device.type == "cuda"),
    )
    embeddings = torch.empty((len(frame), 1024), dtype=torch.float32)
    print(f"Computing deterministic DenseNet embeddings once for {len(frame):,} test images...")
    with torch.no_grad():
        for seen, (indices, images) in enumerate(loader, start=1):
            values = explainer.densenet(images.to(device, non_blocking=True))
            embeddings[indices] = values.cpu()
            completed = min(seen * IMAGE_BATCH_SIZE, len(frame))
            if completed % (IMAGE_BATCH_SIZE * 10) == 0 or completed == len(frame):
                print(f"  {completed}/{len(frame)} images embedded")
    return embeddings


def generate_mc_probabilities(
    explainer: E08Explainer,
    embeddings: torch.Tensor,
    clinical_features: torch.Tensor,
    temperatures: dict[str, float],
    device: torch.device,
) -> np.ndarray:
    """Sample deployed-calibrated E08 probabilities with only fusion dropout active."""
    n_rows = len(embeddings)
    n_labels = len(LABEL_COLUMNS)
    samples = np.empty((n_rows, N_MC_SAMPLES, n_labels), dtype=np.float32)
    temperature_tensor = torch.tensor(
        [temperatures[label] for label in LABEL_COLUMNS], dtype=torch.float32, device=device
    )
    print(f"Generating {N_MC_SAMPLES} MC-dropout samples per test image...")
    with torch.no_grad():
        for sample_index in range(N_MC_SAMPLES):
            for start in range(0, n_rows, FUSION_BATCH_SIZE):
                stop = min(start + FUSION_BATCH_SIZE, n_rows)
                logits = explainer.fusion_model(
                    embeddings[start:stop].to(device, non_blocking=True),
                    clinical_features[start:stop].to(device, non_blocking=True),
                )
                samples[start:stop, sample_index] = torch.sigmoid(logits / temperature_tensor).cpu().numpy()
            if (sample_index + 1) % 5 == 0 or sample_index + 1 == N_MC_SAMPLES:
                print(f"  {sample_index + 1}/{N_MC_SAMPLES} MC samples complete")
    return samples


def save_mc_samples(frame: pd.DataFrame, means: np.ndarray, stds: np.ndarray) -> pd.DataFrame:
    """Save all requested targets, deployed MC means, and MC standard deviations."""
    output = pd.DataFrame({"row_index": np.arange(len(frame), dtype=int)})
    if "patient_id" in frame:
        output["patient_id"] = frame["patient_id"].to_numpy()
    elif "deid_patient_id" in frame:
        output["patient_id"] = frame["deid_patient_id"].to_numpy()
    for index, label in enumerate(LABEL_COLUMNS):
        output[f"{label}_true"] = frame[label].to_numpy(dtype=int)
        output[f"{label}_mc_mean"] = means[:, index]
        output[f"{label}_mc_std"] = stds[:, index]
    output.to_parquet(MC_SAMPLES_PATH, index=False, engine='fastparquet')
    return output


def print_sanity_check(mc_samples: pd.DataFrame, temperatures: dict[str, float]) -> None:
    """Compare deployed MC means against deployed deterministic E08 probabilities."""
    predictions = pd.read_parquet(PREDICTIONS_PATH, engine='fastparquet')
    if len(predictions) != len(mc_samples) or not np.array_equal(
        predictions["row_index"].to_numpy(), mc_samples["row_index"].to_numpy()
    ):
        raise ValueError("test_predictions.parquet row ordering does not match the test manifest.")

    print("\nMC SANITY CHECK: mean absolute difference from deployed single-pass E08")
    for label in LABEL_COLUMNS:
        deployed = calibrate_probabilities(
            predictions[f"{label}_e08"].to_numpy(dtype=float), temperatures[label]
        )
        mad = float(np.mean(np.abs(mc_samples[f"{label}_mc_mean"].to_numpy() - deployed)))
        suffix = " WARNING: exceeds 0.05" if mad > MAD_WARNING_THRESHOLD else ""
        print(f"  {label}: {mad:.6f}{suffix}")


def metrics_at_coverage(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    uncertainty: np.ndarray,
    threshold: float,
    coverage: float,
) -> dict[str, float | int]:
    """Evaluate the lowest-uncertainty retained fraction without inventing undefined rates."""
    n_retained = max(1, int(np.ceil(coverage * len(y_true))))
    retained = np.argsort(uncertainty, kind="stable")[:n_retained]
    true = y_true[retained]
    predicted = (probabilities[retained] >= threshold).astype(int)
    positive = true == 1
    negative = true == 0
    sensitivity = float((predicted[positive] == 1).mean()) if positive.any() else float("nan")
    specificity = float((predicted[negative] == 0).mean()) if negative.any() else float("nan")
    balanced_accuracy = (
        (sensitivity + specificity) / 2
        if not np.isnan(sensitivity) and not np.isnan(specificity)
        else float("nan")
    )
    return {
        "n_retained": n_retained,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "balanced_accuracy": balanced_accuracy,
    }


def build_mc_coverage_results(mc_samples: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Compute E08 coverage curves using ascending MC standard deviation as certainty."""
    rows: list[dict[str, object]] = []
    for label in LABEL_COLUMNS:
        y_true = mc_samples[f"{label}_true"].to_numpy(dtype=int)
        probabilities = mc_samples[f"{label}_mc_mean"].to_numpy(dtype=float)
        uncertainty = mc_samples[f"{label}_mc_std"].to_numpy(dtype=float)
        for coverage in COVERAGES:
            rows.append(
                {
                    "label": label,
                    "model": "e08_mc_dropout",
                    "coverage": float(coverage),
                    **metrics_at_coverage(y_true, probabilities, uncertainty, thresholds[label], float(coverage)),
                }
            )
    return pd.DataFrame(rows)


def load_calibrated_e08_results() -> pd.DataFrame:
    """Load the existing calibrated-confidence E08 curve for a like-for-like overlay."""
    results = pd.read_csv(CALIBRATED_RESULTS_PATH)
    e08 = results.loc[results["model"] == "e08"].copy()
    expected = len(LABEL_COLUMNS) * len(COVERAGES)
    if len(e08) != expected:
        raise ValueError(f"Expected {expected} calibrated E08 coverage rows, found {len(e08)}.")
    return e08


def save_comparison_figure(calibrated: pd.DataFrame, mc_results: pd.DataFrame) -> None:
    """Overlay calibrated-confidence and MC-dropout selective-prediction curves."""
    figure, axes = plt.subplots(2, 3, figsize=(17, 9), sharex=True, sharey=True)
    flat_axes = list(axes.flat)
    for axis, label in zip(flat_axes, LABEL_COLUMNS):
        baseline = calibrated.loc[calibrated["label"] == label].sort_values("coverage", ascending=False)
        mc = mc_results.loc[mc_results["label"] == label].sort_values("coverage", ascending=False)
        axis.plot(baseline["coverage"], baseline["sensitivity"], color="#F58518", linestyle="-", marker="o", label="Calibrated ranking sensitivity")
        axis.plot(baseline["coverage"], baseline["specificity"], color="#F58518", linestyle="--", marker="o", label="Calibrated ranking specificity")
        axis.plot(mc["coverage"], mc["sensitivity"], color="#7A5195", linestyle="-", marker="s", label="MC-dropout ranking sensitivity")
        axis.plot(mc["coverage"], mc["specificity"], color="#7A5195", linestyle="--", marker="s", label="MC-dropout ranking specificity")
        axis.set_title(label.removesuffix("_label"))
        axis.set(xlabel="Coverage (retained fraction)", ylabel="Metric", xlim=(1.0, 0.5), ylim=(0.0, 1.05))
        axis.grid(alpha=0.3)
    flat_axes[-1].axis("off")
    handles, labels = flat_axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=2, frameon=False)
    figure.suptitle("E08 Selective Prediction: Calibrated Confidence vs MC Dropout", y=0.98)
    figure.tight_layout(rect=(0, 0.09, 1, 0.95))
    figure.savefig(FIGURE_PATH, dpi=200, bbox_inches="tight")
    plt.close(figure)


def print_coverage_summary(calibrated: pd.DataFrame, mc_results: pd.DataFrame) -> None:
    """Print the requested label-level balanced-accuracy comparison at 70% coverage."""
    baseline = calibrated.loc[calibrated["coverage"] == 0.7, ["label", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "calibrated_confidence_balanced_accuracy"}
    )
    mc = mc_results.loc[mc_results["coverage"] == 0.7, ["label", "balanced_accuracy"]].rename(
        columns={"balanced_accuracy": "mc_dropout_balanced_accuracy"}
    )
    summary = baseline.merge(mc, on="label", validate="one_to_one")
    print("\nBALANCED ACCURACY AT COVERAGE 0.7")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.4f}"))


def main() -> None:
    """Generate deployed E08 MC-dropout uncertainty and compare coverage curves."""
    required_paths = (
        MANIFEST_PATH,
        EMBEDDINGS_PATH,
        PREDICTIONS_PATH,
        TEMPERATURES_PATH,
        THRESHOLDS_PATH,
        CALIBRATED_RESULTS_PATH,
    )
    for path in required_paths:
        if not path.is_file():
            raise FileNotFoundError(f"Required input was not found: {path}")

    torch.manual_seed(RANDOM_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(RANDOM_SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    manifest = pd.read_parquet(MANIFEST_PATH, engine='fastparquet')
    test_frame = manifest.loc[manifest["split"] == "test"].reset_index(drop=True)
    if test_frame.empty:
        raise ValueError("The manifest has no test rows.")
    print(f"Loaded {len(test_frame):,} test rows from {MANIFEST_PATH}")

    temperatures = load_temperatures()
    thresholds = load_thresholds()
    print("Deployed E08 temperatures: " + ", ".join(f"{label}={temperatures[label]:.4f}" for label in LABEL_COLUMNS))
    print("E08 thresholds: " + ", ".join(f"{label}={thresholds[label]:.2f}" for label in LABEL_COLUMNS))

    # FusionDataset reproduces the exact leakage-safe Phase 6/7 clinical features.
    fusion_dataset = FusionDataset(str(MANIFEST_PATH), "test", str(EMBEDDINGS_PATH))
    if len(fusion_dataset) != len(test_frame):
        raise RuntimeError("FusionDataset and manifest test-row counts differ.")

    explainer = E08Explainer().to(device)
    dropout_modules = enable_e08_dropout(explainer)
    print("Active MC-dropout modules: " + ", ".join(f"{name} (p={module.p})" for name, module in dropout_modules))
    print("DenseNet backbone remains in eval mode: True")

    start = time.time()
    embeddings = compute_embeddings(explainer, test_frame, device)
    samples = generate_mc_probabilities(explainer, embeddings, fusion_dataset.features, temperatures, device)
    means = samples.mean(axis=1)
    stds = samples.std(axis=1)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    mc_samples = save_mc_samples(test_frame, means, stds)
    print_sanity_check(mc_samples, temperatures)

    mc_results = build_mc_coverage_results(mc_samples, thresholds)
    mc_results.to_csv(MC_RESULTS_PATH, index=False)
    calibrated_results = load_calibrated_e08_results()
    save_comparison_figure(calibrated_results, mc_results)
    print_coverage_summary(calibrated_results, mc_results)

    elapsed = time.time() - start
    print(f"\nSaved MC sample data: {MC_SAMPLES_PATH}")
    print(f"Saved MC-dropout coverage data: {MC_RESULTS_PATH}")
    print(f"Saved comparison figure: {FIGURE_PATH}")
    print(f"Completed in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
