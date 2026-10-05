"""Model loading and prediction helpers for the FastAPI app."""

from __future__ import annotations

import json
import base64
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any
import tempfile

import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError

from app.schemas import ClinicalInput, PredictionItem
from src.data.clinical_features import build_clinical_features
from src.data.image_dataset import LABEL_COLUMNS, build_transform
from src.explainability.gradcam import E04Explainer, E08Explainer, generate_gradcam
from src.explainability.shap_explainer import generate_shap_values

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"
TEMPERATURES_PATH = REPO_ROOT / "outputs" / "calibration" / "deployed_temperatures.json"
THRESHOLDS_PATH = REPO_ROOT / "models" / "threshold_optimization_results.json"

_CLEAN_LABELS = [label.removesuffix("_label") for label in LABEL_COLUMNS]


@dataclass(slots=True)
class PredictionArtifacts:
    """All model and preprocessing artifacts needed for inference."""

    train_frame: pd.DataFrame
    transform: Any
    e04: E04Explainer
    e08: E08Explainer
    temperatures: dict[str, float]
    thresholds_e04: dict[str, float]
    thresholds_e08: dict[str, float]
    allowed_values: dict[str, set[str]]
    background_clinical_features: np.ndarray
    background_feature_names: list[str]
    device: torch.device


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def calibrate_probabilities(raw_probabilities: np.ndarray, temperature: float) -> np.ndarray:
    clipped = np.clip(raw_probabilities.astype(np.float64), 1e-7, 1.0 - 1e-7)
    logits = np.log(clipped / (1.0 - clipped))
    return sigmoid(logits / temperature)


def _load_deployed_temperatures() -> dict[str, float]:
    if not TEMPERATURES_PATH.is_file():
        raise FileNotFoundError(f"Missing deployed temperatures: {TEMPERATURES_PATH}")
    with TEMPERATURES_PATH.open("r", encoding="utf-8") as fp:
        rows = json.load(fp)
    temperatures = {row["label"]: float(row["deployed_temperature"]) for row in rows if row["model"] == "e08"}
    missing = [label for label in LABEL_COLUMNS if label not in temperatures]
    if missing:
        raise KeyError(f"Missing E08 deployed temperatures for labels: {missing}")
    return temperatures


def _load_thresholds() -> tuple[dict[str, float], dict[str, float]]:
    if not THRESHOLDS_PATH.is_file():
        raise FileNotFoundError(f"Missing threshold optimization results: {THRESHOLDS_PATH}")
    with THRESHOLDS_PATH.open("r", encoding="utf-8") as fp:
        stored = json.load(fp)

    try:
        e04_thresholds = stored["vision_e04"]
        e08_thresholds = stored["fusion_e08"]
    except KeyError as error:
        raise KeyError("Threshold file is missing vision_e04 or fusion_e08 entries") from error

    thresholds_e04: dict[str, float] = {}
    thresholds_e08: dict[str, float] = {}
    for label in LABEL_COLUMNS:
        try:
            thresholds_e04[label] = float(e04_thresholds[label]["optimal_threshold"])
            thresholds_e08[label] = float(e08_thresholds[label]["optimal_threshold"])
        except KeyError as error:
            raise KeyError(f"Missing threshold entry for {label}") from error
    return thresholds_e04, thresholds_e08


def _build_background_features(train_frame: pd.DataFrame) -> tuple[np.ndarray, list[str]]:
    """Build the deterministic 50-row train clinical background used for SHAP."""
    rng = np.random.default_rng(42)
    candidate_indices = rng.choice(len(train_frame), size=min(50, len(train_frame)), replace=False)
    sample = train_frame.iloc[candidate_indices].copy()
    feature_frame, feature_names = build_clinical_features(sample, train_frame=train_frame)
    return feature_frame.to_numpy(dtype=np.float32), feature_names


def _derive_allowed_values(train_frame: pd.DataFrame) -> dict[str, set[str]]:
    allowed: dict[str, set[str]] = {}
    for column in ["sex", "race", "ethnicity", "insurance_type"]:
        values = train_frame[column].fillna("Unknown").astype(str).str.strip()
        allowed[column] = set(values.unique().tolist())
    return allowed


def _compute_train_medians(train_frame: pd.DataFrame) -> tuple[float, float]:
    age_median = train_frame.loc[~train_frame["age_implausible"], "age"].median()
    if pd.isna(age_median):
        age_median = train_frame["age"].median()
    bmi_median = train_frame.loc[~train_frame["recent_bmi_implausible"], "recent_bmi"].median()
    if pd.isna(bmi_median):
        bmi_median = train_frame["recent_bmi"].median()
    return float(age_median), float(bmi_median)


def _compute_train_standardization_stats(train_frame: pd.DataFrame) -> tuple[float, float, float, float]:
    age_median, bmi_median = _compute_train_medians(train_frame)
    age_filled = train_frame["age"].fillna(age_median)
    bmi_filled = train_frame["recent_bmi"].fillna(bmi_median)

    age_mean = float(age_filled.mean())
    age_std = float(age_filled.std())
    bmi_mean = float(bmi_filled.mean())
    bmi_std = float(bmi_filled.std())

    if age_std <= 0.0 or pd.isna(age_std):
        age_std = 1.0
    if bmi_std <= 0.0 or pd.isna(bmi_std):
        bmi_std = 1.0
    return age_mean, age_std, bmi_mean, bmi_std


class PredictionService:
    """Loads the trained models once and performs request-time inference."""

    def __init__(self) -> None:
        if not MANIFEST_PATH.is_file():
            raise FileNotFoundError(f"Missing manifest: {MANIFEST_PATH}")

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        manifest = pd.read_parquet(MANIFEST_PATH, engine='fastparquet')
        train_frame = manifest.loc[manifest["split"] == "train"].reset_index(drop=True)
        if train_frame.empty:
            raise ValueError("Training split is empty in the manifest.")

        self.train_frame = train_frame
        self.allowed_values = _derive_allowed_values(train_frame)
        self.age_median, self.bmi_median = _compute_train_medians(train_frame)
        self.age_mean, self.age_std, self.bmi_mean, self.bmi_std = _compute_train_standardization_stats(train_frame)
        self.transform = build_transform("val")
        self.e04 = E04Explainer().to(self.device)
        self.e08 = E08Explainer().to(self.device)
        self.e04.eval()
        self.e08.eval()
        self.temperatures = _load_deployed_temperatures()
        self.thresholds_e04, self.thresholds_e08 = _load_thresholds()
        self.background_clinical_features, self.background_feature_names = _build_background_features(train_frame)

    @classmethod
    def from_repository(cls) -> "PredictionService":
        return cls()

    def _validate_categorical(self, field_name: str, value: str) -> str:
        cleaned = value.strip()
        allowed = self.allowed_values[field_name]
        if cleaned not in allowed:
            allowed_text = ", ".join(sorted(allowed))
            raise ValueError(f"Invalid {field_name}: {value!r}. Expected one of: {allowed_text}.")
        return cleaned

    def _build_clinical_frame(self, clinical_input: ClinicalInput) -> pd.DataFrame:
        row = {
            "age": float(clinical_input.age),
            "sex": self._validate_categorical("sex", clinical_input.sex),
            "race": self._validate_categorical("race", clinical_input.race),
            "ethnicity": self._validate_categorical("ethnicity", clinical_input.ethnicity),
            "insurance_type": self._validate_categorical("insurance_type", clinical_input.insurance_type),
            "recent_bmi": clinical_input.recent_bmi,
            "age_implausible": 0,
            "recent_bmi_implausible": 0,
            "sex_missing": 0,
            "race_missing": 0,
            "ethnicity_missing": 0,
            "insurance_type_missing": 0,
        }
        return pd.DataFrame([row])

    def _encode_clinical_features(self, clinical_input: ClinicalInput) -> tuple[torch.Tensor, list[str]]:
        frame = self._build_clinical_frame(clinical_input)
        encoded, feature_names = build_clinical_features(frame, train_frame=self.train_frame)
        return torch.tensor(encoded.to_numpy(dtype=np.float32), dtype=torch.float32, device=self.device), feature_names

    def _load_image(self, image_bytes: bytes) -> torch.Tensor:
        try:
            with Image.open(BytesIO(image_bytes)) as image:
                pil_image = image.convert("RGB")
        except (UnidentifiedImageError, OSError) as error:
            raise ValueError("Uploaded file is not a valid image.") from error
        return self.transform(pil_image).unsqueeze(0).to(self.device)

    def _prepare_inputs(self, image_bytes: bytes, clinical_input: ClinicalInput) -> tuple[torch.Tensor, torch.Tensor, list[str]]:
        image_tensor = self._load_image(image_bytes)
        clinical_tensor, feature_names = self._encode_clinical_features(clinical_input)
        return image_tensor, clinical_tensor, feature_names

    def _image_bytes_to_temp_path(self, image_bytes: bytes) -> Path:
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".png")
        try:
            tmp.write(image_bytes)
            tmp.flush()
        finally:
            tmp.close()
        return Path(tmp.name)

    def predict(self, image_bytes: bytes, clinical_input: ClinicalInput) -> dict[str, PredictionItem]:
        image_tensor, clinical_tensor, _ = self._prepare_inputs(image_bytes, clinical_input)

        with torch.no_grad():
            e04_logits = self.e04(image_tensor)
            e08_logits = self.e08(image_tensor, clinical_tensor)

        e04_probabilities = torch.sigmoid(e04_logits).squeeze(0).cpu().numpy().astype(np.float64)
        e08_raw_probabilities = torch.sigmoid(e08_logits).squeeze(0).cpu().numpy().astype(np.float64)
        e08_probabilities = np.array(
            [
                calibrate_probabilities(np.array([raw_probability]), self.temperatures[label])[0]
                for raw_probability, label in zip(e08_raw_probabilities, LABEL_COLUMNS, strict=True)
            ],
            dtype=np.float64,
        )

        predictions: dict[str, PredictionItem] = {}
        for index, label in enumerate(LABEL_COLUMNS):
            clean_label = _CLEAN_LABELS[index]
            e04_probability = float(e04_probabilities[index])
            e08_probability = float(e08_probabilities[index])
            predictions[clean_label] = PredictionItem(
                e04_probability=e04_probability,
                e04_prediction=bool(e04_probability >= self.thresholds_e04[label]),
                e08_probability=e08_probability,
                e08_prediction=bool(e08_probability >= self.thresholds_e08[label]),
            )
        return predictions

    def explain(self, image_bytes: bytes, clinical_input: ClinicalInput) -> dict[str, dict[str, object]]:
        image_tensor, clinical_tensor, feature_names = self._prepare_inputs(image_bytes, clinical_input)
        temp_path = self._image_bytes_to_temp_path(image_bytes)

        try:
            with torch.no_grad():
                e08_logits = self.e08(image_tensor, clinical_tensor)
                vision_embedding = self.e08.densenet(image_tensor)

            e08_raw_probabilities = torch.sigmoid(e08_logits).squeeze(0).cpu().numpy().astype(np.float64)
            calibrated_e08_probabilities = np.array(
                [
                    calibrate_probabilities(np.array([raw_probability]), self.temperatures[label])[0]
                    for raw_probability, label in zip(e08_raw_probabilities, LABEL_COLUMNS, strict=True)
                ],
                dtype=np.float64,
            )

            explanations: dict[str, dict[str, object]] = {}
            clinical_vector = clinical_tensor.squeeze(0)
            for label_idx, label in enumerate(LABEL_COLUMNS):
                clean_label = _CLEAN_LABELS[label_idx]
                _, overlay_image = generate_gradcam(
                    explainer=self.e08,
                    image_path=str(temp_path),
                    clinical_feat=clinical_vector,
                    label_idx=label_idx,
                    device=self.device,
                )

                shap_values = generate_shap_values(
                    self.e08.fusion_model,
                    vision_embedding.squeeze(0),
                    self.background_clinical_features,
                    clinical_vector,
                    label_idx=label_idx,
                    device=self.device,
                )

                top_indices = np.argsort(np.abs(shap_values))[-5:][::-1]
                top_shap_features = [
                    {"feature_name": feature_names[index], "shap_value": float(shap_values[index])}
                    for index in top_indices
                ]

                with BytesIO() as buffer:
                    Image.fromarray(overlay_image).save(buffer, format="PNG")
                    gradcam_base64 = base64.b64encode(buffer.getvalue()).decode("ascii")

                explanations[clean_label] = {
                    "gradcam_image_base64": gradcam_base64,
                    "calibrated_probability": float(calibrated_e08_probabilities[label_idx]),
                    "top_shap_features": top_shap_features,
                }

            return explanations
        finally:
            try:
                temp_path.unlink(missing_ok=True)
            except OSError:
                pass
