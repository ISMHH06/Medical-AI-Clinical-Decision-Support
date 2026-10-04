"""Request and response schemas for the prediction API."""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field


class PredictionItem(BaseModel):
    """Per-label probabilities and binary decisions for both models."""

    e04_probability: float = Field(..., ge=0.0, le=1.0)
    e04_prediction: bool
    e08_probability: float = Field(..., ge=0.0, le=1.0)
    e08_prediction: bool


class ShapFeatureItem(BaseModel):
    """One feature/value pair for the explanation payload."""

    feature_name: str
    shap_value: float


class ExplanationItem(BaseModel):
    """Per-label Grad-CAM and SHAP explanation payload."""

    gradcam_image_base64: str
    calibrated_probability: float = Field(..., ge=0.0, le=1.0)
    top_shap_features: list[ShapFeatureItem]


class PredictionResponse(BaseModel):
    """Prediction payload keyed by clean label name."""

    predictions: dict[str, PredictionItem]


class ExplanationResponse(BaseModel):
    """Explanation payload keyed by clean label name."""

    explanations: dict[str, ExplanationItem]


class HealthResponse(BaseModel):
    """Simple liveness response."""

    status: str


@dataclass(slots=True)
class ClinicalInput:
    """Parsed clinical form fields used to construct the model features."""

    age: float
    sex: str
    race: str
    ethnicity: str
    insurance_type: str
    recent_bmi: float | None = None
