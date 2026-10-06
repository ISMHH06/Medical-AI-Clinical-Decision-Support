"""Pytest suite for the FastAPI backend, run in-process via TestClient."""

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from app.main import app

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = REPO_ROOT / "data" / "processed" / "image_subset_manifest.parquet"

EXPECTED_LABELS = {"Cardiomegaly", "Edema", "Atelectasis", "Pleural Effusion", "Consolidation"}
ALLOWED_ORIGIN = "http://localhost:5173"

# Regression guard: verified values from Phase 13 manual HTTP verification
# (test split row 0, exact same request shape as the fixture below).
REGRESSION_CARDIO_E04 = 0.8676
REGRESSION_CARDIO_E08 = 0.8648
REGRESSION_TOLERANCE = 1e-4


@pytest.fixture(scope="module")
def client():
    """One TestClient per module: the lifespan loads the models once."""
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture(scope="module")
def test_row0():
    """Test split row 0 image bytes + clinical fields, shared by every test."""
    manifest = pd.read_parquet(MANIFEST_PATH, engine="fastparquet")
    row = manifest.loc[manifest["split"] == "test"].reset_index(drop=True).iloc[0]
    image_path = Path(row["local_image_path"])
    clinical = {
        "age": str(float(row["age"])),
        "sex": str(row["sex"]),
        "race": str(row["race"]),
        "ethnicity": str(row["ethnicity"]),
        "insurance_type": str(row["insurance_type"]),
    }
    if pd.notna(row["recent_bmi"]):
        clinical["recent_bmi"] = str(float(row["recent_bmi"]))
    return {"image_bytes": image_path.read_bytes(), "clinical": clinical}


def _image_files(image_bytes: bytes, filename: str = "test.png") -> dict:
    return {"image": (filename, image_bytes, "image/png")}


def _assert_probability(value: object) -> None:
    assert isinstance(value, (int, float)) and not isinstance(value, bool), f"not a number: {value!r}"
    assert 0.0 <= float(value) <= 1.0, f"probability out of range: {value!r}"


def _post(client, test_row0, path: str, *, clinical: dict | None = None, files: dict | None = None):
    return client.post(
        path,
        files=files if files is not None else _image_files(test_row0["image_bytes"]),
        data=clinical if clinical is not None else test_row0["clinical"],
    )


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_debug_memory_shape(client):
    response = client.get("/debug/memory")
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"cuda_available", "allocated_mb", "max_allocated_mb", "reserved_mb"}
    assert isinstance(body["cuda_available"], bool)
    for key in ("allocated_mb", "max_allocated_mb", "reserved_mb"):
        value = body[key]
        assert value is None or (
            isinstance(value, (int, float)) and not isinstance(value, bool)
        ), f"{key} should be float or null, got {value!r}"


def test_predict_happy_path(client, test_row0):
    response = _post(client, test_row0, "/predict")
    assert response.status_code == 200, response.text
    predictions = response.json()["predictions"]
    assert set(predictions) == EXPECTED_LABELS
    for label, item in predictions.items():
        for key in ("e04_probability", "e04_prediction", "e08_probability", "e08_prediction"):
            assert key in item, f"{label} is missing {key}"
        _assert_probability(item["e04_probability"])
        _assert_probability(item["e08_probability"])
        assert isinstance(item["e04_prediction"], bool), f"{label} e04_prediction not bool"
        assert isinstance(item["e08_prediction"], bool), f"{label} e08_prediction not bool"
    cardio = predictions["Cardiomegaly"]
    assert cardio["e08_probability"] == pytest.approx(REGRESSION_CARDIO_E08, abs=REGRESSION_TOLERANCE), (
        f"Cardiomegaly e08_probability={cardio['e08_probability']!r}, "
        f"expected {REGRESSION_CARDIO_E08} +/- {REGRESSION_TOLERANCE} (Phase 13 verified)"
    )
    assert cardio["e04_probability"] == pytest.approx(REGRESSION_CARDIO_E04, abs=REGRESSION_TOLERANCE), (
        f"Cardiomegaly e04_probability={cardio['e04_probability']!r}, "
        f"expected {REGRESSION_CARDIO_E04} +/- {REGRESSION_TOLERANCE} (Phase 13 verified)"
    )


def test_predict_missing_required_field(client, test_row0):
    clinical = {key: value for key, value in test_row0["clinical"].items() if key != "age"}
    response = _post(client, test_row0, "/predict", clinical=clinical)
    assert response.status_code == 422, response.text


def test_predict_invalid_categorical(client, test_row0):
    clinical = {**test_row0["clinical"], "sex": "NotARealCategory"}
    response = _post(client, test_row0, "/predict", clinical=clinical)
    assert response.status_code == 422, response.text
    detail = str(response.json().get("detail", ""))
    assert "sex" in detail, f"422 detail should mention the field name 'sex', got: {detail!r}"


def test_predict_empty_image(client, test_row0):
    response = _post(client, test_row0, "/predict", files=_image_files(b""))
    assert response.status_code == 422, response.text


def test_predict_corrupt_image(client, test_row0):
    response = _post(
        client, test_row0, "/predict", files=_image_files(b"not an image", filename="corrupt.png")
    )
    assert response.status_code == 422, response.text


@pytest.mark.slow
def test_explain_happy_path(client, test_row0):
    response = _post(client, test_row0, "/explain")
    assert response.status_code == 200, response.text
    explanations = response.json()["explanations"]
    assert set(explanations) == EXPECTED_LABELS
    for label, item in explanations.items():
        gradcam = item.get("gradcam_image_base64")
        assert isinstance(gradcam, str) and gradcam, f"{label}: gradcam_image_base64 must be a non-empty string"
        _assert_probability(item.get("calibrated_probability"))
        shap_features = item.get("top_shap_features")
        assert isinstance(shap_features, list), f"{label}: top_shap_features must be a list"
        assert len(shap_features) == 5, f"{label}: expected 5 SHAP features, got {len(shap_features)}"
        for feature in shap_features:
            assert isinstance(feature.get("feature_name"), str) and feature["feature_name"], (
                f"{label}: each SHAP feature needs a non-empty feature_name"
            )
            shap_value = feature.get("shap_value")
            assert isinstance(shap_value, (int, float)) and not isinstance(shap_value, bool), (
                f"{label}: shap_value must be numeric, got {shap_value!r}"
            )


def test_cors_headers_present(client):
    response = client.get("/health", headers={"Origin": ALLOWED_ORIGIN})
    assert response.status_code == 200
    assert response.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN

    preflight = client.options(
        "/predict",
        headers={
            "Origin": ALLOWED_ORIGIN,
            "Access-Control-Request-Method": "POST",
            "Access-Control-Request-Headers": "content-type",
        },
    )
    assert preflight.status_code == 200, preflight.text
    assert preflight.headers.get("access-control-allow-origin") == ALLOWED_ORIGIN
