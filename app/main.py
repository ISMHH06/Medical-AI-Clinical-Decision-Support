"""FastAPI application exposing multimodal chest X-ray predictions."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

from app.inference import PredictionService
from app.schemas import ClinicalInput, ExplanationResponse, HealthResponse, PredictionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.prediction_service = PredictionService.from_repository()
    yield


app = FastAPI(title="Medical AI Clinical Decision Support", version="0.1.0", lifespan=lifespan)


async def get_prediction_service(request: Request) -> PredictionService:
    service = getattr(request.app.state, "prediction_service", None)
    if service is None:
        raise HTTPException(status_code=503, detail="Prediction service is not ready.")
    return service


async def parse_clinical_input(
    age: str = Form(...),
    sex: str = Form(...),
    race: str = Form(...),
    ethnicity: str = Form(...),
    insurance_type: str = Form(...),
    recent_bmi: str | None = Form(None),
) -> ClinicalInput:
    def parse_float(field_name: str, value: str | None, required: bool) -> float | None:
        if value is None:
            if required:
                raise HTTPException(status_code=422, detail=f"Missing required clinical field: {field_name}")
            return None
        cleaned = value.strip()
        if not cleaned:
            if required:
                raise HTTPException(status_code=422, detail=f"Missing required clinical field: {field_name}")
            return None
        try:
            return float(cleaned)
        except ValueError as error:
            raise HTTPException(status_code=422, detail=f"Invalid numeric value for {field_name}: {value!r}") from error

    return ClinicalInput(
        age=parse_float("age", age, required=True),
        sex=sex.strip() if sex is not None else "",
        race=race.strip() if race is not None else "",
        ethnicity=ethnicity.strip() if ethnicity is not None else "",
        insurance_type=insurance_type.strip() if insurance_type is not None else "",
        recent_bmi=parse_float("recent_bmi", recent_bmi, required=False),
    )


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/predict", response_model=PredictionResponse)
async def predict_endpoint(
    image: UploadFile = File(...),
    clinical_input: ClinicalInput = Depends(parse_clinical_input),
    service: PredictionService = Depends(get_prediction_service),
) -> PredictionResponse:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image file is empty.")

    try:
        predictions = service.predict(image_bytes, clinical_input)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return PredictionResponse(predictions=predictions)


@app.post("/explain", response_model=ExplanationResponse)
async def explain_endpoint(
    image: UploadFile = File(...),
    clinical_input: ClinicalInput = Depends(parse_clinical_input),
    service: PredictionService = Depends(get_prediction_service),
) -> ExplanationResponse:
    image_bytes = await image.read()
    if not image_bytes:
        raise HTTPException(status_code=422, detail="Uploaded image file is empty.")

    try:
        explanations = service.explain(image_bytes, clinical_input)
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return ExplanationResponse(explanations=explanations)
