from contextlib import asynccontextmanager

import mlflow.sklearn
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from diasense.config import (
    CLINICAL_FEATURES,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    RISK_MAPPING_FILE,
)
from diasense.risk import load_risk_mapping

# Global variable to hold our pipeline
model_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):

    global model_pipeline
    try:
        print("Connecting to MLflow to fetch the Production model...")
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{MLFLOW_REGISTERED_MODEL_NAME}@{MLFLOW_MODEL_ALIAS}"
        model_pipeline = mlflow.sklearn.load_model(model_uri)
        print("Model loaded successfully!")
    except Exception as exc:
        print(f"Failed to load model from MLflow: {exc}")
        

    yield  # Hand over control to FastAPI

    # Shutdown
    model_pipeline = None


app = FastAPI(
    title="DiaSense Risk API",
    description="API for predicting diabetes risk based on clinical parameters.",
    version="1.0.0",
    lifespan=lifespan,
)


class PatientInput(BaseModel):
    Pregnancies: int = Field(..., ge=0, description="Number of times pregnant")
    Glucose: float = Field(..., ge=0, description="Plasma glucose concentration")
    BloodPressure: float = Field(..., ge=0, description="Diastolic blood pressure (mm Hg)")
    SkinThickness: float = Field(..., ge=0, description="Triceps skinfold thickness (mm)")
    Insulin: float = Field(..., ge=0, description="2-Hour serum insulin (mu U/mL)")
    BMI: float = Field(..., ge=0, description="Body mass index")
    DiabetesPedigreeFunction: float = Field(..., ge=0, description="Diabetes pedigree function")
    Age: int = Field(..., gt=0, description="Age in years")


class RiskResponse(BaseModel):
    risk_level: int
    risk_category: str
    message: str


@app.get("/")
def read_root():
    return {"message": "Welcome to the DiaSense Risk Prediction API. Try /docs for the Swagger UI."}


@app.get("/health")
def health_check():
    """Healthcheck endpoint used by Docker."""
    status = "healthy" if model_pipeline is not None else "degraded (model not loaded)"
    return {"status": status}


@app.post("/predict", response_model=RiskResponse)
def predict_risk(patient: PatientInput):
    if model_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail="The prediction model is currently unavailable. Check MLflow connection.",
        )

    # 1_Convert input to a single-row DataFrame
    df = pd.DataFrame([patient.model_dump()])

    # 2_Ensure columns are in the exact order expected by the pipeline
    df = df[CLINICAL_FEATURES]
    
    # Cast to float for consistency
    df = df.astype(float)

    # 3_Predict
    try:
        prediction = int(model_pipeline.predict(df)[0])
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction error: {exc}")

    # 4_Format response
    risk_mapping = load_risk_mapping(RISK_MAPPING_FILE)
    category = risk_mapping.get(prediction, "Unknown")
    is_high_risk = category == "High Risk"
    msg = "Diabetes Risk Detected" if is_high_risk else "No Diabetes Risk Detected"

    return RiskResponse(
        risk_level=prediction,
        risk_category=category,
        message=msg,
    )
