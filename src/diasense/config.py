import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(PROJECT_ROOT/ ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
RAW_DATA_FILE = RAW_DATA_DIR / "dataset-diabete.csv"

MODELS_DIR = PROJECT_ROOT / "models"
NOTEBOOKS = PROJECT_ROOT / "notebooks"

REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

METRICS_DIR = REPORTS_DIR / "metrics"


CLINICAL_FEATURES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

OUTLIER_COLUMNS = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
]

# 0 in this columns means not measured so not a real value
ZERO_AS_MISSING = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
]

TARGET_COLUMN = "Cluster"
RISK_COLUMN = "risk_category"

RANDOM_STATE = 42
TEST_SIZE = 0.2

CLINICAL_THRESHOLDS = {
    "Glucose": 126.0,
    "BMI": 30.0,
    "DiabetesPedigreeFunction": 0.5,
}

K_RANGE = range(2, 11)


KNN_NEIGHBORS = 5
CLIP_Z_THRESHOLD = 3.0
CLUSTERING_FEATURES = ["Glucose", "BMI", "DiabetesPedigreeFunction"]
KMEANS_K = 2

RISK_LABELS = {
    0: "Low Risk",
    1: "High Risk",
}


# Artifact file paths 
CLUSTER_PIPELINE_FILE = MODELS_DIR / "cluster_input_pipeline.joblib"
KMEANS_MODEL_FILE = MODELS_DIR / "kmeans_model.joblib"
RISK_MAPPING_FILE = MODELS_DIR / "risk_mapping.json"
CLASSIFICATION_PIPELINE_FILE = MODELS_DIR / "classification_pipeline.joblib"
CLASSIFICATION_RESULTS_FILE = METRICS_DIR / "classification_results.json"
LABELED_DATA_FILE = PROCESSED_DATA_DIR / "dataset_labeled.csv"

# MLflow
MLFLOW_TRACKING_URI       = os.getenv("MLFLOW_TRACKING_URI",       "http://mlflow:5000")
MLFLOW_EXPERIMENT_NAME    = os.getenv("MLFLOW_EXPERIMENT_NAME",    "DiaSense_Diabetes_Risk_Prediction")
MLFLOW_REGISTERED_MODEL_NAME = os.getenv("MLFLOW_REGISTERED_MODEL_NAME", "DiaSense_Risk_Classifier")
MLFLOW_MODEL_ALIAS        = os.getenv("MLFLOW_MODEL_ALIAS",        "production")
MLFLOW_PROJECT_TAG        = os.getenv("MLFLOW_PROJECT_TAG",        "DiaSense")