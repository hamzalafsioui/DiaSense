"""
Extracts data -> Preprocesses -> Clusters -> Learns Risk Mapping -> Trains Classifiers -> Updates MLflow.
"""

from datetime import datetime, timedelta
import pandas as pd
import sys
import os

# Ensure the src folder is in the Python path so we can import 'diasense'
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../src")))

from airflow import DAG
from airflow.operators.python import PythonOperator

# Task 1 ==> Preprocessing
def run_preprocessing():
    from diasense.data import load_raw_data, validate_columns
    from diasense.preprocessing import build_preprocessing_pipeline, save_artifact
    from diasense.config import PROCESSED_DATA_DIR, CLUSTER_PIPELINE_FILE
    
    print("Loading raw data...")
    df = load_raw_data()
    validate_columns(df)
    
    print("Building and fitting preprocessing pipeline...")
    pipeline = build_preprocessing_pipeline()
    df_preprocessed = pipeline.fit_transform(df)
    
    print("Saving preprocessed data and pipeline artifact...")
    save_artifact(pipeline, CLUSTER_PIPELINE_FILE)
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df_preprocessed.to_csv(PROCESSED_DATA_DIR / "dataset_preprocessed.csv", index=False)
    print("Preprocessing completed.")

# Task 2 ==> Clustering & Risk Mapping
def run_clustering_and_mapping():
    from diasense.clustering import fit_kmeans
    from diasense.risk import learn_risk_mapping, add_risk_category, save_risk_mapping
    from diasense.preprocessing import save_artifact, inverse_scale
    import joblib
    from diasense.config import (
        PROCESSED_DATA_DIR,
        CLUSTER_PIPELINE_FILE,
        CLUSTERING_FEATURES,
        KMEANS_K,
        KMEANS_MODEL_FILE,
        RISK_MAPPING_FILE,
        LABELED_DATA_FILE
    )
    
    print("Loading preprocessed data...")
    df = pd.read_csv(PROCESSED_DATA_DIR / "dataset_preprocessed.csv")
    
    print(f"Fitting KMeans with k={KMEANS_K}...")
    X_cluster = df[CLUSTERING_FEATURES]
    kmeans = fit_kmeans(X_cluster, k=KMEANS_K)
    
    print("Saving KMeans model artifact...")
    save_artifact(kmeans, KMEANS_MODEL_FILE)
    
    df["Cluster"] = kmeans.labels_
    
    print("Learning risk mapping...")
    # Need to inverse scale to evaluate thresholds properly
    pipeline = joblib.load(CLUSTER_PIPELINE_FILE)
    df_original = inverse_scale(pipeline, df.drop(columns=["Cluster"]))
    df_original["Cluster"] = df["Cluster"]
    
    cluster_profiles = df_original.groupby("Cluster").mean()
    mapping = learn_risk_mapping(cluster_profiles)
    
    print(f"Mapping learned: {mapping}")
    save_risk_mapping(mapping, RISK_MAPPING_FILE)
    
    df = add_risk_category(df, mapping)
    df.to_csv(LABELED_DATA_FILE, index=False)
    print("Clustering and Risk Mapping completed.")

# Task 3 ==> Classification & MLflow Registration
def run_classification_and_mlflow():
    from diasense.mlflow_tracking import log_clustering_run, log_classification_run
    
    # We use the existing MLflow tracking script which handles everything:
    # - Logs the clustering metrics/params
    # - Trains all classification models (KNN, SVM, LogReg, RF)
    # - Evaluates them
    # - Logs them to MLflow
    # - Registers the BEST one to the Model Registry "Production" stage
    print("Starting MLflow Tracking and Classification Workflow...")
    log_clustering_run()
    log_classification_run()
    print("Classification and MLflow Registration completed.")


# DAG Definition
default_args = {
    "owner": "DiaSense",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    dag_id="diasense_retraining_pipeline",
    default_args=default_args,
    description="End-to-end retraining pipeline for DiaSense",
    schedule_interval="@weekly",
    start_date=datetime(2026, 1, 1),
    catchup=False,
    tags=["diasense", "mlflow", "retraining"],
) as dag:
    
    task_preprocess = PythonOperator(
        task_id="preprocess_data",
        python_callable=run_preprocessing,
    )
    
    task_cluster = PythonOperator(
        task_id="cluster_and_map_risk",
        python_callable=run_clustering_and_mapping,
    )
    
    task_classify = PythonOperator(
        task_id="classify_and_track_mlflow",
        python_callable=run_classification_and_mlflow,
    )
    
    # Task dependencies (Linear Pipeline)
    task_preprocess >> task_cluster >> task_classify
