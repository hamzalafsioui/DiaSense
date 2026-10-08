import os

import joblib
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import numpy
import pandas
import sklearn
from mlflow.client import MlflowClient
from mlflow.models.signature import infer_signature
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    precision_score,
    recall_score,
    silhouette_score,
)

from diasense.classification import evaluate_model, run_classification_workflow
from diasense.config import (
    CLINICAL_FEATURES,
    CLUSTER_PIPELINE_FILE,
    CLUSTERING_FEATURES,
    KMEANS_K,
    KMEANS_MODEL_FILE,
    MLFLOW_EXPERIMENT_NAME,
    MLFLOW_MODEL_ALIAS,
    MLFLOW_REGISTERED_MODEL_NAME,
    MLFLOW_TRACKING_URI,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    REPORTS_DIR,
)

# MLflow Setup | connect once at module load
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)


# Helper ===> save confusion matrix and log it as an artifact
def _log_confusion_matrix(y_test, y_pred, model_name: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax, cmap="Blues")
    ax.set_title(f"Confusion Matrix | {model_name}")

    # Save to reports/figures/
    safe_name = model_name.replace(" ", "_")
    fig_path = REPORTS_DIR / "figures" / f"confusion_matrix_{safe_name}.png"
    fig_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fig_path, bbox_inches="tight")
    plt.close(fig)

    mlflow.log_artifact(str(fig_path), artifact_path="plots")


# Run 1 | Clustering
def log_clustering_run():
    print("\n========== Logging Clustering Run ==========")

    if not KMEANS_MODEL_FILE.exists() or not CLUSTER_PIPELINE_FILE.exists():
        print("  Clustering models not found ==> run the clustering notebook first")
        return

    df = pandas.read_csv(PROCESSED_DATA_DIR / "dataset_preprocessed.csv")
    X_cluster = df[CLUSTERING_FEATURES]

    kmeans = joblib.load(KMEANS_MODEL_FILE)
    inertia = float(kmeans.inertia_)
    sil = float(silhouette_score(X_cluster, kmeans.labels_))

    with mlflow.start_run(run_name="Clustering_KMeans"):
        # Parameters
        mlflow.log_params({
            "k":           KMEANS_K,
            "n_init":      kmeans.n_init,
            "random_state": kmeans.random_state,
            "features":    CLUSTERING_FEATURES,
        })

        # Metrics
        mlflow.log_metrics({
            "inertia":         inertia,
            "silhouette_score": sil,
        })

        # Artifacts | the saved model files
        mlflow.log_artifact(str(KMEANS_MODEL_FILE),     artifact_path="models")
        mlflow.log_artifact(str(CLUSTER_PIPELINE_FILE), artifact_path="models")

    print("  Clustering run logged successfully.")


# Run 2 | Classification (one MLflow run per model)
def log_classification_run():
    print("\n========== Logging Classification Run ==========")

    # Re-use the existing workflow: train + tune all 4 models
    results = run_classification_workflow(features=CLINICAL_FEATURES, save=False)

    best_name    = results["best_name"]
    fitted_models = results["fitted_models"]
    X_test       = results["X_test"]
    y_test       = results["y_test"]

    for model_name, grid_search in fitted_models.items():
        is_best = model_name == best_name
        pipeline = grid_search.best_estimator_

        # Get predictions + metrics for this specific model
        metrics = evaluate_model(pipeline, X_test, y_test, model_name="")
        y_pred  = metrics["y_pred"]

        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_test, y_pred, average="weighted", zero_division=0)

        run_name = f"Classification_{model_name}" + (" [BEST]" if is_best else "")

        with mlflow.start_run(run_name=run_name):
            # Tag so the UI makes it easy to filter
            mlflow.set_tags({
                "model_name":    model_name,
                "is_best_model": str(is_best),
            })

            # 1_Traceability | library versions + features
            mlflow.log_params({
                "features_used":   CLINICAL_FEATURES,
                "sklearn_version": sklearn.__version__,
                "pandas_version":  pandas.__version__,
                "numpy_version":   numpy.__version__,
            })

            # 2_ Best hyperparameters found by GridSearchCV
            classifier_params = pipeline.named_steps["classifier"].get_params()
            for key, value in classifier_params.items():
                mlflow.log_param(f"hp_{key}", value)

            # 3_ Performance metrics
            mlflow.log_metrics({
                "accuracy":  metrics["accuracy"],
                "f1_score":  metrics["f1_weighted"],
                "precision": precision,
                "recall":    recall,
            })

            # 4_ Confusion matrix as image artifact
            _log_confusion_matrix(y_test, y_pred, model_name)

            # 5_ Log model artifact with cloudpickle (avoids skops trusted-type error)
            signature = infer_signature(X_test, y_pred)

            if is_best:
                # Register ONLY the best model in the Model Registry
                print(f"  Registering '{model_name}' in the MLflow Model Registry...")
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="pipeline",
                    signature=signature,
                    input_example=X_test.iloc[[0]],
                    registered_model_name=MLFLOW_REGISTERED_MODEL_NAME,
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )

                # Transition the new version to Production
                client  = MlflowClient()
                versions = client.search_model_versions(
                    f"name='{MLFLOW_REGISTERED_MODEL_NAME}'"
                )
                if versions:
                    latest = sorted(versions, key=lambda v: int(v.version))[-1].version
                    client.transition_model_version_stage(
                        name=MLFLOW_REGISTERED_MODEL_NAME,
                        version=latest,
                        stage="Production",
                        archive_existing_versions=True,
                    )
                    print(f"  Version {latest} transitioned to Production.")
            else:
                # Log artifact only | do not register non-best models
                mlflow.sklearn.log_model(
                    sk_model=pipeline,
                    artifact_path="pipeline",
                    signature=signature,
                    input_example=X_test.iloc[[0]],
                    serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_CLOUDPICKLE,
                )

        print(f"  {'>>> [BEST]' if is_best else '  '} {model_name} logged.")

    print("\n  All classification runs logged successfully.")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 50)
    print("  DiaSense | MLflow Tracking")
    print(f"  Server : {MLFLOW_TRACKING_URI}")
    print(f"  Experiment : {MLFLOW_EXPERIMENT_NAME}")
    print("=" * 50)

    try:
        log_clustering_run()
        log_classification_run()
        print("\n>>> All tracking operations completed successfully >>>")
    except Exception as exc:
        print(f"\n[ERROR] {exc}")
        raise
