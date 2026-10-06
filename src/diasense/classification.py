from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
from imblearn.over_sampling import RandomOverSampler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from diasense.config import (
    CLINICAL_FEATURES,
    MODELS_DIR,
    PROCESSED_DATA_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    TEST_SIZE,
)


LABELED_DATA_PATH = PROCESSED_DATA_DIR / "dataset_labeled.csv"
PIPELINE_SAVE_PATH = MODELS_DIR / "classification_pipeline.joblib"

# Hyperparameter grids | keys must be prefixed with 'classifier__'
PARAM_GRIDS: dict[str, dict[str, list[Any]]] = {
    "KNN": {
        "classifier__n_neighbors": [3, 5, 7, 9, 11],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    },
    "SVM": {
        "classifier__C": [0.1, 1, 10],
        "classifier__kernel": ["rbf", "linear"],
        "classifier__gamma": ["scale", "auto"],
    },
    "Logistic Regression": {
        "classifier__C": [0.01, 0.1, 1, 10],
        "classifier__solver": ["lbfgs", "liblinear"],
        "classifier__max_iter": [500],
    },
    "Random Forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [None, 5, 10],
        "classifier__min_samples_split": [2, 5],
    },
}

# Base estimators | one instance per model name
BASE_ESTIMATORS: dict[str, Any] = {
    "KNN": KNeighborsClassifier(),
    "SVM": SVC(probability=True, random_state=RANDOM_STATE),
    "Logistic Regression": LogisticRegression(random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
}


# Load labeled data

def load_labeled_data(path: Path = LABELED_DATA_PATH) -> pd.DataFrame:
    
    if not path.exists():
        raise FileNotFoundError(
            f"Labeled dataset not found at: {path}\n"
            "Please run the clustering notebook first."
        )
    df = pd.read_csv(path)
    print(f"Loaded {len(df)} rows, {df.shape[1]} columns")
    return df


# Train / test split


def split_data(
    df: pd.DataFrame,
    features: list[str] = CLINICAL_FEATURES,
    target: str = TARGET_COLUMN,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )

    print(f"Train size : {len(X_train)} samples")
    print(f"Test  size : {len(X_test)} samples")
    print(f"Class distribution in train:\n{y_train.value_counts()}\n")

    return X_train, X_test, y_train, y_test


# Class balancing

def balance_classes(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.Series]:
    
    ros = RandomOverSampler(random_state=random_state)
    X_bal, y_bal = ros.fit_resample(X_train, y_train)

    X_bal = pd.DataFrame(X_bal, columns=X_train.columns)
    y_bal = pd.Series(y_bal, name=y_train.name)

    print("Class distribution after oversampling:")
    print(y_bal.value_counts())

    return X_bal, y_bal


# Build a pipeline for one model


def build_model_pipeline(estimator: Any) -> Pipeline:
    
    return Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ]
    )


# Hyperparameter tuning


def tune_model(
    pipeline: Pipeline,
    param_grid: dict[str, list[Any]],
    X_train: pd.DataFrame,
    y_train: pd.Series,
    cv: int = 5,
    scoring: str = "f1_weighted",
) -> GridSearchCV:
   
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,   
        refit=True,
        verbose=0,
    )
    grid_search.fit(X_train, y_train)

    print(f"  Best params : {grid_search.best_params_}")
    print(f"  Best CV {scoring} : {grid_search.best_score_:.4f}")

    return grid_search


# Evaluation


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "",
) -> dict[str, Any]:
   
    y_pred = model.predict(X_test)

    report = classification_report(y_test, y_pred, zero_division=0)
    cm = confusion_matrix(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    accuracy = float((y_pred == y_test.values).mean())

    if model_name:
        separator = "=" * 50
        print(f"\n{separator}")
        print(f"  Results for: {model_name}")
        print(separator)
        print(f"  Accuracy : {accuracy:.4f}")
        print(f"  F1-Score : {f1:.4f}")
        print("\nClassification Report:")
        print(report)
        print("Confusion Matrix:")
        print(cm)

    return {
        "model_name": model_name,
        "accuracy": accuracy,
        "f1_weighted": f1,
        "classification_report": report,
        "confusion_matrix": cm,
        "y_pred": y_pred,
    }


# Step 7 – Train and compare all models


def train_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[pd.DataFrame, dict[str, GridSearchCV]]:

   
    rows: list[dict[str, Any]] = []
    fitted_models: dict[str, GridSearchCV] = {}

    for name, estimator in BASE_ESTIMATORS.items():
        print(f"\n{'─' * 40}")
        print(f"  Training : {name}")
        print(f"{'─' * 40}")

        pipeline = build_model_pipeline(estimator)
        grid_search = tune_model(pipeline, PARAM_GRIDS[name], X_train, y_train)
        fitted_models[name] = grid_search

        metrics = evaluate_model(grid_search.best_estimator_, X_test, y_test, name)

        rows.append(
            {
                "Model": name,
                "Accuracy": round(metrics["accuracy"], 4),
                "F1-Score": round(metrics["f1_weighted"], 4),
            }
        )

    results_df = (
        pd.DataFrame(rows)
        .sort_values("F1-Score", ascending=False)
        .reset_index(drop=True)
    )

    print("\n\n===== MODEL COMPARISON =====")
    print(results_df.to_string(index=False))

    return results_df, fitted_models


# Select best model and save


def get_best_model(
    results_df: pd.DataFrame,
    fitted_models: dict[str, GridSearchCV],
) -> tuple[str, Pipeline]:

    best_name = results_df.iloc[0]["Model"]
    best_pipeline = fitted_models[best_name].best_estimator_
    print(f"\n>>> Best model: {best_name} (F1={results_df.iloc[0]['F1-Score']:.4f})")
    return best_name, best_pipeline


def save_pipeline(
    pipeline: Pipeline,
    path: Path = PIPELINE_SAVE_PATH,
) -> Path:
   
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, path)
    print(f"Pipeline saved ===> {path}")
    return path


def load_pipeline(path: Path = PIPELINE_SAVE_PATH) -> Pipeline:
    
    return joblib.load(path)


# Convenience run the full Step 5 workflow in one call


def run_classification_workflow(
    features: list[str] = CLINICAL_FEATURES,
    save: bool = True,
) -> dict[str, Any]:
   
    print("=" * 60)
    print("  Classification Workflow")
    print("=" * 60)

    # Load
    df = load_labeled_data()

    # Split
    X_train, X_test, y_train, y_test = split_data(df, features=features)

    # Balance training set only
    X_train_bal, y_train_bal = balance_classes(X_train, y_train)

    # Train + tune all models
    results_df, fitted_models = train_all_models(
        X_train_bal, y_train_bal, X_test, y_test
    )

    # Pick the best
    best_name, best_pipeline = get_best_model(results_df, fitted_models)

    # Full evaluation metrics for the best model
    best_metrics = evaluate_model(best_pipeline, X_test, y_test, best_name)

    # Save
    saved_path = save_pipeline(best_pipeline) if save else None

    print("\n" + "=" * 60)
    print("  Workflow complete.")
    print("=" * 60)

    return {
        "results_df": results_df,
        "fitted_models": fitted_models,
        "best_name": best_name,
        "best_pipeline": best_pipeline,
        "best_metrics": best_metrics,
        "saved_path": saved_path,
        "X_test": X_test,
        "y_test": y_test,
    }
