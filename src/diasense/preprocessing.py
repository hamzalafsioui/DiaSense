from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diasense.config import CLIP_Z_THRESHOLD, KNN_NEIGHBORS, ZERO_AS_MISSING


class HiddenZeroToNaN(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        if not isinstance(X, pd.DataFrame):
            raise TypeError(f"{self.__class__.__name__} expects a DataFrame.")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.columns:
            if col in X.columns:
                X[col] = X[col].replace(0, np.nan)
        return X


class FrameStandardScaler(BaseEstimator, TransformerMixin):
   

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        self.scaler_ = StandardScaler().fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        array = self.scaler_.transform(X)
        return pd.DataFrame(array, columns=self.feature_names_in_, index=X.index)


class OutlierCapper(BaseEstimator, TransformerMixin):
    
    def __init__(self, z_threshold: float = 3.0):
        self.z_threshold = z_threshold

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        self.lower_bounds_ = X.mean() - self.z_threshold * X.std()
        self.upper_bounds_ = X.mean() + self.z_threshold * X.std()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col in self.feature_names_in_:
            X[col] = X[col].clip(
                lower=self.lower_bounds_[col], upper=self.upper_bounds_[col]
            )
        return X


class FrameKNNImputer(BaseEstimator, TransformerMixin):
    """KNNImputer that accepts and returns DataFrames."""

    def __init__(self, n_neighbors: int = 5):
        self.n_neighbors = n_neighbors

    def fit(self, X: pd.DataFrame, y=None):
        self.feature_names_in_ = list(X.columns)
        self.imputer_ = KNNImputer(n_neighbors=self.n_neighbors).fit(X, y)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        array = self.imputer_.transform(X)
        return pd.DataFrame(array, columns=self.feature_names_in_, index=X.index)


class ColumnSelector(BaseEstimator, TransformerMixin):
    
    def __init__(self, columns: list[str]):
        self.columns = columns

    def fit(self, X: pd.DataFrame, y=None):
        missing = [c for c in self.columns if c not in X.columns]
        if missing:
            raise ValueError(f"Columns not found in input data: {missing}")
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        return X[self.columns].copy()



def build_preprocessing_pipeline(
    n_neighbors: int = KNN_NEIGHBORS,
    z_threshold: float = CLIP_Z_THRESHOLD,
) -> Pipeline:
    
    return Pipeline(
        steps=[
            ("zero_to_nan", HiddenZeroToNaN(columns=ZERO_AS_MISSING)),
            ("scaler", FrameStandardScaler()),
            ("capper", OutlierCapper(z_threshold=z_threshold)),
            ("imputer", FrameKNNImputer(n_neighbors=n_neighbors)),
        ]
    )


def detect_outliers_iqr(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    
    counts = {}
    for col in columns:
        q1, q3 = df[col].quantile([0.25, 0.75])
        iqr = q3 - q1
        counts[col] = int(((df[col] < q1 - 1.5 * iqr) | (df[col] > q3 + 1.5 * iqr)).sum())
    return pd.Series(counts, name="iqr_outliers")


def inverse_scale(pipe: Pipeline, df_scaled: pd.DataFrame) -> pd.DataFrame:
   
    scaler = pipe.named_steps["scaler"].scaler_
    return pd.DataFrame(
        scaler.inverse_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index
    )


def save_artifact(obj: object, filepath: str | Path) -> None:
    
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, filepath)