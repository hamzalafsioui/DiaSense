import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib
from pathlib import Path

from diasense.config import ZERO_AS_MISSING, CLINICAL_FEATURES, RANDOM_STATE


def replace_hidden_zeros(df:pd.DataFrame)->pd.DataFrame:
    df_out = df.copy()
    for col in ZERO_AS_MISSING:
        if col in df_out.columns:
            df_out[col] = df_out[col].replace(0,np.nan)

    return df_out

def impute_knn(df:pd.DataFrame,n_neighbors:int = 5) ->tuple[pd.DataFrame,KNNImputer]:
    """
        Take my DataFrame, use the 5 most similar observations to estimate missing values, convert the result back into a DataFrame, and give me both the completed data and the KNN imputer.
    """

    imputer = KNNImputer(n_neighbors=n_neighbors)

    # KNNImputer returns a numpy array so we rebuild the dataframe
    impute_array = imputer.fit_transform(df)
    df_out = pd.DataFrame(impute_array,columns=df.columns,index=df.index)

    return df_out, imputer

def detect_outliers_iqr(df:pd.DataFrame,columns:list[str])->pd.DataFrame:

    outlier_mask = pd.DataFrame(False,index=df.index,columns=columns)

    for col in columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 -Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        outlier_mask[col] = (df[col] < lower_bound) | (df[col] > upper_bound)


    return outlier_mask

def cap_outliers_iqr(df:pd.DataFrame,columns:list[str])->pd.DataFrame:

    df_out = df.copy()

    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 -Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df_out[col] = np.clip(df_out[col],lower_bound,upper_bound)

    return df_out

def scale_data(df:pd.DataFrame,columns:list[str],method:str = "standard")->tuple[pd.DataFrame,object]:
    df_out = df.copy()

    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    else:
        raise ValueError("method must be standard | minmax")

    df_out[columns] = scaler.fit_transform(df_out[columns])

    return df_out,scaler




def save_artifact(object_to_save: object, filepath: str | Path):
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True,exist_ok=True)
    joblib.dump(object_to_save,filepath)
