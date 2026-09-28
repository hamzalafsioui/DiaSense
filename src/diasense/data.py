import pandas as pd
from diasense.config import CLINICAL_FEATURES,RAW_DATA_FILE


def load_raw_data() -> pd.DataFrame:
    if not RAW_DATA_FILE.exists():
        raise FileNotFoundError(f"Raw dataset not found at: {RAW_DATA_FILE} | Should be into data/raw folder")

    df = pd.read_csv(RAW_DATA_FILE)
    unnamed_columns = [col for col in df.columns if col.startswith("Unnamed:")]
    if unnamed_columns:
        df = df.drop(columns=unnamed_columns)

    return df


def validate_columns(df:pd.DataFrame):
    missing = [col for col in CLINICAL_FEATURES if col not in df.columns]

    if missing:
        raise ValueError(f"the dataset is missing expected columns: {missing}")
