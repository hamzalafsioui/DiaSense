import pandas as pd
import numpy as np
from diasense.config import ZERO_AS_MISSING, CLINICAL_FEATURES, RANDOM_STATE


def replace_hiden_zeros(df:pd.DataFrame)->pd.DataFrame:
    df_out = df.copy()
    for col in ZERO_AS_MISSING:
        if col in df_out.columns:
            df_out[col] = df_out.replace(0,np.mean)

    return df_out