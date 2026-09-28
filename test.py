from diasense.config import CLINICAL_FEATURES
from diasense.data import load_raw_data,validate_columns

df = load_raw_data()
validate_columns(df)

print(df.shape)
print(list(df.columns))
print("total missing: ",int(df.isna().sum().sum()))
print("Duplicated rows: ",int(df.duplicated().sum()))


print(df.head())