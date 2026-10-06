import json
from pathlib import Path

import pandas as pd

from diasense.config import CLINICAL_THRESHOLDS


def learn_risk_mapping(
    profiles: pd.DataFrame,
    thresholds: dict[str, float] | None = None,
) -> dict[int, str]:
   
    if thresholds is None:
        thresholds = CLINICAL_THRESHOLDS

    missing = set(thresholds) - set(profiles.columns)
    if missing:
        raise ValueError(f"Cluster profiles lack columns: {sorted(missing)}")

    high_risk_clusters = [
        int(cluster_id)
        for cluster_id, row in profiles.iterrows()
        if all(row[feature] > limit for feature, limit in thresholds.items())
    ]

    if len(high_risk_clusters) != 1:
        raise ValueError(
            "Expected exactly one cluster whose means exceed all three "
            f"risk thresholds; found {high_risk_clusters}. "
            "Inspect the cluster profiles before assigning risk labels."
        )

    high_risk_id = high_risk_clusters[0]

    return {
        int(cluster_id): (
            "High Risk" if int(cluster_id) == high_risk_id else "Low Risk"
        )
        for cluster_id in profiles.index
    }


def add_risk_category(
    df: pd.DataFrame,
    mapping: dict[int, str],
) -> pd.DataFrame:
    
    result = df.copy()
    result["risk_category"] = result["Cluster"].map(mapping)

    if result["risk_category"].isna().any():
        raise ValueError("Some cluster IDs are missing from the risk mapping.")

    return result


def save_risk_mapping(
    mapping: dict[int, str],
    path: str | Path,
) -> None:
    
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    path.write_text(
        json.dumps({str(key): value for key, value in mapping.items()}, indent=2),
        encoding="utf-8",
    )


def load_risk_mapping(path: str | Path) -> dict[int, str]:
    
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return {int(key): value for key, value in data.items()}