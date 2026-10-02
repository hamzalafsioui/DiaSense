import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from diasense.config import RANDOM_STATE


def evaluate_k(
    X: pd.DataFrame,
    k_values=range(2, 11),
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    
    results = []

    for k in k_values:
        if k >= len(X):
            raise ValueError(f"k={k} must be smaller than the number of rows.")

        model = KMeans(
            n_clusters=k,
            n_init=20,
            random_state=random_state,
        )
        labels = model.fit_predict(X)

        results.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X, labels),
            }
        )

    return pd.DataFrame(results)


def fit_kmeans(
    X: pd.DataFrame,
    k: int,
    random_state: int = RANDOM_STATE,
) -> KMeans:
    
    model = KMeans(
        n_clusters=k,
        n_init=20,
        random_state=random_state,
    )
    model.fit(X)
    return model