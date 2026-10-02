import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score


def compare_feature_sets(X_prepared, feature_sets, k_values=range(1, 11), random_state=42, n_init=20):
    rows = []
    for name, cols in feature_sets.items():
        X_sub = X_prepared[list(cols)].dropna()
        for k in k_values:
            if not 2 <= k < len(X_sub):
                continue
            km = KMeans(n_clusters=k, n_init=n_init, random_state=random_state)
            labels = km.fit_predict(X_sub)
            sil = float(silhouette_score(X_sub, labels)) if len(np.unique(labels)) > 1 else np.nan
            rows.append({"feature_set": name, "k": k, "inertia": float(km.inertia_), "silhouette": sil})
    return pd.DataFrame(rows)


def plot_feature_set_curves(results):
    fig = plt.figure(figsize=(16, 6))
    for idx, metric in enumerate(["inertia", "silhouette"], 1):
        plt.subplot(1, 2, idx)
        for name, group in results.groupby("feature_set"):
            group = group.sort_values("k")
            plt.plot(group["k"], group[metric], marker="o", label=name, linewidth=2)
        plt.title("Elbow" if metric == "inertia" else "Silhouette")
        plt.xlabel("k")
        plt.ylabel(metric)
        plt.legend()
        plt.grid(alpha=0.3)
    fig.tight_layout()
    return fig