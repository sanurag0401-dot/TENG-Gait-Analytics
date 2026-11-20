
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA


# -----------------------------------------
# 1. LOAD FEATURE DATASET
# -----------------------------------------

file_path = Path(
    "data/processed/physionet/gait_features_S1_0.5.csv"
)

df = pd.read_csv(file_path)

print("Dataset loaded successfully!")

# Exclude potentially incomplete first contact
df = df[
    df["Recording Boundary Contact"].astype(str).str.lower()
    == "false"
].copy()


# -----------------------------------------
# 2. SELECT FEATURES
# -----------------------------------------

# Force impulse is excluded because it is
# strongly correlated with stance duration.
# This avoids giving contact duration
# excessive influence in clustering.

features = [
    "Stance Duration (s)",
    "Peak Force (N)",
    "Mean Force (N)",
    "Force Std (N)"
]

X = df[features].replace(
    [np.inf, -np.inf],
    np.nan
).dropna()

print("\nNumber of observations:", len(X))

print("\nFeatures used:")
print(features)


# -----------------------------------------
# 3. STANDARDISE FEATURES
# -----------------------------------------

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeature standardisation completed!")


# -----------------------------------------
# 4. FIND SUITABLE NUMBER OF CLUSTERS
# -----------------------------------------

silhouette_results = []

print("\n--- SILHOUETTE SCORE ANALYSIS ---")

for k in range(2, 5):

    kmeans = KMeans(
        n_clusters=k,
        random_state=42,
        n_init=20
    )

    labels = kmeans.fit_predict(X_scaled)

    score = silhouette_score(
        X_scaled,
        labels
    )

    silhouette_results.append({
        "Number of Clusters": k,
        "Silhouette Score": score
    })

    print(
        f"Clusters: {k} | Silhouette Score: {score:.4f}"
    )


# -----------------------------------------
# 5. SELECT BEST CLUSTERING MODEL
# -----------------------------------------

results_df = pd.DataFrame(silhouette_results)

best_k = int(
    results_df.loc[
        results_df["Silhouette Score"].idxmax(),
        "Number of Clusters"
    ]
)

print("\nSelected number of clusters:", best_k)


# -----------------------------------------
# 6. TRAIN FINAL K-MEANS MODEL
# -----------------------------------------

final_model = KMeans(
    n_clusters=best_k,
    random_state=42,
    n_init=20
)

cluster_labels = final_model.fit_predict(X_scaled)

print("\nK-Means clustering completed!")


# -----------------------------------------
# 7. SAVE CLUSTER RESULTS
# -----------------------------------------

clustered_data = X.copy()

clustered_data["Cluster"] = cluster_labels

output_folder = Path(
    "data/processed/physionet"
)

output_folder.mkdir(
    parents=True,
    exist_ok=True
)

clustered_data.to_csv(
    output_folder / "gait_clusters.csv",
    index=False
)

results_df.to_csv(
    output_folder / "silhouette_analysis.csv",
    index=False
)

print("\n--- CLUSTER DISTRIBUTION ---")

print(
    clustered_data["Cluster"].value_counts().sort_index()
)


# -----------------------------------------
# 8. PCA DIMENSIONALITY REDUCTION
# -----------------------------------------

pca = PCA(n_components=2)

X_pca = pca.fit_transform(X_scaled)

print(
    "\nPCA explained variance:",
    pca.explained_variance_ratio_
)

print(
    "Total variance represented:",
    pca.explained_variance_ratio_.sum()
)


# -----------------------------------------
# 9. VISUALISE CLUSTERS
# -----------------------------------------

plt.figure(figsize=(10, 7))

sns.scatterplot(
    x=X_pca[:, 0],
    y=X_pca[:, 1],
    hue=cluster_labels,
    palette="viridis",
    s=100
)

plt.title("K-Means Clustering of Gait Features")

plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")

plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    output_folder / "kmeans_gait_clusters.png",
    dpi=300
)

plt.show()


# -----------------------------------------
# 10. CLUSTER FEATURE SUMMARY
# -----------------------------------------

cluster_summary = clustered_data.groupby(
    "Cluster"
)[features].mean()

print("\n--- AVERAGE FEATURES PER CLUSTER ---")

print(cluster_summary.round(3))

cluster_summary.to_csv(
    output_folder / "cluster_feature_summary.csv"
)

print("\nMachine learning analysis completed!")
