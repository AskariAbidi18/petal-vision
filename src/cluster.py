import os
import shutil
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

FEATURES_PATH = "results/features_final.csv"
FEATURES_WITH_ID_PATH = "results/features_final_with_id.csv"

PLOTS_DIR = "outputs/plots"
CLUSTERS_DIR = "outputs/clusters"
SEGMENTED_DIR = "data/segmented"

os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(CLUSTERS_DIR, exist_ok=True)

def run_clustering():
    print("=== CLUSTERING PIPELINE STARTED ===\n")

    # ---------- Load data ----------
    print("Loading feature data...")
    X = pd.read_csv(FEATURES_PATH)
    df = pd.read_csv(FEATURES_WITH_ID_PATH)
    print(f"Loaded {len(X)} samples.\n")

    # ---------- Elbow Method ----------
    print("Running elbow method...")
    inertias = []
    K = range(2, 9)

    for idx, k in enumerate(K):
        print(f"  Elbow step {idx + 1}/{len(K)} -> k = {k}")
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    plt.figure()
    plt.plot(K, inertias, marker="o")
    plt.xlabel("Number of clusters (k)")
    plt.ylabel("Inertia")
    plt.title("Elbow Method")
    plt.savefig(f"{PLOTS_DIR}/elbow.png")
    plt.close()

    print("Elbow plot saved.\n")

    # ---------- Choose k ----------
    k = 4
    print(f"Running K-Means clustering with k = {k}...")

    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X)

    df["cluster"] = labels
    df.to_csv("results/clustered_features.csv", index=False)

    print("K-Means clustering complete.")
    print("Clustered features saved.\n")

    # ---------- PCA ----------
    print("Running PCA for visualization...")
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)

    plt.figure()
    scatter = plt.scatter(
        X_pca[:, 0],
        X_pca[:, 1],
        c=labels,
        cmap="tab10",
        s=20
    )
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("PCA Visualization of Flower Clusters")
    plt.colorbar(scatter)
    plt.savefig(f"{PLOTS_DIR}/pca_clusters.png")
    plt.close()

    print("PCA plot saved.\n")

    # ---------- Cluster folders ----------
    print("Creating cluster folders...")
    for i in range(k):
        os.makedirs(f"{CLUSTERS_DIR}/cluster_{i}", exist_ok=True)

    print("Copying images into cluster folders...")
    total_imgs = len(df)

    for idx, row in df.iterrows():
        print(f"  Copying image {idx + 1}/{total_imgs}", end="\r")

        img_name = row["image"]
        cluster_id = row["cluster"]

        src = os.path.join(SEGMENTED_DIR, img_name)
        dst = os.path.join(CLUSTERS_DIR, f"cluster_{cluster_id}", img_name)

        if os.path.exists(src):
            shutil.copy(src, dst)

    print("\nImage copying complete.\n")

    print("=== CLUSTERING PIPELINE FINISHED ===")

if __name__ == "__main__":
    run_clustering()
