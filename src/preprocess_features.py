import pandas as pd
from sklearn.preprocessing import StandardScaler

INPUT_CSV = "results/features.csv"
OUTPUT_FINAL = "results/features_final.csv"
OUTPUT_WITH_ID = "results/features_final_with_id.csv"

def preprocess_features():
    print("Loading features...")
    df = pd.read_csv(INPUT_CSV)

    original_count = len(df)
    print(f"Total samples loaded: {original_count}")

    # 1. Remove invalid rows (segmentation failures)
    df = df[
        (df["area_ratio"] > 0) &
        (df["perimeter"] > 0) &
        (df["circularity"] > 0)
    ].copy()

    cleaned_count = len(df)
    print(f"Valid samples kept : {cleaned_count}")
    print(f"Samples removed   : {original_count - cleaned_count}")

    # 2. Separate image names
    image_names = df["image"]
    features = df.drop(columns=["image"])

    # 3. Standardize features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(
        features_scaled,
        columns=features.columns
    )

    # 4. Save outputs
    df_scaled.to_csv(OUTPUT_FINAL, index=False)

    df_with_id = df_scaled.copy()
    df_with_id.insert(0, "image", image_names.values)
    df_with_id.to_csv(OUTPUT_WITH_ID, index=False)

    print("\nPreprocessing complete.")
    print(f"Scaled dataset saved to: {OUTPUT_FINAL}")
    print(f"Scaled dataset with image names saved to: {OUTPUT_WITH_ID}")

if __name__ == "__main__":
    preprocess_features()
