import os
import cv2
import numpy as np
import pandas as pd
from skimage.feature import graycomatrix, graycoprops

INPUT_DIR = "data/segmented"
OUTPUT_CSV = "results/features.csv"

os.makedirs("results", exist_ok=True)

def extract_features():
    records = []

    files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".jpg")
    ])

    total = len(files)
    print(f"Starting feature extraction for {total} images...\n")

    for idx, filename in enumerate(files):
        # Progress indicator
        print(f"Processing {idx + 1}/{total}: {filename}", end="\r")

        path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        h, w = img.shape[:2]
        img_area = h * w

        # Mask: non-black pixels are flower
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, mask = cv2.threshold(gray, 1, 255, cv2.THRESH_BINARY)

        flower_area = cv2.countNonZero(mask)
        area_ratio = flower_area / img_area if img_area > 0 else 0

        # Contours
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if contours:
            largest = max(contours, key=cv2.contourArea)
            perimeter = cv2.arcLength(largest, True)
            circularity = (
                (4 * np.pi * flower_area) / (perimeter ** 2)
                if perimeter > 0 else 0
            )
        else:
            perimeter = 0
            circularity = 0

        # Color features (HSV)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        h_vals = hsv[:, :, 0][mask > 0]
        s_vals = hsv[:, :, 1][mask > 0]
        v_vals = hsv[:, :, 2][mask > 0]

        mean_h = np.mean(h_vals) if len(h_vals) else 0
        mean_s = np.mean(s_vals) if len(s_vals) else 0
        mean_v = np.mean(v_vals) if len(v_vals) else 0

        # Texture feature (GLCM contrast)
        gray_small = cv2.resize(gray, (128, 128))
        glcm = graycomatrix(
            gray_small,
            distances=[1],
            angles=[0],
            levels=256,
            symmetric=True,
            normed=True
        )
        contrast = graycoprops(glcm, "contrast")[0, 0]

        records.append([
            filename,
            mean_h,
            mean_s,
            mean_v,
            area_ratio,
            perimeter,
            circularity,
            contrast
        ])

        # Occasional newline so terminal doesn’t feel frozen
        if (idx + 1) % 25 == 0:
            print(f"\nProcessed {idx + 1}/{total} images...")

    columns = [
        "image",
        "mean_hue",
        "mean_saturation",
        "mean_value",
        "area_ratio",
        "perimeter",
        "circularity",
        "texture_contrast"
    ]

    df = pd.DataFrame(records, columns=columns)
    df.to_csv(OUTPUT_CSV, index=False)

    print(f"\n\nDone. Extracted features for {len(df)} images.")
    print(f"Saved to {OUTPUT_CSV}")

if __name__ == "__main__":
    extract_features()
