import os
import cv2
import numpy as np

INPUT_DIR = "data/resized"
SEGMENTED_DIR = "data/segmented"
MASK_DIR = "outputs/masks"

os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def segment_images():
    input_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".jpg")
    ])

    existing_segmented = set(os.listdir(SEGMENTED_DIR))
    processed = 0

    for filename in input_files:
        if filename in existing_segmented:
            continue  # already segmented

        path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(path)
        if img is None:
            continue

        h, w = img.shape[:2]
        mask = np.zeros((h, w), np.uint8)

        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        rect = (5, 5, w - 10, h - 10)

        cv2.grabCut(
            img, mask, rect,
            bg_model, fg_model,
            1, cv2.GC_INIT_WITH_RECT
        )

        final_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255, 0
        ).astype("uint8")

        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        segmented = cv2.bitwise_and(img, img, mask=final_mask)

        cv2.imwrite(os.path.join(MASK_DIR, filename), final_mask)
        cv2.imwrite(os.path.join(SEGMENTED_DIR, filename), segmented)

        processed += 1

    print(f"Segmentation done. New images processed: {processed}")

if __name__ == "__main__":
    segment_images()
