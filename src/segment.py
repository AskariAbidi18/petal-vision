import os
import cv2
import numpy as np

INPUT_DIR = "data/resized"        # already resized images
SEGMENTED_DIR = "data/segmented"
MASK_DIR = "outputs/masks"

os.makedirs(SEGMENTED_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

def segment_images():
    files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".jpg")
    ])

    total = len(files)
    print(f"Starting segmentation on {total} images...\n")

    for idx, filename in enumerate(files):
        print(f"Processing {idx + 1}/{total}", end="\n")

        path = os.path.join(INPUT_DIR, filename)
        img = cv2.imread(path)

        if img is None:
            continue

        h, w = img.shape[:2]

        # GrabCut mask
        mask = np.zeros((h, w), np.uint8)

        bg_model = np.zeros((1, 65), np.float64)
        fg_model = np.zeros((1, 65), np.float64)

        # Rectangle slightly inside the image
        rect = (5, 5, w - 10, h - 10)

        # GrabCut (1 iteration for speed)
        cv2.grabCut(
            img,
            mask,
            rect,
            bg_model,
            fg_model,
            1,
            cv2.GC_INIT_WITH_RECT
        )

        # Binary mask
        final_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
            255,
            0
        ).astype("uint8")

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, kernel)

        # Apply mask
        segmented = cv2.bitwise_and(img, img, mask=final_mask)

        # Save outputs
        cv2.imwrite(os.path.join(MASK_DIR, filename), final_mask)
        cv2.imwrite(os.path.join(SEGMENTED_DIR, filename), segmented)

    print(f"\n\nDone. Segmented {total} images.")

if __name__ == "__main__":
    segment_images()
