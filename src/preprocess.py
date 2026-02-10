import os
import cv2

INPUT_DIR = "data/standardized"
OUTPUT_DIR = "data/resized"

os.makedirs(OUTPUT_DIR, exist_ok=True)

IMG_SIZE = (256, 256)

def preprocess_images():
    processed = 0

    for filename in sorted(os.listdir(INPUT_DIR)):
        if not filename.lower().endswith(".jpg"):
            continue

        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        img = cv2.imread(input_path)

        if img is None:
            print(f"[SKIP] Could not read {filename}")
            continue

        # Resize
        img = cv2.resize(img, IMG_SIZE)

        # Noise reduction
        img = cv2.GaussianBlur(img, (5, 5), 0)

        # Convert BGR -> HSV
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        cv2.imwrite(output_path, img_hsv)
        processed += 1

    print(f"\nDone. Preprocessed {processed} images.")

if __name__ == "__main__":
    preprocess_images()
