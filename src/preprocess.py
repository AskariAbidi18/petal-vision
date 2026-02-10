import os
import cv2

INPUT_DIR = "data/standardized"
OUTPUT_DIR = "data/resized"

IMG_SIZE = (256, 256)

os.makedirs(OUTPUT_DIR, exist_ok=True)

def preprocess_images():
    input_files = sorted([
        f for f in os.listdir(INPUT_DIR)
        if f.lower().endswith(".jpg")
    ])

    existing_outputs = set(os.listdir(OUTPUT_DIR))
    processed = 0

    for filename in input_files:
        if filename in existing_outputs:
            continue  # already processed

        input_path = os.path.join(INPUT_DIR, filename)
        output_path = os.path.join(OUTPUT_DIR, filename)

        img = cv2.imread(input_path)
        if img is None:
            continue

        img = cv2.resize(img, IMG_SIZE)
        img = cv2.GaussianBlur(img, (5, 5), 0)
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        cv2.imwrite(output_path, img_hsv)
        processed += 1

    print(f"Preprocessing done. New images processed: {processed}")

if __name__ == "__main__":
    preprocess_images()
