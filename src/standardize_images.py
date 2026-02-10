import os
from PIL import Image
import pillow_heif

# Enable HEIC/HEIF support
pillow_heif.register_heif_opener()

RAW_DIR = "data/raw"
OUT_DIR = "data/standardized"

os.makedirs(OUT_DIR, exist_ok=True)

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif")

def standardize_images():
    count = 1

    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        input_path = os.path.join(RAW_DIR, filename)

        try:
            img = Image.open(input_path).convert("RGB")

            new_name = f"flower_{count:03d}.jpg"
            output_path = os.path.join(OUT_DIR, new_name)

            img.save(output_path, "JPEG", quality=95)

            print(f"[OK] {filename} -> {new_name}")
            count += 1

        except Exception as e:
            print(f"[SKIP] {filename} | Error: {e}")

    print(f"\nDone. Total images processed: {count - 1}")

if __name__ == "__main__":
    standardize_images()
