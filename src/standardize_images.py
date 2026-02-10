import os
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

RAW_DIR = "data/raw"
OUT_DIR = "data/standardized"

VALID_EXTENSIONS = (".jpg", ".jpeg", ".png", ".heic", ".heif")

def standardize_images():
    # ---- GUARD: run only once ----
    if os.path.exists(OUT_DIR) and len(os.listdir(OUT_DIR)) > 0:
        print("Standardized images already exist. Skipping standardization.")
        return

    os.makedirs(OUT_DIR, exist_ok=True)

    count = 1
    print("Running image standardization...")

    for filename in sorted(os.listdir(RAW_DIR)):
        if not filename.lower().endswith(VALID_EXTENSIONS):
            continue

        input_path = os.path.join(RAW_DIR, filename)

        try:
            img = Image.open(input_path).convert("RGB")
            new_name = f"flower_{count:03d}.jpg"
            img.save(os.path.join(OUT_DIR, new_name), "JPEG", quality=95)
            count += 1
        except Exception:
            continue

    print(f"Standardization complete. {count - 1} images saved.")
