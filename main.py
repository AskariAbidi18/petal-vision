import sys
import os
import contextlib

# Add src/ to Python path
SRC_DIR = os.path.join(os.path.dirname(__file__), "src")
sys.path.append(SRC_DIR)

from standardize_images import standardize_images
from preprocess import preprocess_images
from segment import segment_images
from features import extract_features
from preprocess_features import preprocess_features
from cluster import run_clustering


@contextlib.contextmanager
def suppress_output():
    """Temporarily suppress stdout and stderr."""
    with open(os.devnull, "w") as devnull:
        old_stdout = sys.stdout
        old_stderr = sys.stderr
        sys.stdout = devnull
        sys.stderr = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr


def main():
    print("\n=== PETAL-VISION PIPELINE START ===\n")

    steps = [
        ("Standardizing images", standardize_images),
        ("Preprocessing images", preprocess_images),
        ("Segmenting flowers", segment_images),
        ("Extracting features", extract_features),
        ("Preprocessing features", preprocess_features),
        ("Clustering & visualization", run_clustering),
    ]

    total = len(steps)

    for idx, (name, func) in enumerate(steps, start=1):
        print(f"[{idx}/{total}] {name}...")

        with suppress_output():
            func()

        print("Done\n")

    print("=== PIPELINE COMPLETE ===")


if __name__ == "__main__":
    main()
