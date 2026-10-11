
"""
FILE: 07_train_synthetic_experiment.py

PURPOSE:
    Train an experimental YOLO11n-cls classifier using the
    original training images plus synthetic multi-tomato scenes.

INPUT:
    datasets/tomato_experiment_v1/
        train/Fresh, train/Defect
        val/Fresh, val/Defect

OUTPUT:
    A separate training run under:
    project/runs/experiment_synthetic/

IMPORTANT:
    - Does not overwrite the baseline model.
    - Uses the original validation set.
    - The original test set is not used during training.
    - Synthetic scenes are experimental and may contain
      imperfect tomato segmentation.

NEXT STEP:
    Evaluate the trained experiment on the original test set
    and compare it with the baseline.
"""

from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent
DATASET_PATH = (
    PROJECT_PATH.parent / "datasets" / "tomato_experiment_v1"
)

# Start from the same pretrained model family as the baseline.
MODEL_NAME = "yolo11n-cls.pt"

# Separate output location to protect the baseline.
RUNS_DIRECTORY = PROJECT_PATH / "runs" / "experiment_synthetic"
RUN_NAME = "yolo11n_synthetic_pilot"

# ------------------------------------------------------------
# 2. TRAINING SETTINGS
# ------------------------------------------------------------

EPOCHS = 20
IMAGE_SIZE = 224
BATCH_SIZE = 16
PATIENCE = 5
RANDOM_SEED = 42

# ------------------------------------------------------------
# 3. VERIFY DATASET BEFORE TRAINING
# ------------------------------------------------------------

CLASSES = ["Fresh", "Defect"]


def count_images(folder):
    """Count supported images directly inside a class folder."""
    extensions = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    if not folder.exists():
        return 0

    return sum(
        1 for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in extensions
    )


def verify_dataset():
    """Check the training and validation folders."""
    print("=" * 60)
    print("VERIFYING EXPERIMENTAL DATASET")
    print("=" * 60)
    print(f"Dataset: {DATASET_PATH}")

    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}"
        )

    total_counts = {}

    for split in ["train", "val"]:
        total_counts[split] = 0
        print(f"\n{split.upper()}")

        for class_name in CLASSES:
            class_folder = DATASET_PATH / split / class_name

            if not class_folder.is_dir():
                raise FileNotFoundError(
                    f"Missing class folder: {class_folder}"
                )

            count = count_images(class_folder)

            if count == 0:
                raise RuntimeError(
                    f"No images found in: {class_folder}"
                )

            total_counts[split] += count
            print(f"  {class_name}: {count} images")

        print(f"  Total: {total_counts[split]} images")

    print("\nDataset verification passed.")
    return total_counts


# ------------------------------------------------------------
# 4. TRAIN EXPERIMENTAL MODEL
# ------------------------------------------------------------

def main():
    counts = verify_dataset()

    print("\n" + "=" * 60)
    print("STARTING SYNTHETIC-DATA TRAINING EXPERIMENT")
    print("=" * 60)
    print(f"Model: {MODEL_NAME}")
    print(f"Epochs: {EPOCHS}")
    print(f"Image size: {IMAGE_SIZE}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Output directory: {RUNS_DIRECTORY / RUN_NAME}")

    # Load the pretrained YOLO classification model.
    model = YOLO(MODEL_NAME)

    # Train using only train and validation folders.
    model.train(
        data=str(DATASET_PATH),
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch=BATCH_SIZE,
        patience=PATIENCE,
        seed=RANDOM_SEED,
        workers=0,
        project=str(RUNS_DIRECTORY),
        name=RUN_NAME,
        exist_ok=False,
        plots=True,
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("EXPERIMENTAL TRAINING FINISHED")
    print("=" * 60)
    print(
        "Check the run folder for best.pt, last.pt, "
        "training plots, and results.csv."
    )
    print(f"Run folder: {RUNS_DIRECTORY / RUN_NAME}")


if __name__ == "__main__":
    main()
