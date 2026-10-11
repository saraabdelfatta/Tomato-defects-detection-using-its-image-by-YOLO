
# ============================================================
# FILE: 02_train_baseline.py
#
# PURPOSE:
# Train a baseline YOLO11 image-classification model to
# classify tomato images into Fresh and Defect categories.
#
# INPUT:
# The prepared dataset created by 01_prepare_data.py
#
# OUTPUT:
# Training logs, graphs, and saved model weights.
# ============================================================

from pathlib import Path
import sys

# YOLO is the model interface provided by Ultralytics.
from ultralytics import YOLO


# ------------------------------------------------------------
# STEP 1: CONFIGURATION
# ------------------------------------------------------------

# Location of the prepared dataset.
# This is the OUTPUT path used in our first Python file.
DATASET_PATH = Path(
    r"E:\AUC\Advanced machine learning\project\M1\datasets\tomato"
)

# Name of the pretrained YOLO11 classification model.
# The model will be downloaded if it is not already available.
MODEL_NAME = "yolo11n-cls.pt"

# Number of complete passes through the training images.
EPOCHS = 30

# Images are resized to 224 x 224 pixels for training.
IMAGE_SIZE = 224

# Number of images processed in one batch.
BATCH_SIZE = 32

# Stop early if validation does not improve for this many epochs.
PATIENCE = 10

# Fixed seed helps make training more reproducible.
SEED = 42

# Folder where training results will be saved.
RESULTS_FOLDER = "runs"

# Name for this particular training experiment.
RUN_NAME = "baseline"


# ------------------------------------------------------------
# STEP 2: CHECK THE DATASET
# ------------------------------------------------------------

def check_dataset():
    """
    Check that the expected dataset folders exist before training.
    """

    # The training and validation folders are required by training.
    required_folders = [
        DATASET_PATH / "train" / "Fresh",
        DATASET_PATH / "train" / "Defect",
        DATASET_PATH / "val" / "Fresh",
        DATASET_PATH / "val" / "Defect",
    ]

    # Stop early if any required folder is missing.
    for folder in required_folders:
        if not folder.is_dir():
            sys.exit(
                "\nERROR: Required dataset folder not found:\n"
                f"{folder}\n"
                "Run 01_prepare_data.py or check DATASET_PATH."
            )

    print("Dataset folder checks passed.")
    print(f"Dataset location: {DATASET_PATH}")


# ------------------------------------------------------------
# STEP 3: TRAIN THE BASELINE MODEL
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("TOMATO FRESH/DEFECT BASELINE TRAINING")
    print("=" * 60)

    # Make sure we can find the prepared images.
    check_dataset()

    # Load a pretrained YOLO11 classification model.
    # Its weights are downloaded automatically if needed.
    print("\nLoading pretrained model:", MODEL_NAME)
    model = YOLO(MODEL_NAME)

    # Fine-tune the model on our tomato dataset.
    print("\nStarting training...")

    model.train(
        data=str(DATASET_PATH),  # Dataset with train/val/test folders
        epochs=EPOCHS,            # Maximum number of training epochs
        imgsz=IMAGE_SIZE,         # Resize input images to 224 x 224
        batch=BATCH_SIZE,         # Images processed in each batch
        patience=PATIENCE,        # Early stopping patience
        seed=SEED,                # Reproducibility setting
        project=RESULTS_FOLDER,   # Parent folder for experiment results
        name=RUN_NAME,            # Name of this experiment
    )

    # Find the folder created by the training run.
    run_folder = Path(RESULTS_FOLDER) / RUN_NAME

    # Ultralytics normally saves the best weights here.
    best_weights = run_folder / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("TRAINING FINISHED")
    print("=" * 60)

    # Report where the trained model was saved.
    if best_weights.exists():
        print(f"Best model weights:\n{best_weights}")
        print("\nKeep this file. We will use it for evaluation.")
    else:
        print(
            "Training returned, but best.pt was not found at the "
            f"expected location:\n{best_weights}\n"
            "Check the training output for the actual save location."
        )

    print(f"\nTraining results folder:\n{run_folder}")


# Run main() when this file is executed directly.
if __name__ == "__main__":
    main()
