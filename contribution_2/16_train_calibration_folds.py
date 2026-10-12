"""
FILE: 16_train_calibration_folds.py

PURPOSE:
    Train K fold models so that every training image gets a prediction
    from a model that never saw it. Those out-of-fold predictions give a
    large, unbiased calibration set, and the fold models form an ensemble.

INPUT:
    contribution_2/datasets/tomato/train/{Defect,Fresh}
    contribution_2/datasets/tomato/val/{Defect,Fresh}

OUTPUT:
    contribution_2/datasets/tomato_folds/fold_assignment.csv
    contribution_2/datasets/tomato_folds/fold_<k>/{train,val}/...
    contribution_2/runs/calibration_folds/fold_<k>/weights/best.pt

IMPORTANT:
    - This trains NEW models. The baseline weights and results under
      runs/ are not read for training and are never modified.
    - Settings match 02_train_baseline.py (224 px, batch 32, 30 epochs,
      patience 10, seed 42); only the training images differ.
    - Each fold model stops early on the ORIGINAL validation split, so
      its held-out fold plays no part in training or model selection.
    - The test split is not used here.
    - Folds that already have best.pt are skipped, so the script can be
      re-run after an interruption.
"""

from pathlib import Path
import csv
import random
import shutil

from ultralytics import YOLO

# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

# This script lives in contribution_2/, one level below the project.
SESSION_PATH = Path(__file__).resolve().parent
PROJECT_PATH = SESSION_PATH.parent

DATASET_PATH = SESSION_PATH / "datasets" / "tomato"
FOLDS_PATH = SESSION_PATH / "datasets" / "tomato_folds"
RESULTS_FOLDER = SESSION_PATH / "runs" / "calibration_folds"

MODEL_NAME = str(PROJECT_PATH / "yolo11n-cls.pt")
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("Defect", "Fresh")

FOLDS = 5

# Same training settings as 02_train_baseline.py.
EPOCHS = 30
IMAGE_SIZE = 224
BATCH_SIZE = 32
PATIENCE = 10
SEED = 42


# ------------------------------------------------------------
# 2. ASSIGN TRAINING IMAGES TO FOLDS
# ------------------------------------------------------------

def find_images(folder):
    if not folder.is_dir():
        raise FileNotFoundError(f"Dataset folder not found:\n{folder}")
    return sorted(
        path for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def assign_folds():
    """Give every training image a fold, stratified by class."""
    rng = random.Random(SEED)
    assignment = []
    for class_name in CLASS_NAMES:
        images = find_images(DATASET_PATH / "train" / class_name)
        rng.shuffle(images)
        for index, image in enumerate(images):
            assignment.append((class_name, image, index % FOLDS))
    return assignment


def build_fold_datasets(assignment):
    """Copy images so each fold folder is a normal train/val dataset."""
    FOLDS_PATH.mkdir(parents=True, exist_ok=True)

    with (FOLDS_PATH / "fold_assignment.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["class", "image", "held_out_fold"])
        for class_name, image, fold in sorted(
            assignment, key=lambda row: (row[0], row[1].name)
        ):
            writer.writerow([class_name, image.name, fold])

    for fold in range(FOLDS):
        fold_path = FOLDS_PATH / f"fold_{fold}"
        if (fold_path / "complete.txt").is_file():
            continue
        if fold_path.exists():
            shutil.rmtree(fold_path)

        for class_name in CLASS_NAMES:
            train_folder = fold_path / "train" / class_name
            val_folder = fold_path / "val" / class_name
            train_folder.mkdir(parents=True)
            val_folder.mkdir(parents=True)

            # Train on every training image outside this fold.
            for image_class, image, image_fold in assignment:
                if image_class == class_name and image_fold != fold:
                    shutil.copy2(image, train_folder / image.name)

            # Early stopping uses the original validation split.
            for image in find_images(DATASET_PATH / "val" / class_name):
                shutil.copy2(image, val_folder / image.name)

        (fold_path / "complete.txt").write_text("ok\n", encoding="utf-8")
        print(f"Prepared {fold_path.relative_to(PROJECT_PATH)}")


# ------------------------------------------------------------
# 3. TRAIN ONE MODEL PER FOLD
# ------------------------------------------------------------

def main():
    assignment = assign_folds()
    print(f"Training images: {len(assignment)} in {FOLDS} folds")
    build_fold_datasets(assignment)

    for fold in range(FOLDS):
        run_folder = RESULTS_FOLDER / f"fold_{fold}"
        best_weights = run_folder / "weights" / "best.pt"

        if best_weights.is_file():
            print(f"\nFold {fold}: already trained, skipping.")
            continue
        if run_folder.exists():
            # Remove an interrupted run so the folder name stays fixed.
            shutil.rmtree(run_folder)

        print(f"\n{'=' * 60}\nTRAINING FOLD {fold + 1} OF {FOLDS}\n{'=' * 60}")
        model = YOLO(MODEL_NAME)
        model.train(
            data=str(FOLDS_PATH / f"fold_{fold}"),
            epochs=EPOCHS,
            imgsz=IMAGE_SIZE,
            batch=BATCH_SIZE,
            patience=PATIENCE,
            seed=SEED,
            project=str(RESULTS_FOLDER),
            name=f"fold_{fold}",
        )

        if not best_weights.is_file():
            raise RuntimeError(
                f"Fold {fold} finished without best.pt at:\n{best_weights}"
            )

    print(f"\nAll fold models are in:\n{RESULTS_FOLDER}")


if __name__ == "__main__":
    main()
