
from pathlib import Path
import shutil

# ============================================================
# 06 - PREPARE EXPERIMENTAL DATASET
# ============================================================
# Purpose:
#   Combine the original training images with our pilot
#   synthetic scenes, while preserving validation and test data.
#
# Important:
#   - Original dataset is never modified.
#   - Synthetic scenes are added to TRAIN only.
#   - Validation and test images remain original and unchanged.
#   - Unique filenames prevent accidental overwriting.
# ============================================================

PROJECT_PATH = Path(__file__).resolve().parent
DATASETS_PATH = PROJECT_PATH.parent / "datasets"

ORIGINAL_DATASET = DATASETS_PATH / "tomato"
SYNTHETIC_DATASET = (
    DATASETS_PATH / "synthetic_multi_scenes_pilot" / "train"
)

EXPERIMENT_DATASET = DATASETS_PATH / "tomato_experiment_v1"

CLASSES = ["Fresh", "Defect"]
SPLITS = ["train", "val", "test"]

IMAGE_EXTENSIONS = {
    ".jpg", ".jpeg", ".png", ".bmp", ".webp"
}


def get_images(folder):
    """Find supported image files in a folder."""
    if not folder.exists():
        return []

    return sorted(
        file for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


def copy_images(source_folder, destination_folder, prefix):
    """Copy images using unique filenames."""
    images = get_images(source_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    for index, source_file in enumerate(images, start=1):
        destination_name = (
            f"{prefix}_{index:05d}{source_file.suffix.lower()}"
        )
        destination_file = destination_folder / destination_name

        shutil.copy2(source_file, destination_file)

    return len(images)


def main():
    print("=" * 60)
    print("PREPARING EXPERIMENTAL CLASSIFICATION DATASET")
    print("=" * 60)

    # Safety check: never overwrite an existing experiment.
    if EXPERIMENT_DATASET.exists():
        raise FileExistsError(
            f"\nExperiment folder already exists:\n"
            f"{EXPERIMENT_DATASET}\n\n"
            "Nothing was changed. Rename or remove the existing "
            "experiment folder only if you intentionally want "
            "to rebuild it."
        )

    # Verify the expected original dataset structure.
    for split in SPLITS:
        for class_name in CLASSES:
            folder = ORIGINAL_DATASET / split / class_name

            if not folder.is_dir():
                raise FileNotFoundError(
                    f"Expected dataset folder not found:\n{folder}"
                )

    # Verify the synthetic scene folders.
    for class_name in CLASSES:
        folder = SYNTHETIC_DATASET / class_name

        if not folder.is_dir():
            raise FileNotFoundError(
                f"Synthetic scene folder not found:\n{folder}"
            )

    counts = {}

    try:
        # ----------------------------------------------------
        # 1. Copy original images into the separate dataset.
        # ----------------------------------------------------
        for split in SPLITS:
            counts[split] = {}

            for class_name in CLASSES:
                source = (
                    ORIGINAL_DATASET / split / class_name
                )
                destination = (
                    EXPERIMENT_DATASET / split / class_name
                )

                copied = copy_images(
                    source,
                    destination,
                    prefix=f"original_{split}_{class_name}",
                )

                counts[split][class_name] = {
                    "original": copied,
                    "synthetic": 0,
                }

                if copied == 0:
                    raise RuntimeError(
                        f"No images found in {source}"
                    )

        # ----------------------------------------------------
        # 2. Add synthetic images to TRAIN only.
        # ----------------------------------------------------
        for class_name in CLASSES:
            source = SYNTHETIC_DATASET / class_name
            destination = (
                EXPERIMENT_DATASET / "train" / class_name
            )

            copied = copy_images(
                source,
                destination,
                prefix=f"synthetic_{class_name}",
            )

            counts["train"][class_name]["synthetic"] = copied

            if copied == 0:
                raise RuntimeError(
                    f"No synthetic images found in {source}"
                )

    except Exception:
        # Remove an incomplete experimental dataset if copying fails.
        if EXPERIMENT_DATASET.exists():
            shutil.rmtree(EXPERIMENT_DATASET)
        raise

    # --------------------------------------------------------
    # 3. Print the final dataset summary.
    # --------------------------------------------------------
    print("\nDataset created successfully:")
    print(EXPERIMENT_DATASET)

    print("\nIMAGE COUNTS")
    print("-" * 60)

    for split in SPLITS:
        print(f"\n{split.upper()}")

        for class_name in CLASSES:
            original_count = counts[split][class_name]["original"]
            synthetic_count = counts[split][class_name]["synthetic"]

            print(
                f"  {class_name}: "
                f"{original_count} original"
                f" + {synthetic_count} synthetic"
                f" = {original_count + synthetic_count}"
            )

    print("\nIMPORTANT:")
    print("- Synthetic images were added to TRAIN only.")
    print("- Validation and test contain original images only.")
    print("- The original dataset has not been modified.")
    print("- This dataset is an experiment, not a final dataset.")
    print("=" * 60)


if __name__ == "__main__":
    main()
