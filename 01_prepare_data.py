
# ============================================================
# FILE: 01_prepare_data.py
# PURPOSE:
# Prepare the original tomato images for our machine-learning
# project by dividing them into training, validation, and test
# sets.
# ============================================================

from pathlib import Path
import csv
import random
import shutil
import sys


# ------------------------------------------------------------
# STEP 1: FOLDER PATHS
# ------------------------------------------------------------

# This folder must contain the Fresh and Rotten subfolders.
SOURCE = Path(
    r"E:\AUC\Advanced machine learning\project\M1\project\Original Dataset"
)

# The prepared dataset will be created here.
OUTPUT = Path(
    r"E:\AUC\Advanced machine learning\project\M1\datasets\tomato"
)


# ------------------------------------------------------------
# STEP 2: DATASET SETTINGS
# ------------------------------------------------------------

# File types that we recognize as images.
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# The percentage of images assigned to each split.
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# The remaining images go into the test set.

# A fixed seed makes the random split repeatable.
SEED = 42

# Map the original dataset labels to our project labels.
CLASS_MAP = {
    "Fresh": "Fresh",
    "Rotten": "Defect",
}


# ------------------------------------------------------------
# STEP 3: FIND IMAGES
# ------------------------------------------------------------

def find_images(folder):
    """
    Return image files directly inside the given folder.
    """
    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file()
        and file.suffix.lower() in IMAGE_EXTENSIONS
    )


# ------------------------------------------------------------
# STEP 4: SPLIT THE IMAGES
# ------------------------------------------------------------

def split_images(images):
    """
    Shuffle images and divide them into train, val, and test.
    """
    images = images.copy()

    # Randomize the order of the images.
    random.shuffle(images)

    total = len(images)

    # Calculate the number of images in each split.
    train_count = int(total * TRAIN_RATIO)
    val_count = int(total * VAL_RATIO)

    # Select different parts of the shuffled list.
    train = images[:train_count]
    val = images[train_count:train_count + val_count]
    test = images[train_count + val_count:]

    return {
        "train": train,
        "val": val,
        "test": test,
    }


# ------------------------------------------------------------
# STEP 5: MAIN PROGRAM
# ------------------------------------------------------------

def main():

    print("=" * 60)
    print("TOMATO DATASET PREPARATION")
    print("=" * 60)

    # Check that the original dataset folder exists.
    if not SOURCE.is_dir():
        sys.exit(
            "\nERROR: The original dataset folder was not found.\n"
            f"Expected location:\n{SOURCE}\n\n"
            "Check the SOURCE path near the beginning of this file."
        )

    # Make sure the output is not inside the original dataset.
    source_path = SOURCE.resolve()
    output_path = OUTPUT.resolve()

    if output_path.is_relative_to(source_path):
        sys.exit(
            "\nERROR: The output folder must not be inside "
            "the original dataset."
        )

    print(f"\nOriginal dataset: {SOURCE}")
    print(f"Output dataset:   {OUTPUT}")

    # Use the same random seed every time.
    random.seed(SEED)

    # Find the images belonging to each class.
    images_by_class = {}

    for source_class, project_class in CLASS_MAP.items():

        class_folder = SOURCE / source_class

        # Check that this class folder exists.
        if not class_folder.is_dir():
            sys.exit(
                f"\nERROR: Class folder not found:\n{class_folder}"
            )

        # Find all image files in this class folder.
        images = find_images(class_folder)

        # Store the images using our project class label.
        images_by_class[project_class] = images

        print(
            f"\n{project_class}: {len(images)} original images"
        )

        if not images:
            sys.exit(
                f"\nERROR: No supported images found in:\n{class_folder}"
            )

    # Do not overwrite a previous prepared dataset.
    if OUTPUT.exists():
        sys.exit(
            "\nSTOP: The output folder already exists:\n"
            f"{OUTPUT}\n\n"
            "To protect your work, this script will not overwrite it. "
            "Check the folder before trying again."
        )

    # Keep a record of every copied image.
    manifest_rows = []

    # Process Fresh and Defect separately.
    for class_name, images in images_by_class.items():

        print(f"\nPreparing class: {class_name}")

        # Divide this class into three splits.
        splits = split_images(images)

        # Process train, val, and test.
        for split_name, image_list in splits.items():

            # Create the destination folder.
            destination = OUTPUT / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            # Copy each image into the destination.
            for index, source_file in enumerate(image_list):

                # Give the copied file a clear new name.
                new_name = (
                    f"{class_name}_{split_name}_{index:04d}"
                    f"{source_file.suffix.lower()}"
                )

                destination_file = destination / new_name

                # Copy the image without changing the original.
                shutil.copy2(source_file, destination_file)

                # Record where this image came from.
                manifest_rows.append([
                    split_name,
                    class_name,
                    str(source_file),
                    new_name,
                ])

            # Show how many images were copied.
            print(
                f"  {split_name:5s} | "
                f"{class_name:6s} | "
                f"{len(image_list)} images"
            )

    # --------------------------------------------------------
    # STEP 6: SAVE THE IMAGE RECORD
    # --------------------------------------------------------

    manifest_path = OUTPUT / "split_manifest.csv"

    with manifest_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        writer = csv.writer(file)

        # Write the column headings.
        writer.writerow([
            "split",
            "class",
            "original_path",
            "new_filename",
        ])

        # Write the record for every copied image.
        writer.writerows(manifest_rows)

    # --------------------------------------------------------
    # STEP 7: FINISH
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DATASET PREPARATION COMPLETE!")
    print("=" * 60)

    print(f"\nPrepared dataset:\n{OUTPUT}")
    print(f"\nImage record:\n{manifest_path}")

    print("\nExpected folder structure:")
    print("datasets/tomato/")
    print("    train/")
    print("        Fresh/")
    print("        Defect/")
    print("    val/")
    print("        Fresh/")
    print("        Defect/")
    print("    test/")
    print("        Fresh/")
    print("        Defect/")


# Run the main program when this file is executed.
if __name__ == "__main__":
    main()

