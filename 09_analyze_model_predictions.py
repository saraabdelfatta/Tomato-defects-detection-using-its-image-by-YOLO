
"""
FILE: 09_analyze_model_predictions.py

PURPOSE:
    Analyze and compare baseline and synthetic-experiment predictions
    on the same original test images.

INPUT:
    Original test dataset: datasets/tomato/test/
    Baseline weights: runs/classify/runs/baseline/weights/best.pt
    Experimental weights: runs/experiment_synthetic/
                          yolo11n_synthetic_pilot/weights/best.pt

OUTPUT:
    Per-class precision, recall, and F1-score.
    Confusion matrices.
    Number of images where model predictions differ.
    A CSV file listing changed predictions.

IMPORTANT:
    - The original test set remains unchanged.
    - This is analysis only; no training occurs.
    - Predictions are matched by image path.
"""

from pathlib import Path
import csv
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent
DATASETS_PATH = PROJECT_PATH.parent / "datasets"
TEST_ROOT = DATASETS_PATH / "tomato" / "test"

BASELINE_WEIGHTS = (
    PROJECT_PATH / "runs" / "classify" / "runs"
    / "baseline" / "weights" / "best.pt"
)

EXPERIMENT_WEIGHTS = (
    PROJECT_PATH / "runs" / "experiment_synthetic"
    / "yolo11n_synthetic_pilot" / "weights" / "best.pt"
)

OUTPUT_FOLDER = PROJECT_PATH / "runs" / "prediction_analysis"
OUTPUT_CSV = OUTPUT_FOLDER / "changed_predictions.csv"

CLASSES = ["Defect", "Fresh"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------
# 2. COLLECT TEST IMAGES
# ------------------------------------------------------------

def collect_test_images():
    """Return image paths and true labels from class folders."""
    image_paths = []
    true_labels = []

    for class_name in CLASSES:
        class_folder = TEST_ROOT / class_name

        if not class_folder.is_dir():
            raise FileNotFoundError(
                f"Test class folder not found: {class_folder}"
            )

        files = sorted(
            path for path in class_folder.iterdir()
            if path.is_file()
            and path.suffix.lower() in IMAGE_EXTENSIONS
        )

        for path in files:
            image_paths.append(path)
            true_labels.append(CLASSES.index(class_name))

    if not image_paths:
        raise RuntimeError(f"No test images found in {TEST_ROOT}")

    return image_paths, np.array(true_labels, dtype=int)


# ------------------------------------------------------------
# 3. GET PREDICTIONS
# ------------------------------------------------------------

def predict_images(weights_path, image_paths, model_label):
    """Predict all test images and return class indices."""
    if not weights_path.is_file():
        raise FileNotFoundError(f"Model weights not found: {weights_path}")

    print(f"\nLoading {model_label}...")
    model = YOLO(str(weights_path))

    predictions = []

    # Process in batches to avoid loading all images at once.
    batch_size = 16

    for start in range(0, len(image_paths), batch_size):
        batch = image_paths[start:start + batch_size]

        results = model.predict(
            source=[str(path) for path in batch],
            imgsz=224,
            batch=batch_size,
            device="cpu",
            verbose=False,
        )

        for result in results:
            # Ultralytics returns the predicted class index.
            predictions.append(int(result.probs.top1))

        print(
            f"  {model_label}: "
            f"{min(start + len(batch), len(image_paths))}"
            f"/{len(image_paths)} images processed"
        )

    # Convert model-specific class indices into our class order.
    # This prevents incorrect comparisons if class ordering differs.
    model_names = model.names
    model_index_to_name = {
        int(index): str(name)
        for index, name in model_names.items()
    }

    converted = []

    for prediction in predictions:
        predicted_name = model_index_to_name[prediction]

        if predicted_name not in CLASSES:
            raise ValueError(
                f"Unexpected model class: {predicted_name}"
            )

        converted.append(CLASSES.index(predicted_name))

    return np.array(converted, dtype=int)


# ------------------------------------------------------------
# 4. REPORT METRICS
# ------------------------------------------------------------

def report_metrics(label, true_labels, predictions):
    """Print accuracy, precision, recall, F1, and confusion matrix."""
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=int)

    for actual, predicted in zip(true_labels, predictions):
        confusion[actual, predicted] += 1

    accuracy = np.mean(true_labels == predictions)

    print("\n" + "=" * 60)
    print(label)
    print("=" * 60)
    print(f"Accuracy: {accuracy * 100:.2f}%")

    print("\nConfusion matrix (rows = actual, columns = predicted)")
    print(f"{'':12s}" + "".join(f"{name:>12s}" for name in CLASSES))

    for index, class_name in enumerate(CLASSES):
        print(
            f"{class_name:12s}"
            + "".join(f"{value:12d}" for value in confusion[index])
        )

    print("\nPer-class metrics:")

    for index, class_name in enumerate(CLASSES):
        true_positive = confusion[index, index]
        false_positive = confusion[:, index].sum() - true_positive
        false_negative = confusion[index, :].sum() - true_positive

        precision = (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive else 0.0
        )
        recall = (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative else 0.0
        )
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall else 0.0
        )

        print(
            f"  {class_name}: "
            f"Precision={precision * 100:.2f}%, "
            f"Recall={recall * 100:.2f}%, "
            f"F1={f1 * 100:.2f}%"
        )

    return accuracy


# ------------------------------------------------------------
# 5. MAIN ANALYSIS
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("MODEL PREDICTION ANALYSIS")
    print("=" * 60)

    image_paths, true_labels = collect_test_images()
    print(f"Original test images: {len(image_paths)}")

    baseline_predictions = predict_images(
        BASELINE_WEIGHTS, image_paths, "Baseline"
    )

    experiment_predictions = predict_images(
        EXPERIMENT_WEIGHTS, image_paths, "Synthetic experiment"
    )

    report_metrics("BASELINE", true_labels, baseline_predictions)
    report_metrics(
        "SYNTHETIC EXPERIMENT",
        true_labels,
        experiment_predictions,
    )

    # Identify images on which the models disagree.
    changed_indices = np.where(
        baseline_predictions != experiment_predictions
    )[0]

    print("\n" + "=" * 60)
    print("PREDICTION DIFFERENCES")
    print("=" * 60)
    print(f"Images with different predictions: {len(changed_indices)}")

    baseline_correct = baseline_predictions == true_labels
    experiment_correct = experiment_predictions == true_labels

    improved = np.sum(
        (~baseline_correct[changed_indices])
        & experiment_correct[changed_indices]
    )
    worsened = np.sum(
        baseline_correct[changed_indices]
        & (~experiment_correct[changed_indices])
    )

    print(f"Wrong in baseline, correct in experiment: {improved}")
    print(f"Correct in baseline, wrong in experiment: {worsened}")

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow([
            "image_path",
            "actual_class",
            "baseline_prediction",
            "experiment_prediction",
            "baseline_correct",
            "experiment_correct",
        ])

        for index in changed_indices:
            writer.writerow([
                str(image_paths[index]),
                CLASSES[true_labels[index]],
                CLASSES[baseline_predictions[index]],
                CLASSES[experiment_predictions[index]],
                bool(baseline_correct[index]),
                bool(experiment_correct[index]),
            ])

    print(f"\nChanged predictions saved to:\n{OUTPUT_CSV}")
    print("\nAnalysis finished.")


if __name__ == "__main__":
    main()
