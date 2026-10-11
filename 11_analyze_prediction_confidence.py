
"""
FILE: 11_analyze_prediction_confidence.py

PURPOSE:
    Compare baseline and synthetic-experiment confidence scores
    for the 11 test images on which their predictions differed.

INPUT:
    runs/prediction_analysis/changed_predictions.csv
    Original baseline model weights.
    Synthetic-experiment model weights.

OUTPUT:
    runs/prediction_analysis/prediction_confidence.csv

IMPORTANT:
    - Analysis only; no training occurs.
    - Original test images and model weights remain unchanged.
    - Confidence is the model's predicted class probability.
    - Confidence is not a guarantee that a prediction is correct.

NEXT STEP:
    Use the results to decide whether further data improvements,
    threshold experiments, or repeated training runs are worthwhile.
"""

from pathlib import Path
import csv

from ultralytics import YOLO

# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent

ANALYSIS_FOLDER = (
    PROJECT_PATH / "runs" / "prediction_analysis"
)

INPUT_CSV = ANALYSIS_FOLDER / "changed_predictions.csv"
OUTPUT_CSV = ANALYSIS_FOLDER / "prediction_confidence.csv"

BASELINE_WEIGHTS = (
    PROJECT_PATH / "runs" / "classify" / "runs"
    / "baseline" / "weights" / "best.pt"
)

EXPERIMENT_WEIGHTS = (
    PROJECT_PATH / "runs" / "experiment_synthetic"
    / "yolo11n_synthetic_pilot" / "weights" / "best.pt"
)

IMAGE_SIZE = 224
CLASSES = ["Defect", "Fresh"]


# ------------------------------------------------------------
# 2. LOAD THE EXISTING DISAGREEMENT REPORT
# ------------------------------------------------------------

def load_changed_images():
    """Load the test images where the two models disagreed."""

    if not INPUT_CSV.is_file():
        raise FileNotFoundError(
            f"Input report not found:\n{INPUT_CSV}\n"
            "Run 09_analyze_model_predictions.py first."
        )

    with INPUT_CSV.open(
        "r", newline="", encoding="utf-8"
    ) as file:
        records = list(csv.DictReader(file))

    if not records:
        raise RuntimeError(
            "No changed predictions were found in the CSV."
        )

    return records


# ------------------------------------------------------------
# 3. LOAD MODELS AND GET CLASS PROBABILITIES
# ------------------------------------------------------------

def load_model(weights_path, model_name):
    """Load a trained classifier without retraining it."""

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"{model_name} weights not found:\n{weights_path}"
        )

    print(f"Loading {model_name}...")
    return YOLO(str(weights_path))


def analyze_one_image(model, image_path):
    """Return predicted class and probability for each class."""

    results = model.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        device="cpu",
        verbose=False,
    )

    result = results[0]

    # Convert model class indices to readable class names.
    probabilities = result.probs.data.tolist()
    class_names = {
        int(index): str(name)
        for index, name in model.names.items()
    }

    class_probabilities = {}

    for index, probability in enumerate(probabilities):
        class_name = class_names[index]
        class_probabilities[class_name] = float(probability)

    predicted_index = int(result.probs.top1)
    predicted_class = class_names[predicted_index]
    predicted_confidence = float(result.probs.top1conf.item())

    return (
        predicted_class,
        predicted_confidence,
        class_probabilities,
    )


# ------------------------------------------------------------
# 4. MAIN ANALYSIS
# ------------------------------------------------------------

def main():
    print("=" * 65)
    print("PREDICTION CONFIDENCE ANALYSIS")
    print("=" * 65)

    records = load_changed_images()
    print(f"Images to analyze: {len(records)}")

    baseline_model = load_model(
        BASELINE_WEIGHTS, "Baseline model"
    )

    experiment_model = load_model(
        EXPERIMENT_WEIGHTS, "Synthetic experiment model"
    )

    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

    fields = [
        "image_path",
        "actual_class",
        "baseline_prediction",
        "baseline_confidence",
        "baseline_actual_class_probability",
        "experiment_prediction",
        "experiment_confidence",
        "experiment_actual_class_probability",
        "baseline_correct",
        "experiment_correct",
    ]

    results_summary = []

    with OUTPUT_CSV.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()

        for number, record in enumerate(records, start=1):
            image_path = Path(record["image_path"])
            actual_class = record["actual_class"]

            if not image_path.is_file():
                print(f"Skipping missing image: {image_path}")
                continue

            baseline_pred, baseline_conf, baseline_probs = (
                analyze_one_image(baseline_model, image_path)
            )

            experiment_pred, experiment_conf, experiment_probs = (
                analyze_one_image(experiment_model, image_path)
            )

            row = {
                "image_path": str(image_path),
                "actual_class": actual_class,
                "baseline_prediction": baseline_pred,
                "baseline_confidence": round(baseline_conf, 6),
                "baseline_actual_class_probability": round(
                    baseline_probs[actual_class], 6
                ),
                "experiment_prediction": experiment_pred,
                "experiment_confidence": round(
                    experiment_conf, 6
                ),
                "experiment_actual_class_probability": round(
                    experiment_probs[actual_class], 6
                ),
                "baseline_correct": baseline_pred == actual_class,
                "experiment_correct": (
                    experiment_pred == actual_class
                ),
            }

            writer.writerow(row)
            results_summary.append(row)

            print(
                f"{number:02d}. Actual={actual_class:6s} | "
                f"Baseline={baseline_pred:6s} "
                f"({baseline_conf:.1%}) | "
                f"Experiment={experiment_pred:6s} "
                f"({experiment_conf:.1%})"
            )

    print("\n" + "=" * 65)
    print("CONFIDENCE ANALYSIS FINISHED")
    print("=" * 65)
    print(f"Images analyzed: {len(results_summary)}")
    print(f"CSV saved to:\n{OUTPUT_CSV}")
    print("\nInterpretation:")
    print("- Confidence is a model probability, not a certainty.")
    print("- Compare confidence with actual labels and errors.")
    print("- Do not assume a high-confidence prediction is correct.")
    print("=" * 65)


if __name__ == "__main__":
    main()
