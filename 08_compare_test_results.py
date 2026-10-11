
"""
FILE: 08_compare_test_results.py

PURPOSE:
    Compare the original baseline classifier against the model
    trained with additional synthetic scenes.

INPUT:
    Original test dataset:
        datasets/tomato/test/Fresh
        datasets/tomato/test/Defect

    Baseline model:
        runs/classify/runs/baseline/weights/best.pt

    Experimental model:
        runs/experiment_synthetic/yolo11n_synthetic_pilot/weights/best.pt

OUTPUT:
    Separate test-evaluation results for each model.
    A printed comparison of Top-1 test accuracy.

IMPORTANT:
    - Uses the same untouched test set for both models.
    - Does not retrain or overwrite either model.
    - Test results are used only after training is complete.
    - Higher accuracy alone does not prove generalization;
      inspect per-class metrics and confusion matrices too.

NEXT STEP:
    Interpret the comparison and decide whether synthetic data
    helped or whether the image-generation method needs improvement.
"""

from pathlib import Path
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. PROJECT PATHS
# ------------------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent
DATASETS_PATH = PROJECT_PATH.parent / "datasets"

ORIGINAL_DATASET = DATASETS_PATH / "tomato"

BASELINE_WEIGHTS = (
    PROJECT_PATH / "runs" / "classify" / "runs"
    / "baseline" / "weights" / "best.pt"
)

EXPERIMENT_WEIGHTS = (
    PROJECT_PATH / "runs" / "experiment_synthetic"
    / "yolo11n_synthetic_pilot" / "weights" / "best.pt"
)

EVALUATION_ROOT = PROJECT_PATH / "runs" / "comparison_test"


# ------------------------------------------------------------
# 2. EVALUATE ONE MODEL
# ------------------------------------------------------------

def evaluate_model(model_name, weights_path, run_name):
    """Evaluate one trained classifier on the original test split."""

    if not weights_path.is_file():
        raise FileNotFoundError(
            f"\nModel weights not found:\n{weights_path}\n"
            "Check the model path before continuing."
        )

    print("\n" + "=" * 65)
    print(f"EVALUATING: {model_name}")
    print("=" * 65)
    print(f"Weights: {weights_path}")
    print(f"Test dataset: {ORIGINAL_DATASET / 'test'}")

    model = YOLO(str(weights_path))

    metrics = model.val(
        data=str(ORIGINAL_DATASET),
        split="test",
        project=str(EVALUATION_ROOT),
        name=run_name,
        plots=True,
        workers=0,
        exist_ok=False,
        verbose=True,
    )

    # Ultralytics classification metrics expose Top-1 accuracy.
    top1_accuracy = float(metrics.top1)

    print(f"\n{model_name} Top-1 test accuracy: "
          f"{top1_accuracy * 100:.2f}%")

    return top1_accuracy


# ------------------------------------------------------------
# 3. MAIN COMPARISON
# ------------------------------------------------------------

def main():
    print("=" * 65)
    print("BASELINE VS. SYNTHETIC-DATA EXPERIMENT")
    print("=" * 65)

    if not ORIGINAL_DATASET.is_dir():
        raise FileNotFoundError(
            f"Original dataset not found: {ORIGINAL_DATASET}"
        )

    baseline_accuracy = evaluate_model(
        model_name="Original baseline",
        weights_path=BASELINE_WEIGHTS,
        run_name="baseline_on_original_test",
    )

    experimental_accuracy = evaluate_model(
        model_name="Synthetic-data experiment",
        weights_path=EXPERIMENT_WEIGHTS,
        run_name="synthetic_on_original_test",
    )

    difference = (
        experimental_accuracy - baseline_accuracy
    ) * 100

    print("\n" + "=" * 65)
    print("FINAL TEST ACCURACY COMPARISON")
    print("=" * 65)
    print(f"Baseline:             {baseline_accuracy * 100:.2f}%")
    print(f"Synthetic experiment: {experimental_accuracy * 100:.2f}%")
    print(f"Difference:           {difference:+.2f} percentage points")

    if difference > 0:
        print("\nThe experimental model scored higher on this test set.")
    elif difference < 0:
        print("\nThe baseline scored higher on this test set.")
    else:
        print("\nBoth models achieved the same Top-1 accuracy.")

    print("\nEvaluation outputs:")
    print(EVALUATION_ROOT)
    print("\nReview each run's confusion matrix and per-class results.")
    print("A small pilot result is evidence, not a final conclusion.")
    print("=" * 65)


if __name__ == "__main__":
    main()
