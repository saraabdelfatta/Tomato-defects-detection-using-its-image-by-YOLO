
# ============================================================
# FILE: 03_evaluate_baseline.py
#
# PURPOSE:
# Evaluate the trained baseline YOLO11 classification model
# on the held-out test dataset to measure its performance on
# tomato images it did not use during training or validation.
#
# INPUT:
# The prepared dataset created by 01_prepare_data.py
# The best model weights saved by 02_train_baseline.py
#
# OUTPUT:
# Test accuracy, detailed classification metrics, and
# evaluation plots such as the confusion matrix.
#
# IMPORTANT:
# Use the test split only for final evaluation, not for
# selecting model settings or tuning the training process.
# ============================================================


from pathlib import Path
from ultralytics import YOLO

# The dataset containing train, val, and test folders.
DATASET_PATH = Path(
    r"E:\AUC\Advanced machine learning\project\M1\datasets\tomato"
)

# The actual best.pt location shown in your training output.
MODEL_PATH = Path(
    r"E:\AUC\Advanced machine learning\project\M1\project"
    r"\runs\classify\runs\baseline\weights\best.pt"
)


def main():
    # Check that both required paths exist before evaluation.
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_PATH}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {MODEL_PATH}"
        )

    # Load the best model, not the last training checkpoint.
    model = YOLO(str(MODEL_PATH))

    # Evaluate on the held-out test split.
    metrics = model.val(
        data=str(DATASET_PATH),
        split="test",
        imgsz=224,
        plots=True,
        project="runs",
        name="baseline_test",
        exist_ok=True,
    )

    # Print the main classification metrics.
    print("\n===== BASELINE TEST RESULTS =====")
    print(f"Top-1 accuracy: {metrics.top1:.4f}")
    print(f"Top-5 accuracy: {metrics.top5:.4f}")
    print("\nDetailed results:")
    print(metrics.results_dict)


if __name__ == "__main__":
    main()
