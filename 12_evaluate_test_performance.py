"""
FILE: 12_evaluate_test_performance.py

PURPOSE:
    Evaluate the baseline YOLO11n classifier on the held-out test set.

INPUT:
    datasets/tomato/test/{Defect,Fresh}
    Baseline model weights from runs/classify/runs/baseline/weights.

OUTPUT:
    runs/slide7_baseline/test_predictions.csv
    runs/slide7_baseline/confusion_matrix_test.png
    runs/slide7_baseline/roc_curve_test.png

IMPORTANT:
    - Evaluation only; no training occurs.
    - The test set should be used for final evaluation, not model tuning.
    - Output paths are relative to this script, not the current directory.
"""

from pathlib import Path
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent

DATASET_PATH = (
    PROJECT_PATH.parent / "datasets" / "tomato" / "test"
)

WEIGHTS_PATH = (
    PROJECT_PATH / "runs" / "classify" / "runs"
    / "baseline" / "weights" / "best.pt"
)

OUTPUT_FOLDER = PROJECT_PATH / "runs" / "slide7_baseline"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CLASS_NAMES = ("Defect", "Fresh")
THRESHOLD = 0.5


# ------------------------------------------------------------
# 2. CALCULATE TEST METRICS
# ------------------------------------------------------------

def evaluate(y_true, p_defect, threshold=THRESHOLD):
    """Calculate confusion counts, class metrics, ROC points, and AUC."""
    y_true = np.asarray(y_true, dtype=int)
    p_defect = np.asarray(p_defect, dtype=float)

    if y_true.ndim != 1 or p_defect.ndim != 1:
        raise ValueError("Labels and probabilities must be one-dimensional.")
    if len(y_true) == 0 or len(y_true) != len(p_defect):
        raise ValueError(
            "Labels and probabilities must have the same non-zero length."
        )
    if not np.isin(y_true, (0, 1)).all():
        raise ValueError("Labels must be 1 for Defect or 0 for Fresh.")
    if not np.isfinite(p_defect).all():
        raise ValueError("Predicted probabilities must be finite.")
    if ((p_defect < 0) | (p_defect > 1)).any():
        raise ValueError("Predicted probabilities must be between 0 and 1.")
    if not 0 <= threshold <= 1:
        raise ValueError("The classification threshold must be between 0 and 1.")

    positive_count = int(y_true.sum())
    negative_count = len(y_true) - positive_count
    if not positive_count or not negative_count:
        raise ValueError(
            "The test set must contain at least one Defect and one Fresh image."
        )

    y_pred = (p_defect >= threshold).astype(int)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    def precision_recall_f1(true_positive, false_positive, false_negative):
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
        return precision, recall, f1

    defect_precision, defect_recall, defect_f1 = (
        precision_recall_f1(tp, fp, fn)
    )
    fresh_precision, fresh_recall, fresh_f1 = (
        precision_recall_f1(tn, fn, fp)
    )

    # Add one ROC point per distinct score so tied predictions share a threshold.
    order = np.argsort(-p_defect, kind="mergesort")
    sorted_scores = p_defect[order]
    sorted_labels = y_true[order]
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores)),
        len(sorted_scores) - 1,
    ]
    true_positives = np.cumsum(sorted_labels)[group_ends]
    false_positives = (group_ends + 1) - true_positives
    tpr = np.r_[0.0, true_positives / positive_count]
    fpr = np.r_[0.0, false_positives / negative_count]
    auc = float(
        np.sum(
            (fpr[1:] - fpr[:-1])
            * (tpr[1:] + tpr[:-1])
            / 2
        )
    )

    return {
        "tp": tp,
        "fn": fn,
        "fp": fp,
        "tn": tn,
        "accuracy": (tp + tn) / len(y_true),
        "defect_precision": defect_precision,
        "defect_recall": defect_recall,
        "defect_f1": defect_f1,
        "fresh_precision": fresh_precision,
        "fresh_recall": fresh_recall,
        "fresh_f1": fresh_f1,
        "auc": auc,
        "fpr": fpr,
        "tpr": tpr,
    }


# ------------------------------------------------------------
# 3. SAVE EVALUATION PLOTS
# ------------------------------------------------------------

def plot_confusion(metrics, path):
    """Save a confusion matrix with predicted rows and true columns."""
    grid = np.array([
        [metrics["tp"], metrics["fp"]],
        [metrics["fn"], metrics["tn"]],
    ])

    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=200)
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(int(grid.max()), 1))

    for row in range(2):
        for column in range(2):
            ax.text(
                column,
                row,
                str(grid[row, column]),
                ha="center",
                va="center",
                fontsize=16,
                color="white" if grid[row, column] > grid.max() / 2 else "black",
            )

    ax.set_xticks([0, 1], CLASS_NAMES)
    ax.set_yticks([0, 1], CLASS_NAMES)
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title("Confusion matrix, test set")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def plot_roc(metrics, path):
    """Save the ROC curve and its area under the curve."""
    fig, ax = plt.subplots(figsize=(5, 4.2), dpi=200)
    ax.plot(
        metrics["fpr"],
        metrics["tpr"],
        lw=2,
        color="#2e5e8c",
        label=f"YOLO11n-cls (AUC = {metrics['auc']:.3f})",
    )
    ax.plot([0, 1], [0, 1], "--", color="grey", label="Chance")
    ax.set_xlabel("False positive rate (Fresh flagged as Defect)")
    ax.set_ylabel("True positive rate (Defect recall)")
    ax.set_title("ROC curve, test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


# ------------------------------------------------------------
# 4. RUN TEST-SET EVALUATION
# ------------------------------------------------------------

def main():
    if not DATASET_PATH.is_dir():
        raise FileNotFoundError(
            f"Test dataset folder not found:\n{DATASET_PATH}\n"
            "Run 01_prepare_data.py first or update DATASET_PATH."
        )
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Baseline weights not found:\n{WEIGHTS_PATH}\n"
            "Run 02_train_baseline.py first or update WEIGHTS_PATH."
        )

    files = []
    labels = []

    for class_name, label in (("Defect", 1), ("Fresh", 0)):
        class_folder = DATASET_PATH / class_name
        if not class_folder.is_dir():
            raise FileNotFoundError(
                f"Test class folder not found:\n{class_folder}"
            )

        class_files = sorted(
            path for path in class_folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        files.extend(class_files)
        labels.extend([label] * len(class_files))

    if not files:
        raise RuntimeError(f"No supported test images found in {DATASET_PATH}.")

    y_true = np.asarray(labels, dtype=int)
    print(
        f"Test images: {len(files)} "
        f"(Defect {int(y_true.sum())}, Fresh {len(y_true) - int(y_true.sum())})"
    )

    model = YOLO(str(WEIGHTS_PATH))
    defect_indices = [
        int(index)
        for index, name in model.names.items()
        if name == "Defect"
    ]
    if len(defect_indices) != 1:
        raise ValueError(
            f"Expected exactly one 'Defect' class in model.names; "
            f"found {len(defect_indices)}."
        )
    defect_index = defect_indices[0]

    predictions = model.predict(
        [str(path) for path in files],
        verbose=False,
        stream=True,
    )
    p_defect = np.asarray(
        [
            float(result.probs.data.cpu().numpy()[defect_index])
            for result in predictions
        ],
        dtype=float,
    )
    if len(p_defect) != len(files):
        raise RuntimeError(
            f"Model returned {len(p_defect)} predictions for "
            f"{len(files)} test images."
        )

    metrics = evaluate(y_true, p_defect)
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    predictions_path = OUTPUT_FOLDER / "test_predictions.csv"
    with predictions_path.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["image_path", "actual", "p_defect"])
        for image_path, label, probability in zip(files, y_true, p_defect):
            writer.writerow([
                str(image_path),
                "Defect" if label else "Fresh",
                f"{probability:.6f}",
            ])

    plot_confusion(
        metrics, OUTPUT_FOLDER / "confusion_matrix_test.png"
    )
    plot_roc(metrics, OUTPUT_FOLDER / "roc_curve_test.png")

    print(
        f"\nConfusion: Defect->Defect {metrics['tp']} | "
        f"Defect->Fresh {metrics['fn']} | "
        f"Fresh->Defect {metrics['fp']} | "
        f"Fresh->Fresh {metrics['tn']}"
    )
    for key in (
        "accuracy",
        "defect_precision",
        "defect_recall",
        "defect_f1",
        "fresh_precision",
        "fresh_recall",
        "fresh_f1",
        "auc",
    ):
        print(f"{key:18s} {metrics[key]:.4f}")

    print(f"\nPredictions and plots saved to:\n{OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
