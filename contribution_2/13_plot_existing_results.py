"""
FILE: 13_plot_existing_results.py

PURPOSE:
    Draw report plots from results that are already saved under runs/.

INPUT:
    runs/classify/runs/baseline/results.csv
    runs/experiment_synthetic/yolo11n_synthetic_pilot/results.csv
    runs/slide7_baseline/test_predictions.csv
    runs/prediction_analysis/changed_predictions.csv
    runs/prediction_analysis/prediction_confidence.csv

OUTPUT:
    contribution_2/runs/plots/*.png

IMPORTANT:
    - No training and no model inference occur; only saved CSVs are read.
    - Needs only matplotlib and numpy (no ultralytics, no dataset folder).
    - The synthetic model's test predictions are reconstructed as the
      baseline predictions with the rows in changed_predictions.csv flipped.
"""

from pathlib import Path, PureWindowsPath
import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

# This script lives in contribution_2/, one level below the project.
SESSION_PATH = Path(__file__).resolve().parent
PROJECT_PATH = SESSION_PATH.parent
RUNS_PATH = PROJECT_PATH / "runs"

BASELINE_RESULTS = RUNS_PATH / "classify" / "runs" / "baseline" / "results.csv"
SYNTHETIC_RESULTS = (
    RUNS_PATH / "experiment_synthetic"
    / "yolo11n_synthetic_pilot" / "results.csv"
)
TEST_PREDICTIONS = RUNS_PATH / "slide7_baseline" / "test_predictions.csv"
CHANGED_PREDICTIONS = (
    RUNS_PATH / "prediction_analysis" / "changed_predictions.csv"
)
PREDICTION_CONFIDENCE = (
    RUNS_PATH / "prediction_analysis" / "prediction_confidence.csv"
)

OUTPUT_FOLDER = SESSION_PATH / "runs" / "plots"
THRESHOLD = 0.5

BASELINE_COLOR = "#2a78d6"
SYNTHETIC_COLOR = "#eb6834"
INK = "#0b0b0b"
SECONDARY_INK = "#52514e"
MUTED = "#898781"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"

plt.rcParams.update({
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
    "figure.dpi": 200,
    "font.size": 9,
    "text.color": INK,
    "axes.labelcolor": SECONDARY_INK,
    "axes.edgecolor": MUTED,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "axes.titlelocation": "left",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "axes.axisbelow": True,
    "grid.color": GRID,
    "grid.linewidth": 0.8,
    "xtick.color": MUTED,
    "ytick.color": MUTED,
    "xtick.labelcolor": SECONDARY_INK,
    "ytick.labelcolor": SECONDARY_INK,
    "lines.linewidth": 2,
    "legend.frameon": False,
})


# ------------------------------------------------------------
# 2. LOAD SAVED RESULTS
# ------------------------------------------------------------

def read_rows(path):
    if not path.is_file():
        raise FileNotFoundError(f"Saved results file not found:\n{path}")
    with path.open(newline="", encoding="utf-8") as file:
        return [
            {key.strip(): value.strip() for key, value in row.items()}
            for row in csv.DictReader(file)
        ]


def read_training_log(path):
    rows = read_rows(path)
    return {
        "epoch": np.array([int(row["epoch"]) for row in rows]),
        "train_loss": np.array([float(row["train/loss"]) for row in rows]),
        "val_loss": np.array([float(row["val/loss"]) for row in rows]),
        "val_accuracy": np.array(
            [float(row["metrics/accuracy_top1"]) for row in rows]
        ),
    }


def image_name(path_text):
    """The CSVs were written on Windows, so parse their paths as such."""
    return PureWindowsPath(path_text).name


def confusion(y_true, y_pred):
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())

    def ratio(numerator, denominator):
        return numerator / denominator if denominator else 0.0

    defect_precision = ratio(tp, tp + fp)
    defect_recall = ratio(tp, tp + fn)
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "accuracy": ratio(tp + tn, len(y_true)),
        "defect_precision": defect_precision,
        "defect_recall": defect_recall,
        "defect_f1": ratio(
            2 * defect_precision * defect_recall,
            defect_precision + defect_recall,
        ),
        "fresh_precision": ratio(tn, tn + fn),
        "fresh_recall": ratio(tn, tn + fp),
    }


def roc_points(y_true, p_defect):
    """One ROC point per distinct score, as in 12_evaluate_test_performance."""
    order = np.argsort(-p_defect, kind="mergesort")
    sorted_scores = p_defect[order]
    sorted_labels = y_true[order]
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_scores)),
        len(sorted_scores) - 1,
    ]
    true_positives = np.cumsum(sorted_labels)[group_ends]
    false_positives = (group_ends + 1) - true_positives
    tpr = np.r_[0.0, true_positives / y_true.sum()]
    fpr = np.r_[0.0, false_positives / (len(y_true) - y_true.sum())]
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return fpr, tpr, auc


def load_test_predictions():
    """Return labels, baseline scores, and both models' test predictions."""
    rows = read_rows(TEST_PREDICTIONS)
    names = [image_name(row["image_path"]) for row in rows]
    y_true = np.array([row["actual"] == "Defect" for row in rows], dtype=int)
    p_defect = np.array([float(row["p_defect"]) for row in rows])
    baseline_pred = (p_defect >= THRESHOLD).astype(int)

    synthetic_pred = baseline_pred.copy()
    index_by_name = {name: index for index, name in enumerate(names)}
    for row in read_rows(CHANGED_PREDICTIONS):
        index = index_by_name[image_name(row["image_path"])]
        saved_baseline = int(row["baseline_prediction"] == "Defect")
        if saved_baseline != baseline_pred[index]:
            raise ValueError(
                f"{names[index]}: changed_predictions.csv disagrees with "
                "test_predictions.csv about the baseline prediction."
            )
        synthetic_pred[index] = int(row["experiment_prediction"] == "Defect")

    return y_true, p_defect, baseline_pred, synthetic_pred


# ------------------------------------------------------------
# 3. PLOTS
# ------------------------------------------------------------

def save(fig, name):
    path = OUTPUT_FOLDER / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_PATH)}")


def plot_training_curves(log, title, name):
    """Loss and validation accuracy for one training run."""
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(9, 3.6))
    best = int(np.argmax(log["val_accuracy"]))

    loss_ax.plot(log["epoch"], log["train_loss"],
                 color=BASELINE_COLOR, label="Train loss")
    loss_ax.plot(log["epoch"], log["val_loss"],
                 color=SYNTHETIC_COLOR, label="Validation loss")
    loss_ax.set_title("Loss")
    loss_ax.set_xlabel("Epoch")
    loss_ax.set_ylim(bottom=0)
    loss_ax.legend(loc="upper right")

    accuracy_ax.plot(log["epoch"], 100 * log["val_accuracy"],
                     color=BASELINE_COLOR, marker="o", markersize=4)
    accuracy_ax.scatter(
        log["epoch"][best], 100 * log["val_accuracy"][best],
        s=60, color=BASELINE_COLOR, edgecolor=SURFACE, linewidth=2, zorder=3,
    )
    accuracy_ax.annotate(
        f"Best: {100 * log['val_accuracy'][best]:.2f}% "
        f"(epoch {log['epoch'][best]})",
        (log["epoch"][best], 100 * log["val_accuracy"][best]),
        xytext=(0, 9), textcoords="offset points", ha="center", color=INK,
    )
    accuracy_ax.set_title("Validation top-1 accuracy (%)")
    accuracy_ax.set_xlabel("Epoch")
    accuracy_ax.set_ylim(90, 100.9)

    for ax in (loss_ax, accuracy_ax):
        ax.set_xticks(log["epoch"][::2])
    fig.suptitle(title, x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, name)


def plot_training_comparison(baseline, synthetic):
    """Validation loss and accuracy of both runs on shared axes."""
    fig, (loss_ax, accuracy_ax) = plt.subplots(1, 2, figsize=(9, 3.6))
    runs = (
        ("Baseline", baseline, BASELINE_COLOR),
        ("Synthetic pilot", synthetic, SYNTHETIC_COLOR),
    )
    for label, log, color in runs:
        loss_ax.plot(log["epoch"], log["val_loss"], color=color, label=label)
        accuracy_ax.plot(log["epoch"], 100 * log["val_accuracy"],
                         color=color, marker="o", markersize=4, label=label)

    loss_ax.set_title("Validation loss")
    loss_ax.set_ylim(bottom=0)
    accuracy_ax.set_title("Validation top-1 accuracy (%)")
    accuracy_ax.set_ylim(90, 100.9)
    accuracy_ax.legend(loc="lower right")
    for ax in (loss_ax, accuracy_ax):
        ax.set_xlabel("Epoch")
        ax.set_xticks(baseline["epoch"][::2])
    fig.suptitle("Training runs compared (original validation split)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "training_comparison.png")


def draw_confusion(ax, metrics, title):
    """Predicted rows and true columns, matching the existing test matrix."""
    grid = np.array([
        [metrics["tp"], metrics["fp"]],
        [metrics["fn"], metrics["tn"]],
    ])
    ax.imshow(grid, cmap="Blues", vmin=0, vmax=max(int(grid.max()), 1))
    for row in range(2):
        for column in range(2):
            ax.text(
                column, row, str(grid[row, column]),
                ha="center", va="center", fontsize=16,
                color="white" if grid[row, column] > grid.max() / 2 else INK,
            )
    ax.set_xticks([0, 1], ("Defect", "Fresh"))
    ax.set_yticks([0, 1], ("Defect", "Fresh"))
    ax.set_xlabel("True")
    ax.set_ylabel("Predicted")
    ax.set_title(title)
    ax.grid(False)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def plot_confusion_matrices(baseline, synthetic):
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.9))
    draw_confusion(
        axes[0], baseline,
        f"Baseline ({100 * baseline['accuracy']:.2f}% accuracy)",
    )
    draw_confusion(
        axes[1], synthetic,
        f"Synthetic pilot ({100 * synthetic['accuracy']:.2f}% accuracy)",
    )
    fig.suptitle("Confusion matrices, original test set (300 images)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "confusion_matrices_test.png")


def plot_metric_comparison(baseline, synthetic):
    """Grouped bars of the headline test metrics for both models."""
    metrics = (
        ("Accuracy", "accuracy"),
        ("Defect recall", "defect_recall"),
        ("Defect precision", "defect_precision"),
        ("Fresh recall", "fresh_recall"),
        ("Fresh precision", "fresh_precision"),
    )
    positions = np.arange(len(metrics))
    width = 0.36

    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for offset, label, values, color in (
        (-width / 2, "Baseline", baseline, BASELINE_COLOR),
        (width / 2, "Synthetic pilot", synthetic, SYNTHETIC_COLOR),
    ):
        heights = [100 * values[key] for _, key in metrics]
        bars = ax.bar(positions + offset, heights, width - 0.03,
                      color=color, label=label)
        ax.bar_label(bars, fmt="%.1f", padding=2, color=INK, fontsize=8)

    ax.set_xticks(positions, [label for label, _ in metrics])
    ax.set_ylim(80, 103)
    ax.set_ylabel("%")
    ax.set_title("Test metrics, original test set (axis starts at 80%)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncols=2)
    fig.tight_layout()
    save(fig, "test_metric_comparison.png")


def plot_error_counts(baseline, synthetic):
    """The two error types, where the models differ most."""
    errors = (
        ("Defect predicted as Fresh\n(missed defects)", "fn"),
        ("Fresh predicted as Defect\n(false alarms)", "fp"),
    )
    positions = np.arange(len(errors))
    width = 0.36

    fig, ax = plt.subplots(figsize=(6, 3.8))
    for offset, label, values, color in (
        (-width / 2, "Baseline", baseline, BASELINE_COLOR),
        (width / 2, "Synthetic pilot", synthetic, SYNTHETIC_COLOR),
    ):
        bars = ax.bar(positions + offset, [values[key] for _, key in errors],
                      width - 0.03, color=color, label=label)
        ax.bar_label(bars, padding=2, color=INK)

    ax.set_xticks(positions, [label for label, _ in errors])
    ax.set_ylim(0, 10.5)
    ax.set_ylabel("Test images")
    ax.set_title("Test errors by type (96 Defect, 204 Fresh)")
    ax.grid(axis="x", visible=False)
    ax.legend(loc="upper right")
    fig.tight_layout()
    save(fig, "test_error_counts.png")


def plot_roc(y_true, p_defect):
    fpr, tpr, auc = roc_points(y_true, p_defect)
    fig, ax = plt.subplots(figsize=(5, 4.2))
    ax.plot(fpr, tpr, color=BASELINE_COLOR,
            label=f"Baseline YOLO11n-cls (AUC = {auc:.3f})")
    ax.plot([0, 1], [0, 1], "--", color=MUTED, linewidth=1, label="Chance")
    ax.set_xlabel("False positive rate (Fresh flagged as Defect)")
    ax.set_ylabel("True positive rate (Defect recall)")
    ax.set_title("ROC curve, baseline on test set")
    ax.legend(loc="lower right")
    fig.tight_layout()
    save(fig, "roc_curve_baseline_test.png")


def plot_score_distribution(y_true, p_defect):
    """Baseline Defect probability per true class; most scores sit at 0 or 1."""
    bins = np.linspace(0, 1, 21)
    fig, axes = plt.subplots(2, 1, figsize=(6.4, 4.6), sharex=True)
    for ax, label, value, color in (
        (axes[0], "True Defect", 1, SYNTHETIC_COLOR),
        (axes[1], "True Fresh", 0, BASELINE_COLOR),
    ):
        scores = p_defect[y_true == value]
        ax.hist(scores, bins=bins, color=color,
                edgecolor=SURFACE, linewidth=1)
        ax.axvline(THRESHOLD, color=MUTED, linestyle="--", linewidth=1)
        ax.set_yscale("symlog", linthresh=1)
        ax.set_yticks([0, 1, 10, 100], ["0", "1", "10", "100"])
        ax.set_ylabel("Images (log)")
        ax.set_title(f"{label} (n = {len(scores)})", fontsize=9)
    axes[0].annotate(
        "threshold 0.5", (THRESHOLD, 1), xycoords=("data", "axes fraction"),
        xytext=(4, -10), textcoords="offset points", color=SECONDARY_INK,
        fontsize=8,
    )
    axes[1].set_xlabel("Baseline predicted probability of Defect")
    fig.suptitle("Baseline score distribution, test set",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "score_distribution_baseline_test.png")


def plot_disagreement_confidence():
    """Probability each model gave the true class on the disagreement images."""
    rows = read_rows(PREDICTION_CONFIDENCE)
    rows.sort(key=lambda row: (
        row["actual_class"] != "Defect",
        float(row["baseline_actual_class_probability"]),
    ))
    labels = [
        f"{image_name(row['image_path']).rsplit('.', 1)[0]}"
        for row in rows
    ]
    baseline = [float(row["baseline_actual_class_probability"]) for row in rows]
    synthetic = [
        float(row["experiment_actual_class_probability"]) for row in rows
    ]
    positions = np.arange(len(rows))[::-1]

    fig, ax = plt.subplots(figsize=(7.4, 4.4))
    ax.hlines(positions, baseline, synthetic, color=GRID, linewidth=2, zorder=1)
    ax.scatter(baseline, positions, s=55, color=BASELINE_COLOR,
               edgecolor=SURFACE, linewidth=1.5, label="Baseline", zorder=2)
    ax.scatter(synthetic, positions, s=55, color=SYNTHETIC_COLOR,
               edgecolor=SURFACE, linewidth=1.5, label="Synthetic pilot",
               zorder=2)
    ax.axvline(THRESHOLD, color=MUTED, linestyle="--", linewidth=1)
    ax.text(THRESHOLD, len(rows) - 0.35, " correct →", color=SECONDARY_INK,
            fontsize=8, va="bottom")
    ax.set_yticks(positions, labels)
    ax.set_xlim(-0.03, 1.03)
    ax.set_ylim(-0.6, len(rows) + 0.2)
    ax.set_xlabel("Probability assigned to the true class")
    ax.set_title(
        f"The {len(rows)} test images where the two models disagree"
    )
    ax.grid(axis="y", visible=False)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    fig.tight_layout()
    save(fig, "disagreement_confidence.png")


# ------------------------------------------------------------
# 4. RUN
# ------------------------------------------------------------

def main():
    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    baseline_log = read_training_log(BASELINE_RESULTS)
    synthetic_log = read_training_log(SYNTHETIC_RESULTS)
    plot_training_curves(
        baseline_log, "Baseline training (YOLO11n-cls)",
        "training_curves_baseline.png",
    )
    plot_training_curves(
        synthetic_log, "Synthetic-pilot training (YOLO11n-cls)",
        "training_curves_synthetic.png",
    )
    plot_training_comparison(baseline_log, synthetic_log)

    y_true, p_defect, baseline_pred, synthetic_pred = load_test_predictions()
    baseline = confusion(y_true, baseline_pred)
    synthetic = confusion(y_true, synthetic_pred)

    plot_confusion_matrices(baseline, synthetic)
    plot_metric_comparison(baseline, synthetic)
    plot_error_counts(baseline, synthetic)
    plot_roc(y_true, p_defect)
    plot_score_distribution(y_true, p_defect)
    plot_disagreement_confidence()

    print(f"\n{'Metric':18s} {'Baseline':>9s} {'Synthetic':>10s}")
    for key in ("tp", "fn", "fp", "tn"):
        print(f"{key:18s} {baseline[key]:9d} {synthetic[key]:10d}")
    for key in ("accuracy", "defect_precision", "defect_recall", "defect_f1",
                "fresh_precision", "fresh_recall"):
        print(f"{key:18s} {baseline[key]:9.4f} {synthetic[key]:10.4f}")


if __name__ == "__main__":
    main()
