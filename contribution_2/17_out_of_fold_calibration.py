"""
FILE: 17_out_of_fold_calibration.py

PURPOSE:
    Calibrate on out-of-fold predictions and evaluate the review option
    for the baseline and for the five-model fold ensemble.

INPUT:
    contribution_2/datasets/tomato/{train,test}/{Defect,Fresh}
    contribution_2/datasets/tomato_folds/fold_assignment.csv
    contribution_2/runs/calibration_folds/fold_<k>/weights/best.pt
    contribution_2/runs/contribution2_reject_option/scores.csv
        (baseline test logits saved by 14_calibrated_reject_option.py)

OUTPUT:
    contribution_2/runs/out_of_fold_calibration/oof_scores.csv
    contribution_2/runs/out_of_fold_calibration/test_scores.csv
    contribution_2/runs/out_of_fold_calibration/summary.csv
    contribution_2/runs/out_of_fold_calibration/reliability_diagram_test.png
    contribution_2/runs/out_of_fold_calibration/risk_coverage_test.png

IMPORTANT:
    - Inference only; no training occurs.
    - The temperature and review threshold are fitted on out-of-fold
      predictions: each training image is scored by the one fold model
      that never saw it. The test split is used once, for reporting.
    - The out-of-fold fit describes single fold models. Applying it to
      the baseline (trained on all training images) and to the ensemble
      is an approximation, because neither has out-of-fold predictions.
    - The ensemble averages the five fold models' logit margins.
"""

from pathlib import Path
import csv
import importlib.util

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from ultralytics import YOLO

# ------------------------------------------------------------
# 1. PATHS AND SETTINGS
# ------------------------------------------------------------

# This script lives in contribution_2/, one level below the project.
SESSION_PATH = Path(__file__).resolve().parent
PROJECT_PATH = SESSION_PATH.parent

DATASET_PATH = SESSION_PATH / "datasets" / "tomato"
FOLD_ASSIGNMENT = (
    SESSION_PATH / "datasets" / "tomato_folds" / "fold_assignment.csv"
)
FOLD_RUNS = SESSION_PATH / "runs" / "calibration_folds"
BASELINE_SCORES = (
    SESSION_PATH / "runs" / "contribution2_reject_option" / "scores.csv"
)
OUTPUT_FOLDER = SESSION_PATH / "runs" / "out_of_fold_calibration"

# Reuse the calibration and reject-option functions from script 14, so
# both scripts compute ECE, thresholds, and decisions the same way.
_spec = importlib.util.spec_from_file_location(
    "reject_option", SESSION_PATH / "14_calibrated_reject_option.py"
)
reject_option = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(reject_option)

sigmoid = reject_option.sigmoid
BASELINE_COLOR = reject_option.BASELINE_COLOR
ENSEMBLE_COLOR = reject_option.CALIBRATED_COLOR
INK = reject_option.INK
SECONDARY_INK = reject_option.SECONDARY_INK
MUTED = reject_option.MUTED
SURFACE = reject_option.SURFACE


# ------------------------------------------------------------
# 2. LOAD INPUTS
# ------------------------------------------------------------

def load_fold_assignment():
    """Return {fold: (files, labels)} for the held-out training images."""
    if not FOLD_ASSIGNMENT.is_file():
        raise FileNotFoundError(
            f"Fold assignment not found:\n{FOLD_ASSIGNMENT}\n"
            "Run 16_train_calibration_folds.py first."
        )
    folds = {}
    with FOLD_ASSIGNMENT.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            files, labels = folds.setdefault(
                int(row["held_out_fold"]), ([], [])
            )
            files.append(DATASET_PATH / "train" / row["class"] / row["image"])
            labels.append(int(row["class"] == "Defect"))
    return {
        fold: (files, np.asarray(labels, dtype=int))
        for fold, (files, labels) in sorted(folds.items())
    }


def load_baseline_test_margins(test_files):
    """Baseline test logit margins saved by script 14, in test_files order."""
    if not BASELINE_SCORES.is_file():
        raise FileNotFoundError(
            f"Baseline scores not found:\n{BASELINE_SCORES}\n"
            "Run 14_calibrated_reject_option.py first."
        )
    margin_by_name = {}
    with BASELINE_SCORES.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row["split"] == "test":
                margin_by_name[row["image"]] = float(row["logit_margin"])
    return np.array([margin_by_name[path.name] for path in test_files])


# ------------------------------------------------------------
# 3. EVALUATION HELPERS
# ------------------------------------------------------------

def evaluate_system(margins, y_true, temperature, threshold):
    """Calibration and review-option results for one model on one split."""
    raw = sigmoid(margins)
    scaled = sigmoid(margins / temperature)
    predictions = (margins >= 0).astype(int)
    return {
        "errors": int((predictions != y_true).sum()),
        "accuracy": float((predictions == y_true).mean()),
        "nll_before": reject_option.negative_log_likelihood(
            margins, y_true, 1.0
        ),
        "nll_after": reject_option.negative_log_likelihood(
            margins, y_true, temperature
        ),
        "bins_before": reject_option.reliability_bins(raw, y_true),
        "bins_after": reject_option.reliability_bins(scaled, y_true),
        "curve": reject_option.risk_coverage(scaled, y_true),
        "review": reject_option.summarise_decisions(
            reject_option.decide(scaled, threshold), y_true
        ),
        "decisions": reject_option.decide(scaled, threshold),
        "p_defect_calibrated": scaled,
    }


# ------------------------------------------------------------
# 4. PLOTS
# ------------------------------------------------------------

def save(fig, name):
    path = OUTPUT_FOLDER / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_PATH)}")


def plot_reliability(baseline, ensemble, temperature):
    panels = (
        ("Baseline, before scaling", baseline["bins_before"], BASELINE_COLOR),
        (f"Baseline, T = {temperature:.2f}", baseline["bins_after"],
         BASELINE_COLOR),
        ("Ensemble, before scaling", ensemble["bins_before"], ENSEMBLE_COLOR),
        (f"Ensemble, T = {temperature:.2f}", ensemble["bins_after"],
         ENSEMBLE_COLOR),
    )
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.8), sharey=True)
    for ax, (title, bins, color) in zip(axes, panels):
        edges = bins["edges"]
        width = edges[1] - edges[0]
        filled = bins["counts"] > 0
        ax.bar(edges[:-1][filled] + width / 2, bins["accuracy"][filled],
               width=width - 0.004, color=color)
        ax.plot([0.5, 1], [0.5, 1], "--", color=MUTED, linewidth=1,
                label="Perfect calibration")
        for left, count, accuracy in zip(
            edges[:-1], bins["counts"], bins["accuracy"]
        ):
            if count:
                ax.text(left + width / 2, accuracy + 0.015, f"n={count}",
                        ha="center", fontsize=6.5, color=SECONDARY_INK)
        ax.set_title(f"{title}\nECE = {100 * bins['ece']:.2f}%", fontsize=10)
        ax.set_xlabel("Confidence in predicted class")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Accuracy in bin")
    axes[0].legend(loc="upper left", fontsize=7.5)
    fig.suptitle(
        "Reliability on the test set, temperature fitted on out-of-fold "
        "predictions",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "reliability_diagram_test.png")


def plot_risk_coverage(baseline, ensemble):
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    for label, system, color in (
        ("Baseline", baseline, BASELINE_COLOR),
        ("Five-fold ensemble", ensemble, ENSEMBLE_COLOR),
    ):
        curve = system["curve"]
        ax.step(100 * curve["coverage"], 100 * curve["risk"], where="post",
                color=color, label=label)
        review = system["review"]
        ax.scatter(100 * review["coverage"], 100 * review["risk"], s=60,
                   color=color, edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.scatter([], [], s=40, color=MUTED, edgecolor=SURFACE,
               label="Operating point at the out-of-fold threshold")
    ax.set_xlabel("Coverage: test images decided automatically (%)")
    ax.set_ylabel("Risk: error rate of automatic decisions (%)")
    ax.set_xlim(40, 101.5)
    ax.set_ylim(-0.15, 4.5)
    ax.set_title("Risk-coverage curves, test set")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "risk_coverage_test.png")


# ------------------------------------------------------------
# 5. RUN
# ------------------------------------------------------------

def main():
    folds = load_fold_assignment()
    test_files, test_true = reject_option.list_split("test")

    # Score each fold's held-out training images and the test split.
    oof_rows = []
    fold_test_margins = []
    for fold, (files, labels) in folds.items():
        weights = FOLD_RUNS / f"fold_{fold}" / "weights" / "best.pt"
        if not weights.is_file():
            raise FileNotFoundError(
                f"Fold weights not found:\n{weights}\n"
                "Run 16_train_calibration_folds.py first."
            )
        model = YOLO(str(weights))
        margins = reject_option.collect_logit_margins(model, files)
        oof_rows.extend(zip([fold] * len(files), files, labels, margins))
        fold_test_margins.append(
            reject_option.collect_logit_margins(model, test_files)
        )
        print(f"Fold {fold}: scored {len(files)} held-out and "
              f"{len(test_files)} test images")

    oof_margin = np.array([row[3] for row in oof_rows])
    oof_true = np.array([row[2] for row in oof_rows], dtype=int)
    fold_test_margins = np.array(fold_test_margins)

    # Fit on out-of-fold predictions only.
    temperature = reject_option.fit_temperature(oof_margin, oof_true)
    threshold = reject_option.choose_threshold(
        sigmoid(oof_margin / temperature), oof_true
    )

    oof = evaluate_system(oof_margin, oof_true, temperature, threshold)
    baseline = evaluate_system(
        load_baseline_test_margins(test_files), test_true,
        temperature, threshold,
    )
    ensemble_margin = fold_test_margins.mean(axis=0)
    ensemble = evaluate_system(
        ensemble_margin, test_true, temperature, threshold
    )
    fold_test_errors = [
        int(((margins >= 0).astype(int) != test_true).sum())
        for margins in fold_test_margins
    ]

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_FOLDER / "oof_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "held_out_fold", "image", "actual", "logit_margin",
            "p_defect_calibrated", "decision",
        ])
        for row, scaled, decision in zip(
            oof_rows, oof["p_defect_calibrated"], oof["decisions"]
        ):
            writer.writerow([
                row[0], row[1].name, "Defect" if row[2] else "Fresh",
                f"{row[3]:.6f}", f"{scaled:.6f}", decision,
            ])

    with (OUTPUT_FOLDER / "test_scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(
            ["image", "actual"]
            + [f"fold_{fold}_logit_margin" for fold in folds]
            + ["ensemble_logit_margin", "ensemble_p_defect_calibrated",
               "ensemble_decision", "baseline_decision"]
        )
        for index, path in enumerate(test_files):
            writer.writerow(
                [path.name, "Defect" if test_true[index] else "Fresh"]
                + [f"{value:.6f}" for value in fold_test_margins[:, index]]
                + [f"{ensemble_margin[index]:.6f}",
                   f"{ensemble['p_defect_calibrated'][index]:.6f}",
                   ensemble["decisions"][index],
                   baseline["decisions"][index]]
            )

    summary_rows = [
        ("temperature_from_oof", f"{temperature:.4f}"),
        ("review_threshold_from_oof", f"{threshold:.4f}"),
        ("target_risk_on_oof", f"{reject_option.TARGET_RISK:.4f}"),
        ("fold_models_test_errors",
         " ".join(str(count) for count in fold_test_errors)),
    ]
    for prefix, system in (
        ("oof", oof), ("test_baseline", baseline), ("test_ensemble", ensemble),
    ):
        summary_rows += [
            (f"{prefix}_errors", str(system["errors"])),
            (f"{prefix}_accuracy", f"{system['accuracy']:.4f}"),
            (f"{prefix}_nll_before", f"{system['nll_before']:.4f}"),
            (f"{prefix}_nll_after", f"{system['nll_after']:.4f}"),
            (f"{prefix}_ece_before", f"{system['bins_before']['ece']:.4f}"),
            (f"{prefix}_ece_after", f"{system['bins_after']['ece']:.4f}"),
        ]
        for key, value in system["review"].items():
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            summary_rows.append((f"{prefix}_review_{key}", text))

    with (OUTPUT_FOLDER / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)

    plot_reliability(baseline, ensemble, temperature)
    plot_risk_coverage(baseline, ensemble)

    print()
    for metric, value in summary_rows:
        print(f"{metric:40s} {value}")
    print(f"\nScores, summary, and plots saved to:\n{OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
