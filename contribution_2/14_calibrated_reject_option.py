"""
FILE: 14_calibrated_reject_option.py

PURPOSE:
    Contribution 2: calibrate the baseline classifier's confidence with
    temperature scaling and add an accept / reject / review rule.

INPUT:
    contribution_2/datasets/tomato/val/{Defect,Fresh}    (fit temperature, pick threshold)
    contribution_2/datasets/tomato/test/{Defect,Fresh}   (report results only)
    Baseline model weights from runs/classify/runs/baseline/weights.

OUTPUT:
    contribution_2/runs/contribution2_reject_option/scores.csv
    contribution_2/runs/contribution2_reject_option/summary.csv
    contribution_2/runs/contribution2_reject_option/reliability_diagram_test.png
    contribution_2/runs/contribution2_reject_option/risk_coverage_test.png
    contribution_2/runs/contribution2_reject_option/decision_outcomes_test.png

IMPORTANT:
    - Inference only; no training occurs and the weights are not changed.
    - The temperature and the review threshold are chosen on the
      validation split. The test split is used once, for reporting.
    - Temperature scaling keeps the ranking of images, so it changes ECE
      and the meaning of a threshold but not the risk-coverage curve.
    - MC-dropout is not possible with these weights (dropout = 0.0), and
      an ensemble needs more trained models; neither is done here.
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

# This script lives in contribution_2/, one level below the project.
SESSION_PATH = Path(__file__).resolve().parent
PROJECT_PATH = SESSION_PATH.parent
DATASET_PATH = SESSION_PATH / "datasets" / "tomato"

WEIGHTS_PATH = (
    PROJECT_PATH / "runs" / "classify" / "runs"
    / "baseline" / "weights" / "best.pt"
)

OUTPUT_FOLDER = SESSION_PATH / "runs" / "contribution2_reject_option"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# Number of equal-width confidence bins between 0.5 and 1.0 for ECE.
ECE_BINS = 10

# A tomato goes to human review when the calibrated confidence is below
# the threshold. The threshold is the smallest one that keeps the error
# rate of automatic decisions at or below this target on validation.
TARGET_RISK = 0.005

BASELINE_COLOR = "#2a78d6"
CALIBRATED_COLOR = "#eb6834"
REVIEW_COLOR = "#eda100"
ERROR_COLOR = "#d03b3b"
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
# 2. COLLECT LOGITS
# ------------------------------------------------------------

def list_split(split_name):
    """Return image paths and labels (1 = Defect, 0 = Fresh) for a split."""
    files = []
    labels = []
    for class_name, label in (("Defect", 1), ("Fresh", 0)):
        class_folder = DATASET_PATH / split_name / class_name
        if not class_folder.is_dir():
            raise FileNotFoundError(
                f"Class folder not found:\n{class_folder}\n"
                "Prepare the split with 01_prepare_data.py or update DATASET_PATH."
            )
        class_files = sorted(
            path for path in class_folder.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        )
        files.extend(class_files)
        labels.extend([label] * len(class_files))
    if not files:
        raise RuntimeError(f"No images found in {DATASET_PATH / split_name}.")
    return files, np.asarray(labels, dtype=int)


def collect_logit_margins(model, files):
    """
    Return logit(Defect) - logit(Fresh) for every image.

    Ultralytics reports softmax probabilities, which saturate at 1.0, so
    the raw logits are captured from the final linear layer instead.
    """
    index_by_name = {name: int(index) for index, name in model.names.items()}
    defect_index = index_by_name["Defect"]
    fresh_index = index_by_name["Fresh"]

    captured = []
    hook = model.model.model[-1].linear.register_forward_hook(
        lambda module, inputs, output: captured.append(
            output.detach().float().cpu().numpy()
        )
    )

    margins = []
    try:
        # One image per call, so the last captured output is that image's.
        for path in files:
            result = model.predict(str(path), verbose=False)[0]
            logits = captured[-1][0].astype(np.float64)
            margin = logits[defect_index] - logits[fresh_index]

            # The captured logits must reproduce the reported probability.
            reported = float(result.probs.data.cpu().numpy()[defect_index])
            if abs(sigmoid(margin) - reported) > 1e-3:
                raise RuntimeError(
                    f"Captured logits do not match the prediction for "
                    f"{result.path}."
                )
            margins.append(margin)
    finally:
        hook.remove()

    if len(margins) != len(files):
        raise RuntimeError(
            f"Model returned {len(margins)} predictions for "
            f"{len(files)} images."
        )
    return np.asarray(margins, dtype=np.float64)


# ------------------------------------------------------------
# 3. CALIBRATION
# ------------------------------------------------------------

def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def negative_log_likelihood(margins, y_true, temperature):
    """Mean binary cross-entropy of sigmoid(margin / T)."""
    scaled = margins / temperature
    signed = np.where(y_true == 1, scaled, -scaled)
    return float(np.mean(np.logaddexp(0.0, -signed)))


def fit_temperature(margins, y_true):
    """Find the temperature that minimises validation NLL (1-D search)."""
    candidates = np.logspace(np.log10(0.05), np.log10(20.0), 4000)
    losses = [
        negative_log_likelihood(margins, y_true, temperature)
        for temperature in candidates
    ]
    return float(candidates[int(np.argmin(losses))])


def confidence_and_correct(p_defect, y_true):
    """Confidence in the predicted class and whether that class is right."""
    y_pred = (p_defect >= 0.5).astype(int)
    confidence = np.where(y_pred == 1, p_defect, 1.0 - p_defect)
    return confidence, y_pred == y_true


def reliability_bins(p_defect, y_true):
    """Per-bin count, mean confidence, and accuracy; ECE over all bins."""
    confidence, correct = confidence_and_correct(p_defect, y_true)
    edges = np.linspace(0.5, 1.0, ECE_BINS + 1)
    bin_index = np.clip(
        np.digitize(confidence, edges[1:-1]), 0, ECE_BINS - 1
    )

    counts = np.zeros(ECE_BINS, dtype=int)
    mean_confidence = np.full(ECE_BINS, np.nan)
    accuracy = np.full(ECE_BINS, np.nan)
    ece = 0.0
    for index in range(ECE_BINS):
        members = bin_index == index
        counts[index] = int(members.sum())
        if counts[index]:
            mean_confidence[index] = confidence[members].mean()
            accuracy[index] = correct[members].mean()
            ece += (
                counts[index] / len(y_true)
                * abs(accuracy[index] - mean_confidence[index])
            )
    return {
        "edges": edges,
        "counts": counts,
        "confidence": mean_confidence,
        "accuracy": accuracy,
        "ece": float(ece),
    }


# ------------------------------------------------------------
# 4. REJECT OPTION
# ------------------------------------------------------------

def risk_coverage(p_defect, y_true):
    """
    Risk and coverage as the review threshold is lowered.

    Coverage is the share of images decided automatically; risk is the
    error rate among those. Images with tied confidence enter together.
    """
    confidence, correct = confidence_and_correct(p_defect, y_true)
    order = np.argsort(-confidence, kind="mergesort")
    sorted_confidence = confidence[order]
    errors = np.cumsum(~correct[order])
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_confidence)),
        len(sorted_confidence) - 1,
    ]
    covered = group_ends + 1
    return {
        "threshold": sorted_confidence[group_ends],
        "coverage": covered / len(y_true),
        "risk": errors[group_ends] / covered,
    }


def choose_threshold(p_defect, y_true):
    """Smallest validation threshold whose automatic error rate meets the target."""
    curve = risk_coverage(p_defect, y_true)
    allowed = np.flatnonzero(curve["risk"] <= TARGET_RISK)
    if not len(allowed):
        raise RuntimeError(
            f"No validation threshold reaches a risk of {TARGET_RISK:.2%}."
        )
    return float(curve["threshold"][allowed[-1]])


def decide(p_defect, threshold):
    """Return 'Defect' (reject), 'Fresh' (accept), or 'Review' per image."""
    confidence = np.maximum(p_defect, 1.0 - p_defect)
    decisions = np.where(p_defect >= 0.5, "Defect", "Fresh")
    return np.where(confidence >= threshold, decisions, "Review")


def summarise_decisions(decisions, y_true):
    actual = np.where(y_true == 1, "Defect", "Fresh")
    automatic = decisions != "Review"
    defect = actual == "Defect"
    fresh = actual == "Fresh"
    errors = int((automatic & (decisions != actual)).sum())
    return {
        "images": len(y_true),
        "automatic": int(automatic.sum()),
        "review": int((~automatic).sum()),
        "coverage": float(automatic.mean()),
        "automatic_errors": errors,
        "risk": errors / max(int(automatic.sum()), 1),
        "defect_rejected": int((defect & (decisions == "Defect")).sum()),
        "defect_review": int((defect & (decisions == "Review")).sum()),
        "defect_missed": int((defect & (decisions == "Fresh")).sum()),
        "fresh_accepted": int((fresh & (decisions == "Fresh")).sum()),
        "fresh_review": int((fresh & (decisions == "Review")).sum()),
        "fresh_false_alarm": int((fresh & (decisions == "Defect")).sum()),
    }


# ------------------------------------------------------------
# 5. PLOTS
# ------------------------------------------------------------

def save(fig, name):
    path = OUTPUT_FOLDER / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_PATH)}")


def plot_reliability(before, after, temperature):
    """Accuracy against confidence per bin, before and after scaling."""
    fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
    panels = (
        (axes[0], before, BASELINE_COLOR, "Before scaling (T = 1)"),
        (axes[1], after, CALIBRATED_COLOR,
         f"After temperature scaling (T = {temperature:.2f})"),
    )
    for ax, bins, color, title in panels:
        edges = bins["edges"]
        width = edges[1] - edges[0]
        filled = bins["counts"] > 0
        ax.bar(
            edges[:-1][filled] + width / 2, bins["accuracy"][filled],
            width=width - 0.004, color=color,
        )
        ax.plot([0.5, 1], [0.5, 1], "--", color=MUTED, linewidth=1,
                label="Perfect calibration")
        for left, count, accuracy in zip(
            edges[:-1], bins["counts"], bins["accuracy"]
        ):
            if count:
                ax.text(left + width / 2, accuracy + 0.015, f"n={count}",
                        ha="center", fontsize=7, color=SECONDARY_INK)
        ax.set_title(f"{title}\nECE = {100 * bins['ece']:.2f}%", fontsize=10)
        ax.set_xlabel("Confidence in predicted class")
        ax.set_xlim(0.5, 1.0)
        ax.set_ylim(0, 1.12)
        ax.set_yticks(np.linspace(0, 1, 6))
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Accuracy in bin")
    axes[0].legend(loc="upper left")
    fig.suptitle("Reliability diagram, baseline on test set (300 images)",
                 x=0.01, ha="left", fontsize=12, fontweight="bold")
    fig.tight_layout()
    save(fig, "reliability_diagram_test.png")


def plot_risk_coverage(curve, summary):
    """Error rate of automatic decisions against the share decided."""
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    ax.step(100 * curve["coverage"], 100 * curve["risk"], where="post",
            color=BASELINE_COLOR, label="Baseline with review option")

    full_risk = 100 * curve["risk"][-1]
    ax.scatter(100, full_risk, s=60, color=MUTED, edgecolor=SURFACE,
               linewidth=2, zorder=3)
    ax.annotate(
        f"No review: {full_risk:.2f}% error", (100, full_risk),
        xytext=(-8, 6), textcoords="offset points", ha="right", color=INK,
    )

    point = (100 * summary["coverage"], 100 * summary["risk"])
    ax.scatter(*point, s=60, color=CALIBRATED_COLOR, edgecolor=SURFACE,
               linewidth=2, zorder=3)
    ax.annotate(
        f"Chosen threshold: {point[1]:.2f}% error\n"
        f"at {point[0]:.1f}% coverage",
        point, xytext=(-10, 14), textcoords="offset points", ha="right",
        color=INK,
    )

    ax.set_xlabel("Coverage: test images decided automatically (%)")
    ax.set_ylabel("Risk: error rate of automatic decisions (%)")
    ax.set_xlim(40, 101.5)
    ax.set_ylim(-0.15, max(4.0, full_risk + 1))
    ax.set_title("Risk-coverage curve, test set")
    ax.legend(loc="upper left")
    fig.tight_layout()
    save(fig, "risk_coverage_test.png")


def plot_decision_outcomes(no_review, with_review):
    """Where the true Defect and true Fresh test images end up."""
    rows = (
        ("Defect, with review", with_review, "defect"),
        ("Defect, no review", no_review, "defect"),
        ("Fresh, with review", with_review, "fresh"),
        ("Fresh, no review", no_review, "fresh"),
    )
    fig, ax = plt.subplots(figsize=(8, 3.4))
    for position, (label, summary, group) in enumerate(rows[::-1]):
        if group == "defect":
            parts = (
                (summary["defect_rejected"], BASELINE_COLOR),
                (summary["defect_review"], REVIEW_COLOR),
                (summary["defect_missed"], ERROR_COLOR),
            )
        else:
            parts = (
                (summary["fresh_accepted"], BASELINE_COLOR),
                (summary["fresh_review"], REVIEW_COLOR),
                (summary["fresh_false_alarm"], ERROR_COLOR),
            )
        total = sum(count for count, _ in parts)
        left = 0.0
        for count, color in parts:
            share = 100 * count / total
            ax.barh(position, share, left=left, color=color, height=0.62,
                    edgecolor=SURFACE, linewidth=1.5)
            left += share
        ax.text(
            101, position,
            " / ".join(str(count) for count, _ in parts),
            va="center", color=INK,
        )
    ax.set_yticks(range(len(rows)), [row[0] for row in rows[::-1]])
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of test images of that true class (%)")
    ax.set_title("Test outcomes with and without the review option")
    ax.grid(axis="y", visible=False)
    ax.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, color=BASELINE_COLOR),
            plt.Rectangle((0, 0), 1, 1, color=REVIEW_COLOR),
            plt.Rectangle((0, 0), 1, 1, color=ERROR_COLOR),
        ],
        labels=["Correct automatic decision", "Sent to human review",
                "Wrong automatic decision"],
        loc="upper center", bbox_to_anchor=(0.5, -0.2), ncols=3,
    )
    ax.text(101, len(rows) - 0.45, "correct / review / wrong",
            color=SECONDARY_INK, fontsize=7.5)
    fig.tight_layout()
    save(fig, "decision_outcomes_test.png")


# ------------------------------------------------------------
# 6. RUN
# ------------------------------------------------------------

def main():
    if not WEIGHTS_PATH.is_file():
        raise FileNotFoundError(
            f"Baseline weights not found:\n{WEIGHTS_PATH}\n"
            "Run 02_train_baseline.py first or update WEIGHTS_PATH."
        )

    model = YOLO(str(WEIGHTS_PATH))
    val_files, val_true = list_split("val")
    test_files, test_true = list_split("test")
    print(f"Validation images: {len(val_files)} | Test images: {len(test_files)}")

    val_margin = collect_logit_margins(model, val_files)
    test_margin = collect_logit_margins(model, test_files)

    # Fit on validation only.
    temperature = fit_temperature(val_margin, val_true)
    threshold = choose_threshold(sigmoid(val_margin / temperature), val_true)

    val_raw = sigmoid(val_margin)
    val_scaled = sigmoid(val_margin / temperature)
    test_raw = sigmoid(test_margin)
    test_scaled = sigmoid(test_margin / temperature)

    test_before = reliability_bins(test_raw, test_true)
    test_after = reliability_bins(test_scaled, test_true)
    test_decisions = decide(test_scaled, threshold)
    with_review = summarise_decisions(test_decisions, test_true)
    no_review = summarise_decisions(decide(test_scaled, 0.5), test_true)
    val_with_review = summarise_decisions(
        decide(val_scaled, threshold), val_true
    )

    OUTPUT_FOLDER.mkdir(parents=True, exist_ok=True)

    with (OUTPUT_FOLDER / "scores.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "split", "image", "actual", "logit_margin",
            "p_defect_raw", "p_defect_calibrated", "decision",
        ])
        for split_name, files, y_true, margin, raw, scaled in (
            ("val", val_files, val_true, val_margin, val_raw, val_scaled),
            ("test", test_files, test_true, test_margin, test_raw,
             test_scaled),
        ):
            decisions = decide(scaled, threshold)
            for row in zip(files, y_true, margin, raw, scaled, decisions):
                writer.writerow([
                    split_name, row[0].name,
                    "Defect" if row[1] else "Fresh",
                    f"{row[2]:.6f}", f"{row[3]:.6f}", f"{row[4]:.6f}", row[5],
                ])

    summary_rows = [
        ("temperature", f"{temperature:.4f}"),
        ("review_threshold", f"{threshold:.4f}"),
        ("target_risk_on_validation", f"{TARGET_RISK:.4f}"),
        ("val_nll_before",
         f"{negative_log_likelihood(val_margin, val_true, 1.0):.4f}"),
        ("val_nll_after",
         f"{negative_log_likelihood(val_margin, val_true, temperature):.4f}"),
        ("test_nll_before",
         f"{negative_log_likelihood(test_margin, test_true, 1.0):.4f}"),
        ("test_nll_after",
         f"{negative_log_likelihood(test_margin, test_true, temperature):.4f}"),
        ("val_ece_before", f"{reliability_bins(val_raw, val_true)['ece']:.4f}"),
        ("val_ece_after",
         f"{reliability_bins(val_scaled, val_true)['ece']:.4f}"),
        ("test_ece_before", f"{test_before['ece']:.4f}"),
        ("test_ece_after", f"{test_after['ece']:.4f}"),
    ]
    for prefix, summary in (
        ("val_review", val_with_review),
        ("test_no_review", no_review),
        ("test_review", with_review),
    ):
        for key, value in summary.items():
            text = f"{value:.4f}" if isinstance(value, float) else str(value)
            summary_rows.append((f"{prefix}_{key}", text))

    with (OUTPUT_FOLDER / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)

    plot_reliability(test_before, test_after, temperature)
    plot_risk_coverage(risk_coverage(test_scaled, test_true), with_review)
    plot_decision_outcomes(no_review, with_review)

    print()
    for metric, value in summary_rows:
        print(f"{metric:34s} {value}")
    print(f"\nScores, summary, and plots saved to:\n{OUTPUT_FOLDER}")


if __name__ == "__main__":
    main()
