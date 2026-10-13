"""
FILE: 15_calibration_stability.py

PURPOSE:
    Check whether the temperature and review threshold from
    14_calibrated_reject_option.py are stable, using cross-validated
    calibration inside the validation split.

INPUT:
    contribution_2/runs/contribution2_reject_option/scores.csv

OUTPUT:
    contribution_2/runs/contribution2_reject_option/stability_summary.csv
    contribution_2/runs/contribution2_reject_option/temperature_stability_val.png
    contribution_2/runs/contribution2_reject_option/temperature_sweep_test.png

IMPORTANT:
    - No training and no model inference occur; only saved logits are read.
    - Each repeat splits validation into two stratified halves, fits on
      one half, and scores the other half, which the fit never saw.
    - The test sweep is a diagnostic only. Nothing is chosen from it.
"""

from pathlib import Path
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
OUTPUT_FOLDER = SESSION_PATH / "runs" / "contribution2_reject_option"
SCORES_PATH = OUTPUT_FOLDER / "scores.csv"

REPEATS = 500
SEED = 42

# Same settings as 14_calibrated_reject_option.py.
ECE_BINS = 10
TARGET_RISK = 0.005
TEMPERATURES = np.logspace(np.log10(0.05), np.log10(20.0), 4000)

BASELINE_COLOR = "#2a78d6"
CALIBRATED_COLOR = "#eb6834"
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
    "axes.titlesize": 10,
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
# 2. LOAD SAVED LOGITS
# ------------------------------------------------------------

def load_scores():
    if not SCORES_PATH.is_file():
        raise FileNotFoundError(
            f"Saved scores not found:\n{SCORES_PATH}\n"
            "Run 14_calibrated_reject_option.py first."
        )
    splits = {"val": ([], []), "test": ([], [])}
    with SCORES_PATH.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            margins, labels = splits[row["split"]]
            margins.append(float(row["logit_margin"]))
            labels.append(int(row["actual"] == "Defect"))
    return {
        name: (np.asarray(margins), np.asarray(labels, dtype=int))
        for name, (margins, labels) in splits.items()
    }


# ------------------------------------------------------------
# 3. CALIBRATION AND REJECT OPTION (as in script 14)
# ------------------------------------------------------------

def sigmoid(value):
    return 1.0 / (1.0 + np.exp(-value))


def nll_per_temperature(margins, y_true, temperatures):
    """Mean binary cross-entropy for every temperature at once."""
    signed = np.where(y_true == 1, margins, -margins)
    return np.logaddexp(
        0.0, -signed[None, :] / np.asarray(temperatures)[:, None]
    ).mean(axis=1)


def fit_temperature(margins, y_true):
    losses = nll_per_temperature(margins, y_true, TEMPERATURES)
    return float(TEMPERATURES[int(np.argmin(losses))])


def nll(margins, y_true, temperature):
    return float(nll_per_temperature(margins, y_true, [temperature])[0])


def confidence_and_correct(p_defect, y_true):
    y_pred = (p_defect >= 0.5).astype(int)
    confidence = np.where(y_pred == 1, p_defect, 1.0 - p_defect)
    return confidence, y_pred == y_true


def ece(margins, y_true, temperature):
    confidence, correct = confidence_and_correct(
        sigmoid(margins / temperature), y_true
    )
    edges = np.linspace(0.5, 1.0, ECE_BINS + 1)
    bin_index = np.clip(np.digitize(confidence, edges[1:-1]), 0, ECE_BINS - 1)
    total = 0.0
    for index in range(ECE_BINS):
        members = bin_index == index
        if members.any():
            total += members.mean() * abs(
                correct[members].mean() - confidence[members].mean()
            )
    return float(total)


def choose_threshold(p_defect, y_true):
    """Smallest threshold whose automatic error rate meets the target."""
    confidence, correct = confidence_and_correct(p_defect, y_true)
    order = np.argsort(-confidence, kind="mergesort")
    sorted_confidence = confidence[order]
    errors = np.cumsum(~correct[order])
    group_ends = np.r_[
        np.flatnonzero(np.diff(sorted_confidence)),
        len(sorted_confidence) - 1,
    ]
    risk = errors[group_ends] / (group_ends + 1)
    allowed = np.flatnonzero(risk <= TARGET_RISK)
    if not len(allowed):
        return 1.0
    return float(sorted_confidence[group_ends][allowed[-1]])


def review_outcome(p_defect, y_true, threshold):
    """Coverage and automatic error rate at a review threshold."""
    confidence, correct = confidence_and_correct(p_defect, y_true)
    automatic = confidence >= threshold
    covered = int(automatic.sum())
    risk = float((~correct[automatic]).mean()) if covered else 0.0
    return covered / len(y_true), risk


# ------------------------------------------------------------
# 4. CROSS-VALIDATED CALIBRATION
# ------------------------------------------------------------

def stratified_halves(y_true, rng):
    """Return a boolean mask that puts half of each class in the fit half."""
    fit_half = np.zeros(len(y_true), dtype=bool)
    for label in (0, 1):
        members = rng.permutation(np.flatnonzero(y_true == label))
        fit_half[members[: len(members) // 2]] = True
    return fit_half


def run_repeats(margins, y_true):
    rng = np.random.default_rng(SEED)
    rows = []
    for _ in range(REPEATS):
        fit_half = stratified_halves(y_true, rng)
        held_out = ~fit_half
        temperature = fit_temperature(margins[fit_half], y_true[fit_half])
        threshold = choose_threshold(
            sigmoid(margins[fit_half] / temperature), y_true[fit_half]
        )
        coverage, risk = review_outcome(
            sigmoid(margins[held_out] / temperature),
            y_true[held_out], threshold,
        )
        rows.append({
            "temperature": temperature,
            "ece_before": ece(margins[held_out], y_true[held_out], 1.0),
            "ece_after": ece(margins[held_out], y_true[held_out], temperature),
            "nll_before": nll(margins[held_out], y_true[held_out], 1.0),
            "nll_after": nll(margins[held_out], y_true[held_out], temperature),
            "threshold": threshold,
            "coverage": coverage,
            "risk": risk,
        })
    return {key: np.array([row[key] for row in rows]) for key in rows[0]}


# ------------------------------------------------------------
# 5. PLOTS
# ------------------------------------------------------------

def save(fig, name):
    path = OUTPUT_FOLDER / name
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {path.relative_to(PROJECT_PATH)}")


def plot_stability(repeats, full_temperature):
    """Spread of the fitted temperature and its effect on unseen halves."""
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6))

    ax = axes[0]
    ax.hist(repeats["temperature"], bins=30, color=BASELINE_COLOR,
            edgecolor=SURFACE, linewidth=0.8)
    ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1)
    ax.axvline(full_temperature, color=CALIBRATED_COLOR, linewidth=1.5)
    ax.text(1.0, 0.97, " T = 1\n (no scaling)", color=SECONDARY_INK,
            fontsize=7.5, va="top", transform=ax.get_xaxis_transform())
    ax.text(full_temperature, 0.82, f" full val\n T = {full_temperature:.2f}",
            color=SECONDARY_INK, fontsize=7.5, va="top",
            transform=ax.get_xaxis_transform())
    ax.set_title("Temperature fitted on half of validation")
    ax.set_xlabel("Fitted temperature")
    ax.set_ylabel(f"Splits (of {REPEATS})")

    for ax, key, label, scale in (
        (axes[1], "nll", "NLL", 1.0),
        (axes[2], "ece", "ECE (percentage points)", 100.0),
    ):
        change = scale * (repeats[f"{key}_after"] - repeats[f"{key}_before"])
        improved = 100 * float((change < 0).mean())
        ax.hist(change, bins=30, color=BASELINE_COLOR,
                edgecolor=SURFACE, linewidth=0.8)
        ax.axvline(0.0, color=MUTED, linestyle="--", linewidth=1)
        ax.set_title(
            f"Change in held-out {label.split(' ')[0]}\n"
            f"better in {improved:.0f}% of splits",
        )
        ax.set_xlabel(f"{label}: after − before (below 0 is better)")

    fig.suptitle(
        "Cross-validated temperature scaling inside the validation split",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "temperature_stability_val.png")


def plot_test_sweep(test_margin, test_true, repeats, full_temperature):
    """Diagnostic: test NLL and ECE for every temperature."""
    temperatures = np.logspace(np.log10(0.3), np.log10(4.0), 300)
    low, high = np.percentile(repeats["temperature"], [2.5, 97.5])
    curves = (
        ("Test NLL", nll_per_temperature(test_margin, test_true, temperatures)),
        ("Test ECE (%)", np.array([
            100 * ece(test_margin, test_true, temperature)
            for temperature in temperatures
        ])),
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.7))
    for ax, (title, values) in zip(axes, curves):
        ax.axvspan(low, high, color=CALIBRATED_COLOR, alpha=0.15, linewidth=0,
                   label="95% of validation-half fits")
        ax.plot(temperatures, values, color=BASELINE_COLOR)
        ax.axvline(1.0, color=MUTED, linestyle="--", linewidth=1,
                   label="T = 1 (no scaling)")
        ax.axvline(full_temperature, color=CALIBRATED_COLOR, linewidth=1.5,
                   label=f"Full-validation fit, T = {full_temperature:.2f}")
        best = temperatures[int(np.argmin(values))]
        ax.scatter(best, values.min(), s=50, color=BASELINE_COLOR,
                   edgecolor=SURFACE, linewidth=2, zorder=3)
        ax.annotate(f"lowest at T = {best:.2f}", (best, values.min()),
                    xytext=(8, 10), textcoords="offset points", color=INK)
        ax.set_xscale("log")
        ax.set_xticks([0.3, 0.5, 1, 2, 4], ["0.3", "0.5", "1", "2", "4"])
        ax.set_xlim(temperatures[0], temperatures[-1])
        ax.minorticks_off()
        ax.set_xlabel("Temperature")
        ax.set_title(title)
    axes[1].legend(loc="upper center", fontsize=7.5)
    fig.suptitle(
        "Diagnostic only: what temperature the test set would have needed",
        x=0.01, ha="left", fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    save(fig, "temperature_sweep_test.png")


# ------------------------------------------------------------
# 6. RUN
# ------------------------------------------------------------

def main():
    scores = load_scores()
    val_margin, val_true = scores["val"]
    test_margin, test_true = scores["test"]

    full_temperature = fit_temperature(val_margin, val_true)
    repeats = run_repeats(val_margin, val_true)

    plot_stability(repeats, full_temperature)
    plot_test_sweep(test_margin, test_true, repeats, full_temperature)

    def spread(values, scale=1.0):
        low, median, high = np.percentile(scale * values, [2.5, 50, 97.5])
        return f"{median:.4f} ({low:.4f} to {high:.4f})"

    nll_change = repeats["nll_after"] - repeats["nll_before"]
    ece_change = repeats["ece_after"] - repeats["ece_before"]
    summary_rows = [
        ("repeats", str(REPEATS)),
        ("full_validation_temperature", f"{full_temperature:.4f}"),
        ("temperature_median_95pct", spread(repeats["temperature"])),
        ("share_of_fits_below_1",
         f"{float((repeats['temperature'] < 1).mean()):.4f}"),
        ("held_out_nll_before_median_95pct", spread(repeats["nll_before"])),
        ("held_out_nll_after_median_95pct", spread(repeats["nll_after"])),
        ("held_out_nll_change_median_95pct", spread(nll_change)),
        ("share_of_splits_nll_improved",
         f"{float((nll_change < 0).mean()):.4f}"),
        ("held_out_ece_before_median_95pct", spread(repeats["ece_before"])),
        ("held_out_ece_after_median_95pct", spread(repeats["ece_after"])),
        ("held_out_ece_change_median_95pct", spread(ece_change)),
        ("share_of_splits_ece_improved",
         f"{float((ece_change < 0).mean()):.4f}"),
        ("review_threshold_median_95pct", spread(repeats["threshold"])),
        ("held_out_coverage_median_95pct", spread(repeats["coverage"])),
        ("held_out_risk_median_95pct", spread(repeats["risk"])),
        ("share_of_splits_meeting_target_risk",
         f"{float((repeats['risk'] <= TARGET_RISK).mean()):.4f}"),
    ]

    with (OUTPUT_FOLDER / "stability_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])
        writer.writerows(summary_rows)

    print()
    for metric, value in summary_rows:
        print(f"{metric:38s} {value}")


if __name__ == "__main__":
    main()
