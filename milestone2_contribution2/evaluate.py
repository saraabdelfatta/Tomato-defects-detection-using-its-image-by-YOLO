"""
Final step: build the metrics table and plots A-D from cached predictions.

    python evaluate.py        (seconds; no model is loaded)

Thresholds come from results/step4_threshold.json and step5_ensemble.json
(chosen on validation). Test is only used here, once, for reporting.

Outputs:
    results/metrics.csv        test metrics, one row per step (+ MobileNetV2 reference)
    results/metrics_val.csv    same rows on validation, for transparency
    plots/A_accuracy_defect_recall.png
    plots/B_confusion_baseline_vs_best.png
    plots/C_ablation_recalls.png
    plots/D_threshold_sweep_val.png
"""

import csv

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import c2_common as C

# Reference palette (dataviz skill), light mode.
BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e6e5e1"
DPI = 200

plt.rcParams.update({
    "font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "axes.grid.axis": "y",
    "grid.color": GRID, "grid.linewidth": 0.8, "axes.axisbelow": True,
    "axes.titleweight": "bold", "axes.titlesize": 11, "legend.frameon": False,
})

COLS = ["step", "model", "role", "threshold", "accuracy",
        "defect_precision", "defect_recall", "defect_f1",
        "fresh_precision", "fresh_recall", "fresh_f1", "roc_auc", "tp", "fn", "fp", "tn"]


def build_rows(split):
    s4 = C.read_json("step4_threshold.json")
    s5 = C.read_json("step5_ensemble.json")
    s6 = C.read_json("step6_synthetic.json") or {"status": "skipped"}
    if s4 is None or s5 is None:
        raise SystemExit("Run step4_threshold.py and step5_ensemble.py first.")
    loss_label = {"weighted": "Weighted CE", "focal": "Focal (gamma=2)"}

    # (step, short label, model, prediction name, threshold, role)
    spec = [
        ("1", "Baseline", "Baseline YOLO11n-cls (M1)", "baseline", 0.5, "main"),
        ("2", "Weighted CE", "Class-weighted cross-entropy", "weighted", 0.5, "main"),
        ("3", "Focal", "Class-weighted focal loss (gamma=2)", "focal", 0.5, "main"),
        ("4", f"{loss_label[s4['best_model']]}\n+ threshold",
         f"{loss_label[s4['best_model']]} + val threshold", s4["best_model"], s4["threshold"], "main"),
        ("5", "Ensemble\n+ threshold", "5-fold ensemble (mean prob) + val threshold",
         "ensemble", s5["threshold"], "main"),
    ]
    if s6.get("status") == "done":
        spec.append(("6", "Synthetic\n+ best C2", "Synthetic model + best C2 setup",
                     "synthetic", s6["threshold"], "main"))
    spec += [
        ("1b", "Baseline\n+ threshold", "Baseline + val threshold (supplementary)", "baseline",
         s4["baseline_supplementary"]["threshold"], "supplementary"),
        ("5b", "Ensemble\n@ 0.5", "5-fold ensemble at 0.5 (supplementary)", "ensemble", 0.5, "supplementary"),
    ]

    rows = []
    if split == "test":
        rows.append(dict(C.MOBILENET_ROW, label="MobileNetV2\n(reference)", role="reference",
                         tp="", fn="", fp="", tn=""))
    for step, label, model, name, t, role in spec:
        d = C.load_predictions(name)[split]
        m = C.metrics(d["y"], d["p"], t)
        rows.append(dict(m, step=step, label=label, model=model, role=role, pred=name, threshold=t))
    return rows


def write_csv(rows, path):
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(COLS)
        for r in rows:
            out = []
            for c in COLS:
                v = r.get(c, "")
                if isinstance(v, float):
                    v = "" if np.isnan(v) else (f"{v:.6g}" if c == "threshold" else f"{v:.4f}")
                out.append(v)
            w.writerow(out)
    print(f"  saved {path.relative_to(C.PROJECT_DIR)}")


def save(fig, name):
    path = C.PLOTS_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved {path.relative_to(C.PROJECT_DIR)}")


def grouped_bars(ax, labels, a, b, a_name, b_name, ymin):
    x = np.arange(len(labels))
    w = 0.38
    for off, vals, color, name in ((-w / 2, a, BLUE, a_name), (w / 2, b, ORANGE, b_name)):
        bars = ax.bar(x + off, np.asarray(vals) - ymin, w - 0.03, bottom=ymin, color=color, label=name)
        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, v + 0.002, f"{v:.3f}",
                    ha="center", va="bottom", fontsize=7.5, color=INK2)
    ax.set_xticks(x, labels, fontsize=8.5)
    ax.set_ylim(ymin, 1.012)
    ax.set_yticks(np.round(np.arange(ymin, 1.0001, 0.05 if ymin < 0.9 else 0.02), 2))


def plot_a(rows):
    rs = [r for r in rows if r["role"] in ("reference", "main")]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    grouped_bars(ax, [r["label"] for r in rs], [r["accuracy"] for r in rs],
                 [r["defect_recall"] for r in rs], "Accuracy", "Defect recall", 0.85)
    ax.axhline(0.91, ls="--", lw=1.2, color=INK2, label="MobileNetV2 Defect recall (0.91)")
    ax.set_ylabel("Score on test set (n = 300; y-axis starts at 0.85)")
    ax.set_title("A. Accuracy and Defect recall on test: MobileNetV2 vs baseline vs Contribution 2 steps",
                 loc="left", pad=28)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0), borderaxespad=0.2, fontsize=9)
    save(fig, "A_accuracy_defect_recall.png")


def plot_b(rows, best_name):
    base = next(r for r in rows if r["step"] == "1")
    best_step = "5" if best_name.startswith("ensemble") else "4"
    best = next(r for r in rows if r["step"] == best_step)
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.0))
    for ax, r, title in ((axes[0], base, "Baseline (threshold 0.5)"),
                         (axes[1], best, f"Best C2: {r_title(best)}")):
        cm = np.array([[r["tp"], r["fn"]], [r["fp"], r["tn"]]])
        rowpct = cm / cm.sum(axis=1, keepdims=True)
        ax.imshow(rowpct, cmap="Blues", vmin=0, vmax=1.25)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f"{cm[i, j]}\n({100 * rowpct[i, j]:.1f}%)", ha="center", va="center",
                        fontsize=11, color="white" if rowpct[i, j] > 0.6 else INK)
        ax.set_xticks([0, 1], ["Defect", "Fresh"])
        ax.set_yticks([0, 1], ["Defect", "Fresh"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.grid(False)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(title, fontsize=10)
    fig.suptitle("B. Test confusion matrices (counts, row %)", x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "B_confusion_baseline_vs_best.png")


def r_title(r):
    return r["label"].replace("\n", " ") + f" (t = {r['threshold']:.3g})"


def plot_c(rows):
    rs = [r for r in rows if r["role"] == "main" and r["step"] in ("1", "2", "3", "4", "5")]
    labels = [f"{r['step']}. {r['label']}" for r in rs]
    fig, ax = plt.subplots(figsize=(9, 4.4))
    grouped_bars(ax, labels, [r["defect_recall"] for r in rs], [r["fresh_recall"] for r in rs],
                 "Defect recall", "Fresh recall", 0.90)
    ax.axhline(C.FRESH_RECALL_FLOOR, ls=":", lw=1.2, color=INK2,
               label=f"Fresh-recall floor used on validation ({C.FRESH_RECALL_FLOOR})")
    ax.set_ylabel("Recall on test set (y-axis starts at 0.90)")
    ax.set_title("C. Ablation: Defect and Fresh recall on test as each step is added", loc="left", pad=28)
    ax.legend(loc="lower left", ncol=3, bbox_to_anchor=(0, 1.0), borderaxespad=0.2, fontsize=9)
    save(fig, "C_ablation_recalls.png")


def plot_d():
    s4, s5 = C.read_json("step4_threshold.json"), C.read_json("step5_ensemble.json")
    loss_label = {"weighted": "weighted CE", "focal": "focal loss"}
    panels = [(s4["best_model"], s4["threshold"], f"Step 4: {loss_label[s4['best_model']]} model"),
              ("ensemble", s5["threshold"], "Step 5: 5-fold ensemble")]
    grid = C.sigmoid(np.linspace(-9.3, 9.3, 1201))
    ticks = [1e-4, 1e-3, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999]
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4.0), sharey=True)
    for ax, (name, t, title) in zip(axes, panels):
        v = C.load_predictions(name)["val"]
        dr = [C.metrics(v["y"], v["p"], g)["defect_recall"] for g in grid]
        fr = [C.metrics(v["y"], v["p"], g)["fresh_recall"] for g in grid]
        ax.plot(grid, dr, color=BLUE, lw=2, label="Defect recall")
        ax.plot(grid, fr, color=ORANGE, lw=2, label="Fresh recall")
        ax.set_xscale("logit")
        ax.set_xlim(grid[0], grid[-1])
        ax.set_xticks(ticks, [f"{x:g}" for x in ticks], fontsize=8)
        ax.minorticks_off()
        ax.axhline(C.FRESH_RECALL_FLOOR, ls=":", lw=1, color=MUTED)
        ax.axvline(0.5, ls="--", lw=1.2, color=INK2, label="default 0.5")
        ax.axvline(t, ls="-", lw=1.2, color=INK, label=f"chosen {t:.3f}")
        ax.text(0.97, 0.03, f"chosen t = {t:.3f}  (default 0.5)", transform=ax.transAxes,
                ha="right", va="bottom", fontsize=8.5, color=INK,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=GRID))
        ax.set_ylim(0.8, 1.01)
        ax.set_xlabel("Defect decision threshold (logit scale)")
        ax.set_title(title, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("Recall on validation (n = 297)")
    L = plt.Line2D
    fig.legend([L([], [], color=BLUE, lw=2), L([], [], color=ORANGE, lw=2),
                L([], [], color=INK2, lw=1.2, ls="--"), L([], [], color=INK, lw=1.2),
                L([], [], color=MUTED, lw=1, ls=":")],
               ["Defect recall", "Fresh recall", "default threshold 0.5",
                "chosen threshold (validation)", "Fresh-recall floor 0.95"],
               loc="lower center", ncol=5, bbox_to_anchor=(0.5, -0.07), fontsize=9)
    axes[0].text(grid[0], C.FRESH_RECALL_FLOOR - 0.004, " Fresh floor 0.95", va="top", fontsize=7.5, color=MUTED)
    fig.suptitle("D. Recall vs decision threshold on validation (predict Defect if p >= threshold)",
                 x=0.02, ha="left", fontweight="bold")
    fig.tight_layout()
    save(fig, "D_threshold_sweep_val.png")


def main():
    test_rows, val_rows = build_rows("test"), build_rows("val")
    write_csv(test_rows, C.RESULTS_DIR / "metrics.csv")
    write_csv(val_rows, C.RESULTS_DIR / "metrics_val.csv")
    best = C.read_json("best_setup.json") or {"name": "ensemble"}
    plot_a(test_rows)
    plot_b(test_rows, best["name"])
    plot_c(test_rows)
    plot_d()
    print()
    for r in test_rows:
        print(f"  {r['step']:>3} {r['model'][:46]:46s} t={r['threshold']:<8.4g} acc={r['accuracy']:.4f} "
              f"DefR={r['defect_recall']:.4f} FreshR={r['fresh_recall']:.4f} AUC={r['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
