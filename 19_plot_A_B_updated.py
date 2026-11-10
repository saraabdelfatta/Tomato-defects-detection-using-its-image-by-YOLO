"""Regenerates Plot A (accuracy + Defect recall) and Plot B (confusion matrices) with C1 and C1+C2 included."""
import csv
import numpy as np
import matplotlib.pyplot as plt
from tomato_utils import OUT

metrics_path = OUT / "metrics_all.csv"
if not metrics_path.is_file():
    raise FileNotFoundError(
        f"Metrics file not found: {metrics_path}. "
        "Run 18_evaluate_all_configs.py first."
    )
with metrics_path.open(newline="", encoding="utf-8") as file:
    rows = list(csv.DictReader(file))
if not rows:
    raise ValueError(f"No evaluation rows found in {metrics_path}")
OUT.mkdir(parents=True, exist_ok=True)
for r in rows:
    for k, v in r.items():
        if k not in ("config",): r[k] = float(v)

# ---- Plot A ----
labels = ["MobileNetV2\n(reported)"] + [r["config"].replace(" (", "\n(") for r in rows]
acc = [0.96] + [r["accuracy"] for r in rows]
rec = [0.91] + [r["defect_recall"] for r in rows]
x = np.arange(len(labels)); w = 0.38
fig, ax = plt.subplots(figsize=(9, 4.5))
b1 = ax.bar(x - w / 2, acc, w, label="Accuracy", color="#7a8b99")
b2 = ax.bar(x + w / 2, rec, w, label="Defect recall", color="#c0392b")
for b in list(b1) + list(b2):
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003, f"{b.get_height():.3f}", ha="center", fontsize=8)
ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9); ax.set_ylim(0.85, 1.02)
ax.set_ylabel("Test set"); ax.legend(frameon=False, loc="upper left")
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "A_accuracy_defect_recall.png", dpi=200); plt.close(fig)


# ---- Plot B ----
def cm(ax, r, title):
    mat = np.array([[r["tp"], r["fn"]], [r["fp"], r["tn"]]])          # rows: true Defect, true Fresh
    ax.imshow(mat, cmap="Blues")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=16,
                    color="white" if mat[i, j] > mat.max() / 2 else "black")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["Pred Defect", "Pred Fresh"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["True Defect", "True Fresh"])
    ax.set_title(title, fontsize=10)


best = max(rows, key=lambda r: (r["defect_recall"], r["accuracy"]))
base = rows[0]
fig, axs = plt.subplots(1, 2, figsize=(8, 3.6))
cm(axs[0], base, f"{base['config']} (thr {base['threshold']:.3f})")
cm(axs[1], best, f"{best['config']} (thr {best['threshold']:.3f})")
fig.tight_layout(); fig.savefig(OUT / "B_confusion_baseline_vs_best.png", dpi=200); plt.close(fig)

fig, axs = plt.subplots(1, len(rows), figsize=(4 * len(rows), 3.6))
for ax, r in zip(axs, rows): cm(ax, r, f"{r['config']} (thr {r['threshold']:.3f})")
fig.tight_layout(); fig.savefig(OUT / "B_confusion_all.png", dpi=200); plt.close(fig)
print("saved A and B to", OUT, "| best config:", best["config"])
