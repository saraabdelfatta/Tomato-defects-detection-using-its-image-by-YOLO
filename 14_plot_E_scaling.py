"""PLOT E: Defect and Fresh recall on validation vs synthetic scenes added."""
import csv
import matplotlib.pyplot as plt
from tomato_utils import *

SIZES = [0, 100, 250, 500, 1000]                 # 0 = baseline
RUNS = PROJECT / "runs" / "experiment_scaling"
paths, y = list_images(VAL_DIR)

rows = []
for n in SIZES:
    w = BASELINE_WEIGHTS if n == 0 else RUNS / f"N{n}" / "weights" / "best.pt"
    if not w.is_file():
        raise FileNotFoundError(
            f"Model weights not found: {w}\n"
            "Train the baseline and scaling models before plotting."
        )
    m = metrics(y, defect_probs([w], paths), 0.5)
    m["n_synthetic"] = n
    rows.append(m)
    print(n, f"defect {m['tp']}/{m['tp']+m['fn']}  fresh {m['tn']}/{m['tn']+m['fp']}  acc {m['accuracy']:.3f}")

OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "plotE_scaling_validation.csv", "w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
    wr.writeheader(); wr.writerows(rows)

x = range(len(SIZES))
fig, ax = plt.subplots(figsize=(7, 4.2))
ax.plot(x, [r["defect_recall"] for r in rows], "o-", color="#c0392b", lw=2.2, label="Defect recall")
ax.plot(x, [r["fresh_recall"] for r in rows], "s-", color="#2f6b45", lw=2.2, label="Fresh recall")
for i, r in enumerate(rows):
    ax.annotate(f"{r['tp']}/{r['tp']+r['fn']}", (i, r["defect_recall"]), textcoords="offset points",
                xytext=(0, -15), ha="center", fontsize=8, color="#c0392b")
ax.set_xticks(list(x)); ax.set_xticklabels([str(n) for n in SIZES])
ax.set_xlabel("Synthetic scenes added to the 1,389 real training images (0 = baseline)")
ax.set_ylabel("Recall on validation set"); ax.set_ylim(0.85, 1.01)
ax.grid(axis="y", alpha=.3); ax.legend(loc="lower right", frameon=False)
for s in ("top", "right"): ax.spines[s].set_visible(False)
fig.tight_layout(); fig.savefig(OUT / "E_synthetic_scaling_validation.png", dpi=200)
plt.close(fig)
print("saved", OUT / "E_synthetic_scaling_validation.png")
