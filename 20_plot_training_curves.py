"""Training curves: baseline vs best synthetic (C1) run, read from Ultralytics results.csv."""
import csv
import matplotlib.pyplot as plt
from tomato_utils import *

SCALING_RESULTS = OUT / "plotE_scaling_validation.csv"
if not SCALING_RESULTS.is_file():
    raise FileNotFoundError(
        f"Validation scaling results not found: {SCALING_RESULTS}. "
        "Run 14_plot_E_scaling.py after training the scaling models."
    )
BEST_N = select_best_scaling_size(SCALING_RESULTS)
RUNS = {
    "Baseline": BASELINE_WEIGHTS.parent.parent / "results.csv",
    f"+ C1 (N={BEST_N})": PROJECT / f"runs/experiment_scaling/N{BEST_N}/results.csv",
}


def load(path):
    with open(path) as f:
        rd = csv.DictReader(f)
        return [{k.strip(): float(v) for k, v in r.items() if v.strip()} for r in rd]


fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
for (name, path), col in zip(RUNS.items(), ("#7a8b99", "#c0392b")):
    if not path.is_file():
        raise FileNotFoundError(
            f"Training results not found: {path}\n"
            "Train the baseline and selected scaling model first."
        )
    d = load(path); ep = [r["epoch"] for r in d]
    axs[0].plot(ep, [r["train/loss"] for r in d], color=col, ls="--", label=f"{name} train")
    axs[0].plot(ep, [r["val/loss"] for r in d], color=col, label=f"{name} val")
    axs[1].plot(ep, [r["metrics/accuracy_top1"] for r in d], color=col, label=name)
axs[0].set_title("Loss"); axs[1].set_title("Validation top-1 accuracy")
for a in axs:
    a.set_xlabel("Epoch"); a.grid(alpha=.3); a.legend(frameon=False, fontsize=8)
OUT.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(OUT / "training_curves.png", dpi=200); plt.close(fig)
print("saved", OUT / "training_curves.png")
