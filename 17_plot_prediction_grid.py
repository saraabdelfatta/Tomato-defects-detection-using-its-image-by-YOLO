"""Prediction grid for slide 9: every test error + confident hits + borderline hits, with confidence."""
import csv
import importlib.util
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from ultralytics import YOLO
from tomato_utils import *

CONTRIBUTION_2 = PROJECT / "contribution_2"
FOLD_RUNS = CONTRIBUTION_2 / "runs" / "calibration_folds"
CALIBRATION_SUMMARY = (
    CONTRIBUTION_2 / "runs" / "out_of_fold_calibration" / "summary.csv"
)
ENSEMBLE = sorted(FOLD_RUNS.glob("fold_*/weights/best.pt"))
if len(ENSEMBLE) != 5:
    raise FileNotFoundError(
        f"Expected five fold-model weights under {FOLD_RUNS}, "
        f"found {len(ENSEMBLE)}. Run contribution_2/16_train_calibration_folds.py."
    )
if not CALIBRATION_SUMMARY.is_file():
    raise FileNotFoundError(
        f"Calibration summary not found: {CALIBRATION_SUMMARY}\n"
        "Run contribution_2/17_out_of_fold_calibration.py first."
    )

calibration = {}
with CALIBRATION_SUMMARY.open(newline="", encoding="utf-8") as file:
    calibration = {
        row["metric"]: float(row["value"])
        for row in csv.DictReader(file)
        if row["metric"] in {
            "temperature_from_oof",
            "review_threshold_from_oof",
        }
    }
try:
    TEMPERATURE = calibration["temperature_from_oof"]
    REVIEW_THRESHOLD = calibration["review_threshold_from_oof"]
except KeyError as exc:
    raise ValueError(
        f"Missing calibration value {exc.args[0]} in {CALIBRATION_SUMMARY}"
    ) from exc

calibration_script = CONTRIBUTION_2 / "14_calibrated_reject_option.py"
spec = importlib.util.spec_from_file_location("reject_option", calibration_script)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load calibration helpers from {calibration_script}")
reject_option = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reject_option)

paths, y = list_images(TEST_DIR)
margins = np.mean(
    [
        reject_option.collect_logit_margins(
            YOLO(str(weights)), paths
        )
        for weights in ENSEMBLE
    ],
    axis=0,
)
p = reject_option.sigmoid(margins / TEMPERATURE)
pred = (p >= 0.5).astype(int)
confidence = np.maximum(p, 1.0 - p)
needs_review = confidence < REVIEW_THRESHOLD

wrong = list(np.where(pred != y)[0])
right = np.where(pred == y)[0]
tp = right[y[right] == 1]; tn = right[y[right] == 0]
chosen = wrong + list(tp[np.argsort(-p[tp])][:3]) + list(tn[np.argsort(p[tn])][:3]) \
         + list(right[np.argsort(np.abs(confidence[right] - REVIEW_THRESHOLD))][:2])
chosen = list(dict.fromkeys(int(i) for i in chosen))

names = {0: "Fresh", 1: "Defect"}
columns = 4
rows = max(1, int(np.ceil(len(chosen) / columns)))
fig, axes = plt.subplots(rows, columns, figsize=(12, 3.2 * rows), squeeze=False)
for ax in axes.ravel(): ax.axis("off")
for ax, i in zip(axes.ravel(), chosen):
    with Image.open(paths[i]) as image:
        ax.imshow(image.convert("RGB"))
    ok = pred[i] == y[i]
    col = "#2f6b45" if ok else "#c0392b"
    review_label = "Review" if needs_review[i] else "Automatic"
    ax.set_title(
        f"True {names[y[i]]} | Pred {names[pred[i]]} | {review_label}\n"
        f"P(Defect) = {p[i]:.2f} | confidence = {confidence[i]:.2f}",
                 fontsize=9, color=col, fontweight="bold")
    ax.axis("on"); ax.set_xticks([]); ax.set_yticks([])
    for s in ax.spines.values(): s.set_edgecolor(col); s.set_linewidth(4)
fig.suptitle(
    "Ensemble test predictions (class threshold 0.50; "
    f"review threshold {REVIEW_THRESHOLD:.3f}); green = correct, red = error",
    fontsize=11,
)
fig.tight_layout()
OUT.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT / "prediction_grid.png", dpi=170)
plt.close(fig)
print("errors shown:", len(wrong), "| saved", OUT / "prediction_grid.png")
