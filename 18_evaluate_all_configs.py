"""Scores every configuration on the SAME real test set -> fills the [TBD] cells of slide 8."""
import csv
import importlib.util
import numpy as np
from ultralytics import YOLO
from tomato_utils import *

SCALING_RESULTS = OUT / "plotE_scaling_validation.csv"
if not SCALING_RESULTS.is_file():
    raise FileNotFoundError(
        f"Validation scaling results not found: {SCALING_RESULTS}. "
        "Run 14_plot_E_scaling.py after training the scaling models."
    )
BEST_N = select_best_scaling_size(SCALING_RESULTS)
SCALING_WEIGHT = (
    PROJECT / "runs" / "experiment_scaling" / f"N{BEST_N}" / "weights" / "best.pt"
)
CONTRIBUTION_2 = PROJECT / "contribution_2"
FOLD_WEIGHTS = sorted(
    (CONTRIBUTION_2 / "runs" / "calibration_folds").glob(
        "fold_*/weights/best.pt"
    )
)
CALIBRATION_SUMMARY = (
    CONTRIBUTION_2 / "runs" / "out_of_fold_calibration" / "summary.csv"
)
calibration_script = CONTRIBUTION_2 / "14_calibrated_reject_option.py"
spec = importlib.util.spec_from_file_location("reject_option", calibration_script)
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load calibration helpers from {calibration_script}")
reject_option = importlib.util.module_from_spec(spec)
spec.loader.exec_module(reject_option)


def require_file(path, description):
    if not path.is_file():
        raise FileNotFoundError(f"{description} not found: {path}")


require_file(BASELINE_WEIGHTS, "Baseline weights")
require_file(SCALING_WEIGHT, f"N={BEST_N} scaling weights")
if len(FOLD_WEIGHTS) != 5:
    raise FileNotFoundError(
        f"Expected five fold-model weights under "
        f"{CONTRIBUTION_2 / 'runs' / 'calibration_folds'}, "
        f"found {len(FOLD_WEIGHTS)}."
    )
require_file(CALIBRATION_SUMMARY, "Out-of-fold calibration summary")

calibration = {}
with CALIBRATION_SUMMARY.open(newline="", encoding="utf-8") as file:
    calibration = {
        row["metric"]: float(row["value"])
        for row in csv.DictReader(file)
        if row["metric"] == "temperature_from_oof"
    }
try:
    temperature = calibration["temperature_from_oof"]
except KeyError as exc:
    raise ValueError(
        f"Missing calibration value {exc.args[0]} in {CALIBRATION_SUMMARY}"
    ) from exc


def margins(weight, paths):
    return reject_option.collect_logit_margins(YOLO(str(weight)), paths)


def ensemble_margins(paths):
    return np.mean([margins(weight, paths) for weight in FOLD_WEIGHTS], axis=0)


tp_paths, ty = list_images(TEST_DIR)
vp_paths, vy = list_images(VAL_DIR)
baseline_test = margins(BASELINE_WEIGHTS, tp_paths)
c1_test = margins(SCALING_WEIGHT, tp_paths)
c2_test = ensemble_margins(tp_paths)
c1_val = margins(SCALING_WEIGHT, vp_paths)
c2_val = ensemble_margins(vp_paths)
c1_c2_test = (c1_test + c2_test) / 2
c1_c2_val = (c1_val + c2_val) / 2

configs = [
    ("YOLO11 baseline", reject_option.sigmoid(baseline_test), 0.5, 1),
    ("+ C1 (synthetic)", reject_option.sigmoid(c1_test), 0.5, 1),
    (
        "+ C2 (five-fold)",
        reject_option.sigmoid(c2_test / temperature),
        0.5,
        len(FOLD_WEIGHTS),
    ),
]
c1_c2_val_probs = reject_option.sigmoid(c1_c2_val)
c1_c2_threshold = tune_threshold(vy, c1_c2_val_probs)
configs.append(
    (
        "+ C1 + C2",
        reject_option.sigmoid(c1_c2_test),
        c1_c2_threshold,
        len(FOLD_WEIGHTS) + 1,
    )
)

rows = []
for name, probabilities, threshold, n_models in configs:
    m = metrics(ty, probabilities, threshold)
    m.update(config=name, n_models=n_models)
    rows.append(m)

OUT.mkdir(parents=True, exist_ok=True)
with (OUT / "metrics_all.csv").open("w", newline="", encoding="utf-8") as f:
    wr = csv.DictWriter(
        f,
        fieldnames=["config", "n_models"]
        + [k for k in rows[0] if k not in ("config", "n_models")],
    )
    wr.writeheader(); wr.writerows(rows)

print(f"{'config':18} thr    acc    D-prec D-rec  D-F1   F-F1   (missed defects / false alarms)")
for m in rows:
    print(f"{m['config']:18} {m['threshold']:.3f}  {m['accuracy']*100:5.1f}%  {m['defect_precision']:.3f}  "
          f"{m['defect_recall']:.3f}  {m['defect_f1']:.3f}  {m['fresh_f1']:.3f}   ({m['fn']} / {m['fp']})")
print(f"Selected C1 size from validation results: N={BEST_N}")
print("saved", OUT / "metrics_all.csv")
