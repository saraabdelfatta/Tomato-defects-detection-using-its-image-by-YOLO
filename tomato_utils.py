"""Shared helpers for the Milestone 2 plots.
EDIT THE PATHS IN THIS FILE ONLY. Put this file next to 01_prepare_data.py.
"""
from pathlib import Path
import csv
import numpy as np

PROJECT = Path(__file__).resolve().parent
DATASETS = PROJECT.parent / "datasets"          # README: datasets/ sits next to project/
REAL = DATASETS / "tomato"                      # train/ val/ test/ with Fresh/ Defect/
VAL_DIR, TEST_DIR = REAL / "val", REAL / "test"

BASELINE_WEIGHTS = PROJECT / "runs/classify/runs/baseline/weights/best.pt"
OUT = PROJECT / "runs" / "milestone2_plots"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMGSZ = 224                                     # use the same imgsz as 02_train_baseline.py


def list_images(split_dir):
    """Returns (paths, labels). Label 1 = Defect, 0 = Fresh."""
    paths, y = [], []
    for cls, label in (("Fresh", 0), ("Defect", 1)):
        for p in sorted((Path(split_dir) / cls).iterdir()):
            if p.suffix.lower() in IMG_EXT:
                paths.append(p)
                y.append(label)
    return paths, np.array(y)


def defect_probs(weights, paths, imgsz=IMGSZ, batch=32):
    """Mean P(Defect) over one model or an ensemble (soft voting)."""
    from ultralytics import YOLO
    per_model = []
    for w in weights:
        model = YOLO(str(w))
        idx = [i for i, n in model.names.items() if n.lower() == "defect"][0]
        out = []
        for i in range(0, len(paths), batch):
            res = model.predict([str(p) for p in paths[i:i + batch]],
                                imgsz=imgsz, verbose=False)
            out += [float(r.probs.data[idx]) for r in res]
        per_model.append(out)
    return np.mean(per_model, axis=0)


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())

    def prf(t, f_p, f_n):
        pr = t / (t + f_p) if t + f_p else 0.0
        rc = t / (t + f_n) if t + f_n else 0.0
        return pr, rc, (2 * pr * rc / (pr + rc) if pr + rc else 0.0)

    d_pr, d_rc, d_f1 = prf(tp, fp, fn)
    _, f_rc, f_f1 = prf(tn, fn, fp)             # Fresh treated as the positive class
    return dict(threshold=thr, tp=tp, fn=fn, fp=fp, tn=tn,
                accuracy=(tp + tn) / len(y), defect_precision=d_pr,
                defect_recall=d_rc, defect_f1=d_f1,
                fresh_recall=f_rc, fresh_f1=f_f1)


def tune_threshold(y_val, p_val, fresh_floor=0.95):
    """Slide 13 rule: maximise Defect recall subject to Fresh recall >= floor.
    Ties -> highest threshold. If your C2 script uses a different tie rule, reuse it."""
    best_thr, best_rc = 0.5, -1.0
    for thr in np.round(np.arange(0.05, 0.951, 0.001), 3):
        m = metrics(y_val, p_val, thr)
        if m["fresh_recall"] >= fresh_floor and m["defect_recall"] >= best_rc:
            best_thr, best_rc = float(thr), m["defect_recall"]
    return best_thr


def select_best_scaling_size(results_path):
    """Choose N using validation recall, then Fresh recall, then the smaller N."""
    with Path(results_path).open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    candidates = [
        row for row in rows
        if int(row["n_synthetic"]) > 0
    ]
    if not candidates:
        raise ValueError(f"No synthetic scaling results found in {results_path}")
    best = max(
        candidates,
        key=lambda row: (
            float(row["defect_recall"]),
            float(row["fresh_recall"]),
            -int(row["n_synthetic"]),
        ),
    )
    return int(best["n_synthetic"])
