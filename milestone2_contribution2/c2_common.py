"""
Shared paths, inference, metrics and threshold helpers for
Milestone 2 / Contribution 2 ("Optimizing for Defect recall").

Every step script imports this module. Nothing here trains a model.
"""

from pathlib import Path
import csv
import json
import time

import numpy as np

# ------------------------------------------------------------
# Paths
# ------------------------------------------------------------

C2_DIR = Path(__file__).resolve().parent
PROJECT_DIR = C2_DIR.parent

# Leakage-free M1 split (70/15/15), as prepared by 01_prepare_data.py.
DATASET_DIR = PROJECT_DIR / "contribution_2" / "datasets" / "tomato"

BASELINE_WEIGHTS = PROJECT_DIR / "runs" / "classify" / "runs" / "baseline" / "weights" / "best.pt"
FOLD_WEIGHTS = [
    PROJECT_DIR / "contribution_2" / "runs" / "calibration_folds" / f"fold_{k}" / "weights" / "best.pt"
    for k in range(5)
]
PRETRAINED = PROJECT_DIR / "yolo11n-cls.pt"

RESULTS_DIR = C2_DIR / "results"
PLOTS_DIR = C2_DIR / "plots"
RUNS_DIR = C2_DIR / "runs"
for _d in (RESULTS_DIR, PLOTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

RUNTIME_LOG = RESULTS_DIR / "runtime_log.csv"

SEED = 42
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
FRESH_RECALL_FLOOR = 0.95  # constraint for threshold selection (validation only)

# MobileNetV2 reference from the milestone brief (AUC not reported).
MOBILENET_ROW = {
    "model": "MobileNetV2 (reference)", "step": "ref", "threshold": np.nan,
    "accuracy": 0.96, "defect_precision": 1.00, "defect_recall": 0.91, "defect_f1": 0.95,
    "fresh_precision": 0.93, "fresh_recall": 1.00, "fresh_f1": 0.96, "roc_auc": np.nan,
}


# ------------------------------------------------------------
# Data
# ------------------------------------------------------------

def list_split(split):
    """Return (paths, labels) for one split; label 1 = Defect, 0 = Fresh."""
    paths, labels = [], []
    for class_name, label in (("Defect", 1), ("Fresh", 0)):
        folder = DATASET_DIR / split / class_name
        if not folder.is_dir():
            raise FileNotFoundError(f"Missing split folder: {folder}")
        files = sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS)
        paths += files
        labels += [label] * len(files)
    return paths, np.asarray(labels, dtype=int)


def pick_device():
    try:
        import torch
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


# ------------------------------------------------------------
# Inference (logits captured from the final linear layer, because
# Ultralytics' softmax output saturates at exactly 1.0)
# ------------------------------------------------------------

def logit_margins(weights, paths, device=None):
    """Return logit(Defect) - logit(Fresh) for every image path."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    names = {v: int(k) for k, v in model.names.items()}
    d_idx, f_idx = names["Defect"], names["Fresh"]

    captured = []
    hook = model.model.model[-1].linear.register_forward_hook(
        lambda m, i, o: captured.append(o.detach().float().cpu().numpy())
    )
    device = device or pick_device()
    out = []
    try:
        for p in paths:
            try:
                model.predict(str(p), verbose=False, device=device)
            except Exception:
                if device == "cpu":
                    raise
                device = "cpu"  # MPS failure -> fall back once
                model.predict(str(p), verbose=False, device=device)
            z = captured[-1][0].astype(np.float64)
            out.append(z[d_idx] - z[f_idx])
            captured.clear()
    finally:
        hook.remove()
    return np.asarray(out)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def cache_predictions(name, weights, splits=("val", "test"), force=False):
    """Compute (or load) cached per-image Defect probabilities for a model."""
    path = RESULTS_DIR / f"preds_{name}.csv"
    if path.is_file() and not force:
        return load_predictions(name)
    rows = []
    for split in splits:
        paths, y = list_split(split)
        t0 = time.time()
        m = logit_margins(weights, paths)
        print(f"  {name}/{split}: {len(paths)} images in {time.time() - t0:.1f}s")
        rows += [(split, p.name, int(t), float(mm), float(sigmoid(mm))) for p, t, mm in zip(paths, y, m)]
    save_predictions(name, rows)
    return load_predictions(name)


def save_predictions(name, rows):
    path = RESULTS_DIR / f"preds_{name}.csv"
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["split", "image", "label_defect", "logit_margin", "p_defect"])
        for split, image, y, m, p in rows:
            w.writerow([split, image, y, f"{m:.6f}", f"{p:.8f}"])
    print(f"  saved {path.relative_to(PROJECT_DIR)}")


def load_predictions(name):
    """Return {split: {"image": [...], "y": array, "p": array}}."""
    path = RESULTS_DIR / f"preds_{name}.csv"
    if not path.is_file():
        raise FileNotFoundError(f"No cached predictions: {path}. Run the step that creates it.")
    data = {}
    with path.open() as f:
        for r in csv.DictReader(f):
            d = data.setdefault(r["split"], {"image": [], "y": [], "p": []})
            d["image"].append(r["image"])
            d["y"].append(int(r["label_defect"]))
            d["p"].append(float(r["p_defect"]))
    for d in data.values():
        d["y"] = np.asarray(d["y"])
        d["p"] = np.asarray(d["p"])
    return data


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------

def roc_auc(y, score):
    """Mann-Whitney AUC with tie handling (no sklearn needed)."""
    y = np.asarray(y)
    score = np.asarray(score, dtype=np.float64)
    order = np.argsort(score, kind="mergesort")
    ranks = np.empty(len(score))
    sorted_s = score[order]
    i = 0
    while i < len(score):
        j = i
        while j + 1 < len(score) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    n_pos, n_neg = y.sum(), (1 - y).sum()
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def confusion(y, pred):
    """[[TP_defect, FN_defect], [FP_defect, TN_defect]] in Defect/Fresh row order."""
    tp = int(((y == 1) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    return tp, fn, fp, tn


def _safe(a, b):
    return a / b if b else 0.0


def metrics(y, p, threshold=0.5):
    pred = (p >= threshold).astype(int)
    tp, fn, fp, tn = confusion(y, pred)
    dp, dr = _safe(tp, tp + fp), _safe(tp, tp + fn)
    fp_, fr = _safe(tn, tn + fn), _safe(tn, tn + fp)
    return {
        "threshold": threshold,
        "accuracy": (tp + tn) / len(y),
        "defect_precision": dp, "defect_recall": dr, "defect_f1": _safe(2 * dp * dr, dp + dr),
        "fresh_precision": fp_, "fresh_recall": fr, "fresh_f1": _safe(2 * fp_ * fr, fp_ + fr),
        "roc_auc": roc_auc(y, p),
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
    }


# ------------------------------------------------------------
# Threshold selection (validation only)
# ------------------------------------------------------------

def choose_threshold(y_val, p_val, floor=FRESH_RECALL_FLOOR):
    """
    Pick the Defect threshold t (predict Defect if p >= t) on validation
    that maximises Defect recall subject to Fresh recall >= floor.
    Ties are broken by the highest Fresh recall. Candidate thresholds are
    midpoints (in logit space) between consecutive distinct val scores,
    so the chosen cut never sits exactly on a validation image.
    """
    eps = 1e-12
    z = np.log(np.clip(p_val, eps, 1 - eps)) - np.log(np.clip(1 - p_val, eps, 1 - eps))
    u = np.unique(z)
    cands = np.concatenate([[u[0] - 1.0], (u[:-1] + u[1:]) / 2, [u[-1] + 1.0]])
    best = None
    for c in cands:
        pred = (z >= c).astype(int)
        tp, fn, fp, tn = confusion(y_val, pred)
        dr, fr = _safe(tp, tp + fn), _safe(tn, tn + fp)
        if fr < floor:
            continue
        key = (dr, fr)
        if best is None or key > best[0]:
            best = (key, c)
    return float(sigmoid(best[1]))


# ------------------------------------------------------------
# Runtime log
# ------------------------------------------------------------

def log_runtime(step, seconds, note=""):
    new = not RUNTIME_LOG.is_file()
    with RUNTIME_LOG.open("a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["step", "seconds", "minutes", "note", "logged_at"])
        w.writerow([step, f"{seconds:.1f}", f"{seconds / 60:.2f}", note, time.strftime("%Y-%m-%d %H:%M:%S")])


def write_json(name, obj):
    path = RESULTS_DIR / name
    path.write_text(json.dumps(obj, indent=2, default=float))
    print(f"  saved {path.relative_to(PROJECT_DIR)}")
    return path


def read_json(name):
    path = RESULTS_DIR / name
    return json.loads(path.read_text()) if path.is_file() else None
