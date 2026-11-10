"""
Steps 2 and 3: retrain yolo11n-cls with a Defect-favouring loss.

    python step2_3_train_loss.py --loss weighted   # step 2: class-weighted cross-entropy
    python step2_3_train_loss.py --loss focal      # step 3: class-weighted focal loss, gamma = 2

Ultralytics' classification loss (v8ClassificationLoss) is a plain
F.cross_entropy and ignores `cls_pw`, so a minimal custom trainer swaps
in our own criterion. Ultralytics strips `model.criterion` before saving
checkpoints, so the resulting best.pt is a normal, portable YOLO model.

Settings match M1 (02_train_baseline.py): imgsz 224, batch 32, 30 epochs,
patience 10, seed 42, model selection on the validation split. The test
split is never touched during training.

A run whose best.pt already exists is not retrained; only its
predictions are (re)cached.

Outputs:
    runs/<loss>/weights/best.pt
    results/preds_<loss>.csv          val/test Defect probabilities
    results/train_<loss>.json         class weights, epochs, runtime
    results/runtime_log.csv           appended
"""

import argparse
import shutil
import sys
import time

import torch
import torch.nn.functional as F

from ultralytics import YOLO
from ultralytics.models.yolo.classify import ClassificationTrainer

import c2_common as C

EPOCHS, IMGSZ, BATCH, PATIENCE = 30, 224, 32, 10
FOCAL_GAMMA = 2.0


# ------------------------------------------------------------
# Losses
# ------------------------------------------------------------

class WeightedCELoss:
    """Cross-entropy with per-class weights (inverse class frequency)."""

    def __init__(self, weights):
        self.weights = weights

    def __call__(self, preds, batch):
        logits = preds[1] if isinstance(preds, (list, tuple)) else preds
        w = self.weights.to(logits.device, logits.dtype)
        loss = F.cross_entropy(logits, batch["cls"], weight=w, reduction="mean")
        return loss, {"loss": loss.detach()}


class FocalLoss:
    """Class-weighted focal loss: -alpha_t * (1 - p_t)^gamma * log(p_t)."""

    def __init__(self, weights, gamma=FOCAL_GAMMA):
        self.weights = weights
        self.gamma = gamma

    def __call__(self, preds, batch):
        logits = preds[1] if isinstance(preds, (list, tuple)) else preds
        target = batch["cls"].long()
        logp_t = F.log_softmax(logits.float(), dim=1).gather(1, target[:, None]).squeeze(1)
        p_t = logp_t.exp()
        alpha_t = self.weights.to(logits.device)[target]
        loss = (-alpha_t * (1 - p_t) ** self.gamma * logp_t).sum() / alpha_t.sum()
        return loss, {"loss": loss.detach()}


# ------------------------------------------------------------
# Minimal custom trainer
# ------------------------------------------------------------

LOSS_KIND = None  # set in main() before training starts


def class_weights(names):
    """Inverse-frequency weights N / (K * n_c), in the model's class-index order."""
    counts = {}
    for idx, name in names.items():
        folder = C.DATASET_DIR / "train" / name
        counts[int(idx)] = sum(1 for p in folder.iterdir() if p.suffix.lower() in C.IMAGE_EXTENSIONS)
    n, k = sum(counts.values()), len(counts)
    return torch.tensor([n / (k * counts[i]) for i in sorted(counts)], dtype=torch.float32), counts


class C2Trainer(ClassificationTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg=cfg, weights=weights, verbose=verbose)
        w, counts = class_weights(self.data["names"])
        model.criterion = WeightedCELoss(w) if LOSS_KIND == "weighted" else FocalLoss(w)
        self.c2_info = {
            "names": {int(k): v for k, v in self.data["names"].items()},
            "train_counts": counts,
            "class_weights": w.tolist(),
            "loss": type(model.criterion).__name__,
        }
        print(f"[C2] loss={self.c2_info['loss']} counts={counts} weights={[round(x, 3) for x in w.tolist()]}")
        return model


def check_criterion(trainer):
    """Fail fast if Ultralytics silently fell back to its default loss."""
    crit = getattr(trainer.model, "criterion", None)
    expected = WeightedCELoss if LOSS_KIND == "weighted" else FocalLoss
    if not isinstance(crit, expected):
        raise RuntimeError(f"Custom loss not active (criterion = {type(crit).__name__})")


# ------------------------------------------------------------
# Run
# ------------------------------------------------------------

def train(loss_kind, data_dir, run_name):
    global LOSS_KIND
    LOSS_KIND = loss_kind
    run_dir = C.RUNS_DIR / run_name
    best = run_dir / "weights" / "best.pt"
    if best.is_file():
        print(f"[C2] {best} exists, not retraining.")
        return best, None

    for device in (C.pick_device(), "cpu"):
        if run_dir.exists():
            shutil.rmtree(run_dir)
        model = YOLO(str(C.PRETRAINED))
        model.add_callback("on_train_epoch_end", check_criterion)
        t0 = time.time()
        try:
            model.train(
                trainer=C2Trainer, data=str(data_dir), epochs=EPOCHS, imgsz=IMGSZ,
                batch=BATCH, patience=PATIENCE, seed=C.SEED, deterministic=True,
                device=device, project=str(C.RUNS_DIR), name=run_name, exist_ok=True,
                plots=True, verbose=False,
            )
        except Exception as exc:  # e.g. an MPS op that is not implemented
            if device == "cpu":
                raise
            print(f"[C2] training on {device} failed ({exc!r}); retrying on CPU")
            continue
        seconds = time.time() - t0
        info = dict(model.trainer.c2_info, device=device, seconds=seconds,
                    epochs_run=model.trainer.epoch + 1,
                    best_epoch=getattr(model.trainer.stopper, "best_epoch", -1) + 1,
                    best_val_top1=float(model.trainer.best_fitness or 0), data=str(data_dir))
        C.write_json(f"train_{run_name}.json", info)
        C.log_runtime(f"train_{run_name}", seconds, f"device={device}")
        print(f"[C2] {run_name}: {seconds / 60:.1f} min on {device}")
        return best, info
    raise RuntimeError("training failed on every device")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--loss", choices=["weighted", "focal"], required=True)
    ap.add_argument("--data", default=str(C.DATASET_DIR), help="train/val folder dataset (default: M1 split)")
    ap.add_argument("--name", default=None, help="run name (default: the loss name)")
    args = ap.parse_args()
    name = args.name or args.loss

    best, _ = train(args.loss, args.data, name)
    t0 = time.time()
    C.cache_predictions(name, best, force=True)
    C.log_runtime(f"predict_{name}", time.time() - t0)


if __name__ == "__main__":
    sys.exit(main())
