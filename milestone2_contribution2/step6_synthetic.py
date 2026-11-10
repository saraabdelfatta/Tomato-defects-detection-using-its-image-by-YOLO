"""
Step 6: apply the best Contribution-2 setup to Sara's synthetic-data model.

    python step6_synthetic.py --weights /path/to/sara/best.pt
    python step6_synthetic.py --weights /path/to/sara/best.pt --train-data /path/to/synthetic_dataset
    (or set SYNTH_MODEL_PATH / SYNTH_DATA_PATH instead of the flags)

If no weights path is given, or it does not exist, the step is skipped
cleanly (exit code 0) and results/step6_synthetic.json records "skipped".

What "best setup" means is read from results/best_setup.json, which step5
writes using VALIDATION results only:
  * threshold (always applied): cache the model's val/test probabilities
    on the M1 split, pick its Defect threshold on validation with the
    same rule as step 4 (no retraining).
  * loss (only with --train-data): if the best setup includes the
    weighted or focal loss, retrain yolo11n-cls with that loss on the
    synthetic training set (one training run), then threshold it.
    Without --train-data the loss part is not applied, and this is noted.
  * ensemble: cannot be applied to one model; noted in the JSON.

Outputs: results/preds_synthetic.csv, results/step6_synthetic.json
"""

import argparse
import os
import sys
import time
from pathlib import Path

import c2_common as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=os.environ.get("SYNTH_MODEL_PATH"))
    ap.add_argument("--train-data", default=os.environ.get("SYNTH_DATA_PATH"))
    args = ap.parse_args()

    if not args.weights or not Path(args.weights).is_file():
        msg = f"synthetic model not available ({args.weights or 'no path given'}); step 6 skipped"
        print(f"  {msg}")
        C.write_json("step6_synthetic.json", {"status": "skipped", "reason": msg})
        stale = C.RESULTS_DIR / "preds_synthetic.csv"
        if stale.is_file():
            stale.unlink()
        return 0

    best = C.read_json("best_setup.json") or {"loss": None, "ensemble": False}
    notes, weights = [], Path(args.weights)
    if best.get("ensemble"):
        notes.append("best setup is the fold ensemble; only its threshold rule can be applied to one model")
    if best.get("loss"):
        if args.train_data:
            import step2_3_train_loss as T
            weights, _ = T.train(best["loss"], args.train_data, f"synthetic_{best['loss']}")
            notes.append(f"retrained with {best['loss']} loss on {args.train_data}")
        else:
            notes.append(f"best loss ({best['loss']}) not applied: no --train-data given; threshold only")

    t0 = time.time()
    preds = C.cache_predictions("synthetic", weights, force=True)
    C.log_runtime("predict_synthetic", time.time() - t0)
    v = preds["val"]
    t = C.choose_threshold(v["y"], v["p"])
    C.write_json("step6_synthetic.json", {
        "status": "done", "weights": str(weights), "source_weights": args.weights,
        "threshold": t, "val_at_threshold": C.metrics(v["y"], v["p"], t),
        "val_at_0.5": C.metrics(v["y"], v["p"], 0.5), "notes": notes,
    })
    return 0


if __name__ == "__main__":
    sys.exit(main())
