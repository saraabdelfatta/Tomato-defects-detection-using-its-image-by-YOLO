"""
Step 5: ensemble of the five existing fold models (no retraining).

    python step5_ensemble.py

The fold models (contribution_2/runs/calibration_folds/fold_k) were each
trained with the M1 settings on 4/5 of the training split and early-
stopped on the same validation split. Their val/test Defect probabilities
are cached once, averaged, and the step-4 threshold rule is applied on
validation.

Output:
    results/preds_fold_<k>.csv
    results/preds_ensemble.csv   (mean of the five Defect probabilities)
    results/step5_ensemble.json
"""

import time

import numpy as np

import c2_common as C


def main():
    t0 = time.time()
    folds = []
    for k, w in enumerate(C.FOLD_WEIGHTS):
        if not w.is_file():
            raise FileNotFoundError(f"Fold weights missing: {w}")
        folds.append(C.cache_predictions(f"fold_{k}", w))
    C.log_runtime("predict_folds", time.time() - t0, "cached files are reused if present")

    rows = []
    for split in ("val", "test"):
        imgs = folds[0][split]["image"]
        for f in folds[1:]:
            assert f[split]["image"] == imgs, "fold prediction files are not aligned"
        p = np.mean([f[split]["p"] for f in folds], axis=0)
        y = folds[0][split]["y"]
        z = np.log(np.clip(p, 1e-12, 1)) - np.log(np.clip(1 - p, 1e-12, 1))
        rows += [(split, i, int(t), float(zz), float(pp)) for i, t, zz, pp in zip(imgs, y, z, p)]
    C.save_predictions("ensemble", rows)

    v = C.load_predictions("ensemble")["val"]
    t = C.choose_threshold(v["y"], v["p"])
    at_t, at_05 = C.metrics(v["y"], v["p"], t), C.metrics(v["y"], v["p"], 0.5)
    print(f"  ensemble val: t={t:.4g}  defect_recall {at_05['defect_recall']:.4f} -> {at_t['defect_recall']:.4f}  "
          f"fresh_recall {at_05['fresh_recall']:.4f} -> {at_t['fresh_recall']:.4f}")
    C.write_json("step5_ensemble.json", {
        "members": [str(w.relative_to(C.PROJECT_DIR)) for w in C.FOLD_WEIGHTS],
        "combination": "mean of Defect probabilities",
        "threshold": t, "val_at_threshold": at_t, "val_at_0.5": at_05,
    })

    # Best overall C2 setup, chosen on VALIDATION only (used by step 6 and the plots).
    s4 = C.read_json("step4_threshold.json")
    if s4 is None:
        print("  step4_threshold.json missing; run step 4 first to pick the best setup")
        return
    s4_val = s4["candidates"][s4["best_model"]]["val_at_threshold"]
    key = lambda m: (m["defect_recall"], m["fresh_recall"], m["roc_auc"])
    use_ens = key(at_t) > key(s4_val)
    best = ({"name": "ensemble+threshold", "ensemble": True, "loss": None, "threshold": t}
            if use_ens else
            {"name": f"{s4['best_model']}+threshold", "ensemble": False,
             "loss": s4["best_model"], "threshold": s4["threshold"]})
    best["rule"] = "higher val Defect recall at chosen threshold; ties: val Fresh recall, then val AUC"
    print(f"  best C2 setup (validation): {best['name']}")
    C.write_json("best_setup.json", best)


if __name__ == "__main__":
    main()
