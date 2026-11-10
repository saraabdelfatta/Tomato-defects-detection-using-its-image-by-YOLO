"""
Step 4: decision threshold on the best model from steps 2-3.

    python step4_threshold.py

1. For each of the step-2 (weighted CE) and step-3 (focal) models, choose
   the Defect threshold on VALIDATION that maximises Defect recall while
   keeping Fresh recall >= 0.95 (c2_common.choose_threshold).
2. "Best model" = higher validation Defect recall at its chosen threshold
   (ties: higher validation Fresh recall, then validation ROC-AUC).
   The test set plays no part in either choice.
3. The chosen threshold is then applied once to test (in evaluate.py).

The same rule is also applied to the baseline as a supplementary
"threshold only" row, to separate the effect of the loss from that of
the threshold.

Input:  results/preds_{weighted,focal,baseline}.csv
Output: results/step4_threshold.json
"""

import c2_common as C


def tune(name):
    v = C.load_predictions(name)["val"]
    t = C.choose_threshold(v["y"], v["p"])
    at_t, at_05 = C.metrics(v["y"], v["p"], t), C.metrics(v["y"], v["p"], 0.5)
    print(f"  {name:9s} val: t={t:.4g}  defect_recall {at_05['defect_recall']:.4f} -> {at_t['defect_recall']:.4f}  "
          f"fresh_recall {at_05['fresh_recall']:.4f} -> {at_t['fresh_recall']:.4f}  auc={at_t['roc_auc']:.4f}")
    return {"threshold": t, "val_at_threshold": at_t, "val_at_0.5": at_05}


def main():
    cands = {name: tune(name) for name in ("weighted", "focal")}
    best = max(cands, key=lambda n: (cands[n]["val_at_threshold"]["defect_recall"],
                                     cands[n]["val_at_threshold"]["fresh_recall"],
                                     cands[n]["val_at_threshold"]["roc_auc"]))
    print(f"  best model from steps 2-3 (validation): {best}")
    C.write_json("step4_threshold.json", {
        "rule": f"max val Defect recall s.t. val Fresh recall >= {C.FRESH_RECALL_FLOOR}",
        "best_model": best, "threshold": cands[best]["threshold"],
        "candidates": cands, "baseline_supplementary": tune("baseline"),
    })


if __name__ == "__main__":
    main()
