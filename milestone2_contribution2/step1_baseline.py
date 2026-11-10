"""
Step 1: cache the M1 baseline's validation and test Defect probabilities.

    python step1_baseline.py

Input:  runs/classify/runs/baseline/weights/best.pt  (M1 baseline, not retrained)
Output: results/preds_baseline.csv
"""

import time

import c2_common as C


def main():
    t0 = time.time()
    preds = C.cache_predictions("baseline", C.BASELINE_WEIGHTS, force=True)
    C.log_runtime("predict_baseline", time.time() - t0)
    for split in ("val", "test"):
        m = C.metrics(preds[split]["y"], preds[split]["p"])
        print(f"  baseline {split}: acc={m['accuracy']:.4f} defect_recall={m['defect_recall']:.4f} "
              f"fresh_recall={m['fresh_recall']:.4f} auc={m['roc_auc']:.4f}")


if __name__ == "__main__":
    main()
