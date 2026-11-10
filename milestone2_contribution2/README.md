# Milestone 2 – Contribution 2: Optimizing for Defect recall

YOLO11n-cls, tomato Fresh/Defect. Each step builds on the previous one. All
predictions are cached in `results/`, so thresholds, the metrics table and the plots
rerun in seconds without loading or training a model.

## Ground rules

- **Data:** the leakage-free M1 70/15/15 split in `../contribution_2/datasets/tomato`
  (train 944 Fresh / 445 Defect, val 202 / 95, test 204 / 96).
- **No tuning on test.** Model choice, thresholds and the "best setup" all use the
  validation split. Test is read once, by `evaluate.py`.
- **Reuse:** M1 baseline `../runs/classify/runs/baseline/weights/best.pt` and the five
  fold models `../contribution_2/runs/calibration_folds/fold_k/weights/best.pt` are only
  used for inference. Their cached margins match the earlier `contribution_2` CSVs exactly.
- **Training settings (same as M1):** imgsz 224, batch 32, 30 epochs, patience 10,
  seed 42, `deterministic=True`, device `mps` (falls back to CPU automatically).
- **New training runs:** 2 (steps 2 and 3). Runtime per run is logged in
  `results/runtime_log.csv` and `results/train_<loss>.json`.

## Environment

Use the existing virtual environment (Ultralytics 8.4.113, torch 2.14, matplotlib, numpy;
no sklearn or pandas needed):

```bash
cd milestone2_contribution2
PY=../contribution_2/.venv/bin/python
```

## Steps

| # | Command | What it does | Produces |
|---|---|---|---|
| 1 | `$PY step1_baseline.py` | Loads the M1 baseline and caches its val/test Defect probabilities. | `results/preds_baseline.csv` |
| 2 | `$PY step2_3_train_loss.py --loss weighted` | Retrains yolo11n-cls with **class-weighted cross-entropy**. Weights are inverse class frequency, N/(K·n_c): Defect 1.561, Fresh 0.736 (≈2.1 : 1). | `runs/weighted/`, `results/preds_weighted.csv`, `results/train_weighted.json` |
| 3 | `$PY step2_3_train_loss.py --loss focal` | Same, with **class-weighted focal loss**, γ = 2 (α = the step-2 class weights). | `runs/focal/`, `results/preds_focal.csv`, `results/train_focal.json` |
| 4 | `$PY step4_threshold.py` | For each of the step-2 and step-3 models, picks the Defect threshold on **validation** that maximises Defect recall while Fresh recall ≥ 0.95 (ties: highest Fresh recall). The better model on validation is kept. | `results/step4_threshold.json` |
| 5 | `$PY step5_ensemble.py` | Averages the Defect probabilities of the 5 existing fold models (no retraining) and applies the same threshold rule. Also picks the overall best C2 setup on validation. | `results/preds_fold_k.csv`, `results/preds_ensemble.csv`, `results/step5_ensemble.json`, `results/best_setup.json` |
| 6 | `$PY step6_synthetic.py --weights <sara_best.pt>` | Applies the best C2 setup to Sara's synthetic-data model. **Skips cleanly** if no path is given or the file is missing. With `--train-data <synthetic dataset>` it also retrains with the best loss (one extra run), if the best setup includes one. | `results/preds_synthetic.csv`, `results/step6_synthetic.json` |
| – | `$PY evaluate.py` | Builds the metrics table and plots A–D from the caches. | `results/metrics.csv`, `results/metrics_val.csv`, `plots/*.png` |

Run them in this order. Steps 1, 4, 5 and `evaluate.py` take seconds to about a minute.
Steps 2 and 3 skip training if `runs/<loss>/weights/best.pt` already exists.

### How the custom loss works

Ultralytics' classification loss (`v8ClassificationLoss`) is a plain `F.cross_entropy`. The
`cls_pw` class-weight option exists only for detection-style losses, so classification has
no native class weighting. `step2_3_train_loss.py` subclasses `ClassificationTrainer` and,
in `get_model`, attaches a `WeightedCELoss` or `FocalLoss` object as `model.criterion`. A
callback checks each epoch that the custom loss is still active. Ultralytics sets
`criterion` to `None` before saving checkpoints, so `best.pt` is an ordinary YOLO model that
loads without this code. Oversampling was not needed.

Model selection (`best.pt`) still uses Ultralytics' default fitness, which is validation
top-1 accuracy.

## Outputs

- `results/metrics.csv`: test metrics, one tidy row per step. Columns: accuracy,
  Defect/Fresh precision/recall/F1, ROC-AUC, threshold, and confusion counts
  (tp/fn/fp/tn, with Defect as the positive class). It includes the MobileNetV2 reference row
  (AUC not reported) and two supplementary rows: baseline + threshold, and ensemble at 0.5.
  These separate the effect of the threshold from that of the model.
- `results/metrics_val.csv`: the same rows on validation.
- `plots/A_accuracy_defect_recall.png`: accuracy and Defect recall, MobileNetV2 vs baseline
  vs each C2 step (plus synthetic when available), with a dashed line at 0.91.
- `plots/B_confusion_baseline_vs_best.png`: test confusion matrices, baseline vs best C2 setup.
- `plots/C_ablation_recalls.png`: Defect and Fresh recall on test as each step is added.
- `plots/D_threshold_sweep_val.png`: Defect and Fresh recall vs threshold on validation, with
  the default 0.5 and the chosen threshold marked.

Bar charts A and C start the y-axis at 0.85 / 0.90 so that differences of one or two images
are visible. Each bar is labelled with its value.

## Results (test, n = 300; 96 Defect / 204 Fresh)

| Step | Threshold | Acc | Defect P | Defect R | Defect F1 | Fresh P | Fresh R | Fresh F1 | AUC |
|---|---|---|---|---|---|---|---|---|---|
| MobileNetV2 (reference) | – | 0.960 | 1.000 | 0.910 | 0.950 | 0.930 | 1.000 | 0.960 | – |
| 1 Baseline | 0.5 | 0.967 | 0.939 | 0.958 | 0.949 | 0.980 | 0.971 | 0.975 | 0.996 |
| 2 Weighted CE | 0.5 | 0.980 | 0.969 | 0.969 | 0.969 | 0.985 | 0.985 | 0.985 | 0.999 |
| 3 Focal (γ=2) | 0.5 | 0.963 | 0.921 | 0.969 | 0.944 | 0.985 | 0.961 | 0.973 | 0.997 |
| 4 Weighted CE + val threshold | 0.489 | 0.980 | 0.969 | 0.969 | 0.969 | 0.985 | 0.985 | 0.985 | 0.999 |
| 5 Ensemble + val threshold | 0.451 | **0.990** | 0.990 | **0.979** | 0.984 | 0.990 | 0.995 | 0.993 | 0.998 |

Step 6 was skipped because no synthetic model path was given. Total compute was about 22 min:
two training runs of 10.6 and 10.2 min on MPS, plus about 1.5 min of inference. The full
table, including the supplementary rows, is in `results/metrics.csv`.

## Notes and caveats

- One test image is worth 1.04 pp of Defect recall (96 Defect images) and 0.49 pp of Fresh
  recall (204 Fresh images). Differences of one or two images are within noise. Only one seed
  was run per configuration (seed 42).
- The fold ensemble members were trained with plain cross-entropy (M1 settings), not with
  the step-2/3 losses. Step 5 therefore stacks on the threshold rule of step 4, not on the
  loss change.
- Validation was already used for early stopping and checkpoint selection, so validation
  scores are slightly optimistic, and the threshold rule has little room to move when a
  model already separates validation almost perfectly.
