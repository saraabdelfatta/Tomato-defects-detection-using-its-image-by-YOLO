# Presentation Requirements Gap and Asset Guide

This note compares the Milestone 1 requirements with `Presentation2.pptx`, the proposal PDF, the local code/data folders, and the run outputs currently available.

## Short version

- The deck has 12 slides and already covers the proposal, problem, related work, dataset, baseline choice, findings, timeline, team, and references.
- Slides 6–11 still contain important template placeholders. Fill those with the actual experiment status, results, observations, dates, and each person's technical work.
- The baseline and a small synthetic-data training pilot have been run. This is an offline image-classification experiment, not a deployed conveyor-belt system.
- The proposal's real belt-style test and calibrated reject/review option are not evidenced by the current code/results. State that they are pending rather than implying they are finished.

## Where the folders are

The main Git repository is this `project/` folder. `datasets/` and `others/` are sibling folders next to it, not subfolders of the Git repository.

```text
M1/
├── project/                         # Git repository: code, README, runs
│   ├── Original Dataset/
│   │   ├── Fresh/                   # Original Fresh image examples
│   │   └── Rotten/                  # Original Rotten examples; prepared as Defect
│   ├── runs/                        # Training and evaluation outputs
│   └── *.py                         # Preparation, training, and analysis scripts
├── datasets/                        # Local sibling folder, outside project/
│   ├── tomato/                      # Prepared original train/val/test split
│   ├── tomato_experiment_v1/        # Original data plus the synthetic pilot
│   ├── synthetic_multi_scenes_pilot/
│   ├── synthetic_scenes_pilot/
│   ├── synthetic_backgrounds/
│   └── tomato_crop_cache/
└── others/                          # Visual Studio screenshots/log evidence
```

The main GitHub repository has the source `Original Dataset/` and several `runs/` outputs. The prepared `datasets/` sibling is not in the local repository tree, so do not assume that someone cloning the main repository will also receive those prepared/synthetic folders. The current local `main` branch is one commit behind `origin/main`; the local `README.md` is modified, and `12_evaluate_test_performance.py` plus `runs/slide7_baseline/` are not committed yet.

## Dataset report

The prepared original split under `../datasets/tomato/` contains 1,986 images:

| Split | Fresh | Defect | Total |
|---|---:|---:|---:|
| Train | 944 | 445 | 1,389 |
| Validation | 202 | 95 | 297 |
| Test | 204 | 96 | 300 |
| **Total** | **1,350** | **636** | **1,986** |

The training split is imbalanced (about 2.12 Fresh images per Defect image). The source folders use `Rotten`; the prepared folders and model outputs use `Defect`.

`../datasets/tomato_experiment_v1/` contains 2,006 images: the same 1,986 original images plus 20 synthetic training scenes (10 per class). The original validation and test sets remain unchanged. There are also 20 images in each of `../datasets/synthetic_scenes_pilot/` and `../datasets/synthetic_multi_scenes_pilot/`, 20 generated backgrounds, and 60 cached tomato crops.

## Results and run folders

The README reports the two models evaluated on the same 300 original test images:

| Test metric | Baseline YOLO11n-cls | Synthetic-data pilot |
|---|---:|---:|
| Accuracy | 96.67% (290/300) | 97.00% (291/300) |
| Defect recall | 95.83% (92/96) | 90.63% (87/96) |
| Fresh recall | 97.06% (198/204) | 100.00% (204/204) |
| Defect images predicted Fresh | 4 | 9 |
| Fresh images predicted Defect | 6 | 0 |

The synthetic pilot gains one correct test prediction overall but misses five more Defect images. Do not present the 0.33 percentage-point accuracy gain as proof that the proposed approach is better; Defect recall is the proposal's more important metric for this sorting use case.

Run folders and useful contents:

- `runs/classify/runs/baseline/` — baseline training outputs, `results.csv`, `args.yaml`, `results.png`, and saved `weights/best.pt` / `weights/last.pt`.
- `runs/experiment_synthetic/yolo11n_synthetic_pilot/` — the corresponding synthetic-pilot training outputs and weights.
- `runs/comparison_test/baseline_on_original_test/` and `runs/comparison_test/synthetic_on_original_test/` — side-by-side test evaluation outputs for the same original test split, including confusion matrices and sample predictions.
- `runs/prediction_analysis/` — `changed_predictions.csv`, `prediction_confidence.csv`, and the image contact sheet for predictions on which the models disagreed.
- `runs/slide7_baseline/` — local baseline test artifacts from the additional evaluation script: test confusion matrix, ROC curve, and `test_predictions.csv`. This folder is currently local/uncommitted.

The recorded training settings are:

- Baseline: pretrained `yolo11n-cls.pt`, 224-pixel images, batch 32, maximum 30 epochs, patience 10, seed 42; `results.csv` ends at epoch 18 (zero-based) and records 1,822.39 seconds.
- Synthetic pilot: same model family and image size, batch 16, maximum 20 epochs, patience 5, seed 42; `results.csv` ends at epoch 11 and records 1,001.05 seconds.
- The recorded `device` setting is blank (automatic selection), so the hardware is not identified in the run settings. Do not guess a GPU/CPU name. The elapsed times are the values in each run's `results.csv`.
- These test metrics are from the held-out test outputs/README, not the final validation accuracy in the training `results.csv`.

## Which images/plots to insert

Use the actual saved plots/images rather than the screenshots in `../others/`:

| Slide purpose | File or folder to open |
|---|---|
| Show original Fresh examples | `Original Dataset/Fresh/` |
| Show original Defect examples | `Original Dataset/Rotten/` |
| Show generated synthetic scenes | `../datasets/synthetic_multi_scenes_pilot/train/Fresh/` and `../datasets/synthetic_multi_scenes_pilot/train/Defect/` |
| Show the baseline learning curves | `runs/classify/runs/baseline/results.png` |
| Show the baseline test confusion matrix | `runs/slide7_baseline/confusion_matrix_test.png` |
| Show the baseline test ROC curve and AUC | `runs/slide7_baseline/roc_curve_test.png` |
| Compare baseline and synthetic test matrices | `runs/comparison_test/baseline_on_original_test/confusion_matrix.png` and `runs/comparison_test/synthetic_on_original_test/confusion_matrix.png` |
| Show which predictions changed | `runs/prediction_analysis/changed_predictions_contact_sheet.jpg` |
| Show confidence numbers for changed predictions | `runs/prediction_analysis/prediction_confidence.csv` |

The two folders under `Original Dataset/` contain individual tomato pictures. The synthetic pilot contains composed scenes. Neither one is a collected, real conveyor-belt test set.

## Missing presentation content, slide by slide

| Slide | Status | What to add or correct |
|---|---|---|
| 1 — Title | Mostly complete | Keep the project title and team names. |
| 2 — Problem | Covered | The diagram is a proposed workflow. Clarify that the current code classifies an input image; it does not locate individual tomatoes in a live belt stream or control a sorter. |
| 3 — Related work | Covered | Keep the cited comparisons and the selected YOLO11-cls baseline. Explain that the reported results from different datasets are not directly comparable. |
| 4 — Dataset | Covered | The current prepared split totals 1,986 images as listed above. Use an actual image example if the slide needs a visual. |
| 5 — Proposed work | Covered, with a status caveat | Keep the baseline and the two planned contributions. Do not imply that the belt-style test or calibrated reject option is already complete. |
| 6 — Baseline progress | **Incomplete** | Replace all `[Status]` and `[N]` placeholders. Suggested factual status: baseline trained and evaluated offline; not deployed. Report model, 224 input, batch 32, 30-epoch maximum/early stopping, seed 42, and actual test metrics. Hardware is not recorded. Mention training time only as logged in `results.csv`; include real problems/fixes only if the team can confirm them. |
| 7 — Initial results | **Incomplete** | Replace `[X]` and `[ ]` values and insert plots. Baseline test accuracy is 96.67%; Defect recall is 95.83%; Defect precision is 93.88%; Defect F1 is about 94.85%; Fresh precision is 98.02%; Fresh recall is 97.06%; Fresh F1 is about 97.54%; ROC AUC is 0.96. Use the confusion matrix and ROC paths above. Clearly label this as the 300-image original test split. |
| 8 — Proposed-solution progress | **Incomplete** | Contribution 1: the synthetic-scene pipeline and a 20-image pilot exist, and the synthetic model was tested on the original test set. A real belt-style test set is not present. Contribution 2: confidence comparison exists, but there is no implemented/calibrated reject option, temperature scaling, ECE/reliability diagram, or risk-coverage result in the available artifacts. Mark these as not yet done. |
| 9 — Findings | **Partly incomplete** | Replace the two remaining placeholders. Good evidence-based findings: the data is imbalanced; baseline missed 4/96 Defect images versus 9/96 for the synthetic pilot; synthetic training removed the baseline's 6 Fresh-to-Defect errors but increased missed defects; the small pilot does not prove an improvement. Use the disagreement contact sheet. |
| 10 — Next steps/timeline | **Incomplete** | Replace every `[Week/date]` and `[name]` with real dates and owners. Include the remaining belt-style data/evaluation, calibration/reject option if still in scope, repeatability checks, and final presentation/video preparation. Do not invent dates or ownership. |
| 11 — Team contributions | **Incomplete** | Fill both members' actual technical contributions. For your coding contribution, list only work you personally did (for example, data preparation, training/evaluation scripts, synthetic-data pipeline, or error analysis if accurate). Record the partner's actual technical work separately. Presentation design does not count. |
| 12 — References | Covered | Keep the references and check that every comparison on slide 3 is cited. |

One important scope difference to say aloud: the proposal describes detecting tomatoes and defects at a packing line, but the current trained YOLO11 model is a **classifier** trained on tomato images. It does not yet find each tomato in a crowded scene, track a belt, or trigger physical sorting.

## Deliverables and speaking time

The deck alone is not the full milestone submission. The instructions also require:

1. Submitted presentation slides.
2. A recorded presentation video between 5 and 7 minutes.
3. A live presentation of at most 7 minutes, followed by the instructors' reserved 2 minutes.

Rehearse and time the final version. A 12-slide deck can fit only if each slide is concise and the speaker focuses on the results and remaining work.
