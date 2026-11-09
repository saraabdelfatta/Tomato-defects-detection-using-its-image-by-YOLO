# Tomato Fresh/Defect Classification — ML Project

## 1. Project overview

This project explores automatic image classification of tomatoes into two classes:

- **Fresh**
- **Defect** (the original `Rotten` class was mapped to `Defect` during preparation)

The baseline uses a pretrained **Ultralytics YOLO11n-cls** image-classification model and fine-tunes it on the prepared tomato dataset. Two small synthetic-data pilots were then tested: one using generated belt-like backgrounds and a separate one compositing multiple class-matched tomato crops onto the supplied industrial conveyor photo.

**Important:** this is currently an image-classification prototype. It does not yet detect individual tomatoes in a live conveyor-belt scene, draw bounding boxes, or perform reliable instance segmentation. Object detection and improved cropping/segmentation remain future work.

## 2. Results from the earlier synthetic-data pilot

Both models were evaluated on the same original test split of 300 images.

| Metric | Baseline | Synthetic-data experiment |
|---|---:|---:|
| Top-1 test accuracy | 96.67% (290/300) | 97.00% (291/300) |
| Defect recall | 95.83% (92/96) | 90.63% (87/96) |
| Fresh recall | 97.06% (198/204) | 100.00% (204/204) |
| Defect images predicted as Fresh | 4 | 9 |
| Fresh images predicted as Defect | 6 | 0 |

The experimental model improved overall test accuracy by one correct prediction, but it missed more Defect tomatoes. Therefore, the 0.33-percentage-point accuracy increase is not enough to conclude that synthetic data improved the system overall. For a sorting application, Defect recall is especially important.

The earlier synthetic-data experiment is a small pilot, not a final result. The separate conveyor-photo pilot results are reported below.

### Conveyor-photo pilot

The pilot contains 10 Fresh and 10 Defect synthetic scenes, with 5–9 class-matched tomato crops per scene. These 20 scenes were added to the training split only; the validation and 300-image test splits are unchanged. Training stopped early after 11 epochs, with the best checkpoint from epoch 6 (97.64% validation accuracy). This is one run with seed 42.

| Test metric | Baseline | Conveyor-photo experiment |
|---|---:|---:|
| Top-1 accuracy | 96.67% (290/300) | 98.00% (294/300) |
| Defect precision | 93.88% | 96.88% |
| Defect recall | 95.83% (92/96) | 96.88% (93/96) |
| Defect F1 | 94.85% | 96.88% |
| Fresh precision | 98.02% | 98.53% |
| Fresh recall | 97.06% (198/204) | 98.53% (201/204) |
| Fresh F1 | 97.54% | 98.53% |
| Defect images predicted as Fresh | 4 | 3 |
| Fresh images predicted as Defect | 6 | 3 |

On this test set, the conveyor-photo model made four more correct predictions overall; among the eight images where predictions differed, six changed from wrong to correct and two changed from correct to wrong. This is an encouraging pilot result, not proof of general improvement: it uses one training seed and a small set of synthetic scenes, and the color-mask crop boundaries remain approximate. The model is still an image classifier, not an individual-tomato detector for a live conveyor.

Detailed test outputs are under `runs/comparison_test_conveyor/`; per-image disagreements and confidence values are under `runs/prediction_analysis_conveyor/`.

## 3. Repository and dataset layout

The project was developed on Windows. The paths below describe the current development layout; change them if the project is moved to another computer.

```text
project/
├── 01_prepare_data.py
├── 02_train_baseline.py
├── 03_evaluate_baseline.py
├── 05_create_synthetic_scenes.py
├── 06_prepare_experiment_data.py
├── 07_train_synthetic_experiment.py
├── 08_compare_test_results.py
├── 09_analyze_model_predictions.py
├── 10_review_changed_predictions.py
├── 11_analyze_prediction_confidence.py
├── 12_evaluate_test_performance.py
├── assets/
│   └── industrial_conveyor.png
├── README.md
├── .venv/                         # local Python virtual environment
└── runs/                          # generated model weights and reports

datasets/
├── tomato/
│   ├── train/
│   │   ├── Fresh/
│   │   └── Defect/
│   ├── val/
│   │   ├── Fresh/
│   │   └── Defect/
│   └── test/
│       ├── Fresh/
│       └── Defect/
├── synthetic_backgrounds/
├── synthetic_multi_scenes_pilot/
├── tomato_crop_cache/
├── tomato_experiment_v1/
│   ├── train/
│   │   ├── Fresh/
│   │   └── Defect/
│   ├── val/
│   │   ├── Fresh/
│   │   └── Defect/
│   └── test/
│       ├── Fresh/
│       └── Defect/
├── synthetic_conveyor_scenes_pilot/
│   └── train/
│       ├── Fresh/
│       └── Defect/
├── tomato_conveyor_crop_cache_v7/
│   ├── Fresh/
│   └── Defect/
└── tomato_conveyor_experiment_v1/
    ├── train/
    │   ├── Fresh/
    │   └── Defect/
    ├── val/
    │   ├── Fresh/
    │   └── Defect/
    └── test/
        ├── Fresh/
        └── Defect/
```

The dataset folders are outside the `project/` directory in the current setup. Some generated folders may not exist if they have been moved or cleaned up.

### Dataset split

The original prepared dataset contains 1,986 images:

| Split | Fresh | Defect | Total |
|---|---:|---:|---:|
| Train | 944 | 445 | 1,389 |
| Validation | 202 | 95 | 297 |
| Test | 204 | 96 | 300 |
| **Total** | **1,350** | **636** | **1,986** |

The original validation and test sets were kept free of synthetic images. The experiment adds synthetic scenes to the training split only. The original test split is used for both models so the comparison is on the same images.

## 4. Python scripts and their purposes

| File | Purpose |
|---|---|
| `01_prepare_data.py` | Organizes the source images into Fresh/Defect train, validation, and test splits. The source `Rotten` class is mapped to `Defect`. It also records a split manifest. |
| `02_train_baseline.py` | Fine-tunes the baseline YOLO11n-cls model on the original prepared training data. |
| `03_evaluate_baseline.py` | Evaluates the baseline model and reports classification metrics on the held-out test set. |
| `05_create_synthetic_scenes.py` | Extracts class-specific tomato groups from training images with a saturation mask and composites them onto `assets/industrial_conveyor.png`. Masks are approximate and must be reviewed. |
| `06_prepare_experiment_data.py` | Builds a separate conveyor experiment dataset with synthetic scenes in training only; validation and test remain original. |
| `07_train_synthetic_experiment.py` | Trains a separate YOLO11n-cls conveyor experiment without overwriting baseline or earlier pilot weights. |
| `08_compare_test_results.py` | Evaluates the baseline and conveyor experiment on the same original test split. |
| `09_analyze_model_predictions.py` | Computes per-class precision, recall, F1-score, confusion matrices, and the list of images where baseline and conveyor-model predictions differ. |
| `10_review_changed_predictions.py` | Creates a contact sheet of disagreement images, showing the actual class and both models' predictions. |
| `11_analyze_prediction_confidence.py` | Compares the predicted class and confidence values for disagreement images and saves a CSV report. |
| `12_evaluate_test_performance.py` | Evaluates the baseline model on the held-out test set, saving per-image Defect probabilities, a confusion matrix, and an ROC curve. |

## 5. Important output files and folders

The current runs are saved under `project/runs/`.

- Baseline weights: `runs/classify/runs/baseline/weights/best.pt`
- Baseline last checkpoint: `runs/classify/runs/baseline/weights/last.pt`
- Synthetic experiment best weights: `runs/experiment_synthetic/yolo11n_synthetic_pilot/weights/best.pt`
- Conveyor experiment best weights: `runs/experiment_conveyor/yolo11n_conveyor_synthetic_pilot/weights/best.pt`
- Conveyor synthetic scenes: `../datasets/synthetic_conveyor_scenes_pilot/train/`
- Earlier pilot comparison outputs: `runs/comparison_test/`
- Conveyor comparison outputs: `runs/comparison_test_conveyor/`
- Conveyor prediction analysis CSV: `runs/prediction_analysis_conveyor/changed_predictions.csv`
- Conveyor disagreement contact sheet: `runs/prediction_analysis_conveyor/changed_predictions_contact_sheet.jpg`
- Conveyor confidence analysis CSV: `runs/prediction_analysis_conveyor/prediction_confidence.csv`
- Baseline test evaluation: `runs/slide7_baseline/`

Ultralytics may create nested run directories depending on the `project` setting and installed version. If a path is missing, inspect the actual run output printed by the script rather than assuming the folder layout.

## 6. Milestone 1 presentation: what to fill and where to find it

The current slide deck is `../Presentation2.pptx` (relative to this repository). The main incomplete slides are 6–11. Use the following checklist to fill the existing slides; it is not necessary to add every detail if it makes the talk exceed seven minutes.

| Slide | Information to add | Local file or folder to use |
|---|---|---|
| 2 — Problem | A visual example of Fresh and Defect tomatoes, alongside the proposed sorting workflow. Make clear the live conveyor-belt system is the goal, not the current implementation. | `Original Dataset/Fresh/` and `Original Dataset/Rotten/` |
| 4 — Dataset | Dataset source, split sizes, class imbalance, and (optionally) example images. The prepared split has 1,986 images: train 1,389, validation 297, test 300. | `../datasets/tomato/`; examples in `Original Dataset/` |
| 5 — Proposed work | State what is implemented versus planned: baseline, an initial synthetic pilot, and a conveyor-photo pilot are implemented; a real belt-style test and calibrated reject/review option are not evidenced yet. | `05_create_synthetic_scenes.py`–`08_compare_test_results.py`; `../datasets/synthetic_conveyor_scenes_pilot/train/` |
| 6 — Baseline progress | Report that YOLO11n-cls was fine-tuned and evaluated offline. Settings: 224-pixel input, batch 32, maximum 30 epochs, patience 10, seed 42. Hardware is not recorded in the run configuration. | `runs/classify/runs/baseline/args.yaml`, `results.csv`, and `results.png` |
| 7 — Initial results | On the original 300-image test set: accuracy 96.67%, Defect recall 95.83%, Fresh recall 97.06%. Add the test confusion matrix and ROC curve. | `runs/slide7_baseline/confusion_matrix_test.png`, `runs/slide7_baseline/roc_curve_test.png`, and `runs/slide7_baseline/test_predictions.csv` |
| 8 — Proposed-solution progress | Show the conveyor-photo scenes and measured test results: 98.00% accuracy and 96.88% Defect recall versus the baseline's 96.67% and 95.83%. Clearly state that no real belt-style test set or calibrated reject-option results are available yet. | `../datasets/synthetic_conveyor_scenes_pilot/train/Fresh/`, `../datasets/synthetic_conveyor_scenes_pilot/train/Defect/`, and `runs/comparison_test_conveyor/` |
| 9 — Findings | Compare the conveyor model with the baseline using its measured metrics; do not assume the previous pilot's result carries over. | `runs/prediction_analysis_conveyor/changed_predictions_contact_sheet.jpg`, `changed_predictions.csv`, and `prediction_confidence.csv` |
| 10 — Next steps | Add actual dates and owners for remaining experiments and final deliverables. These must be agreed by the team; do not invent them. | Team schedule (not recorded in the current run files) |
| 11 — Team contributions | List each member's actual technical work only. Do not count slide design; the code/history alone cannot establish who personally completed each task. | Confirm responsibilities with each team member |

### Confusion matrix path clarification

The example path `auc/basline.matrix.png` is not present in this project. The baseline **test-set** confusion matrix to use is:

```text
runs/slide7_baseline/confusion_matrix_test.png
```

Other matrices exist, but check their purpose before using them: `runs/classify/runs/baseline/confusion_matrix.png` is a training/validation run plot. The earlier pilot uses `runs/comparison_test/`; the conveyor experiment uses `runs/comparison_test_conveyor/baseline_on_original_test/confusion_matrix.png` and `runs/comparison_test_conveyor/conveyor_on_original_test/confusion_matrix.png`.

### Milestone deliverables and time limit

Submit the completed slides, a recorded presentation of **5–7 minutes**, and be ready for the live presentation of no more than **7 minutes** (the remaining two minutes are reserved for instructor questions/feedback). Rehearse with a timer. The deck should report observed work honestly: this project currently classifies tomato images and is not a deployed detector/sorter for a live conveyor belt.

## 7. Requirements and installation

Use **Python 3.10 or 3.11** as a practical starting point, and use the same Python environment for all scripts. Exact compatibility depends on the installed PyTorch and Ultralytics versions.

### Main libraries

- **ultralytics** — YOLO model loading, training, prediction, and evaluation.
- **torch** — model training and inference backend; normally installed as an Ultralytics dependency.
- **opencv-python** — image loading, resizing, image composition, and contact-sheet generation.
- **numpy** — array and image operations.

The scripts also use Python standard-library modules such as `pathlib`, `csv`, `math`, and `random`; these do not need separate installation. Pandas is not required by the scripts described here.

### Create and activate a virtual environment (Windows PowerShell)

Run these commands from the `project` directory:


If Python 3.11 is not installed, install a compatible Python version first or use an already installed supported version.

### Install libraries

```powershell
python -m pip install ultralytics opencv-python Pillow numpy
```

PyTorch installation can vary by operating system and whether GPU acceleration is needed. The command above normally installs compatible dependencies, but for a CUDA-enabled setup, follow the official PyTorch installation selector and verify the installed configuration.

Check the environment:

```powershell
python --version
python -m pip show ultralytics torch opencv-python Pillow numpy
python -c "import ultralytics, torch, cv2, PIL, numpy; print('Imports successful'); print('Ultralytics:', ultralytics.__version__); print('PyTorch:', torch.__version__)"
```

For reproducibility, record the versions used in the working environment:


## 8. How to run the workflow

Run the scripts from the `project` directory, with `.venv` activated.

```powershell
python 01_prepare_data.py
python 02_train_baseline.py
python 03_evaluate_baseline.py
python 05_create_synthetic_scenes.py
python 06_prepare_experiment_data.py
python 07_train_synthetic_experiment.py
python 08_compare_test_results.py
python 09_analyze_model_predictions.py
python 10_review_changed_predictions.py
python 11_analyze_prediction_confidence.py
python 12_evaluate_test_performance.py
```

**Do not blindly rerun all scripts in order.** Some scripts may refuse to overwrite existing output folders, retrain models, or regenerate datasets. In this workspace, the baseline and both synthetic pilots have already been run; the conveyor experiment's dataset, model, test comparison, and analyses are saved in their separate folders. For a fresh run, use scripts 05–08 for the conveyor experiment and scripts 09–11 for its prediction analysis.

The earlier generated-background pilot is retained as a historical experiment. The current conveyor-photo workflow uses the supplied `assets/industrial_conveyor.png` background directly.

If a script reports that a dataset or model path cannot be found, compare the path in the error with the actual folder layout on your machine.

## 9. Methodology

1. Prepare a class-folder dataset with train, validation, and test splits.
2. Train a YOLO11n-cls baseline on the original training images.
3. Evaluate the baseline on the original held-out test set.
4. The earlier pilot generated belt-like backgrounds; those outputs are retained for historical comparison.
5. Composite class-matched tomato crops onto the provided industrial conveyor photo for the new pilot.
6. Add the new synthetic scenes to training only; preserve original validation and test sets.
7. Train a separate conveyor synthetic-data classifier.
8. Evaluate the baseline and conveyor experiment on the same original test images.
9. Inspect per-class errors, disagreement images, confidence values, and the generated scenes.

The conveyor scene-generation step uses a saturation mask, not a trained tomato detector or validated instance-segmentation model. It is tailored to the pale tabletop in these source photos; inspect every generated crop/scene for missing tomato pixels or background fragments before treating the experiment as evidence.

## 10. Limitations and next steps

- The current model classifies an image; it does not locate and classify each tomato separately in a crowded scene.
- Synthetic scene generation is a small pilot, and crop boundaries are imperfect.
- The accuracy difference is only one test image and should not be treated as proof of improvement.
- The experimental model has lower Defect recall than the baseline, which is important for a sorting use case.
- Confidence scores are not guarantees of correctness and do not prove the cause of an error.
- Test images should not be used to tune a threshold repeatedly; keep them as a final evaluation set.

Recommended future work:

1. Inspect the original dataset's labeling criteria and ambiguous examples.
2. Improve tomato localization and foreground segmentation before compositing.
3. Generate more varied, realistic synthetic scenes while keeping labels unambiguous.
4. Compare several training seeds and report mean and variability.
5. Consider precision-recall curves and threshold selection on validation data, especially if missing a Defect tomato is costly.
6. If the eventual application needs individual tomato localization on a conveyor, build and evaluate an object-detection or instance-segmentation pipeline separately.

## 11. References

- Dataset: Tomato Maturity Detection and Quality Grading, Kaggle  
  https://www.kaggle.com/datasets/sujaykapadnis/tomato-maturity-detection-and-quality-grading
- Ultralytics documentation: https://docs.ultralytics.com/
- PyTorch installation guide: https://pytorch.org/get-started/locally/

---
NOTE 
- you will find a text called requirements-lock will have all the instualtion you need.
- there is a folder called otehrs this is a screenshots from the VC in order to track the model.

**Project status:** baseline classification and a small synthetic-data experiment have been completed. The README should be updated whenever paths, dependencies, scripts, or results change.
