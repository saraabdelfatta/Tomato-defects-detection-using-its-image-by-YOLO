# Tomato Fresh/Defect Classification — ML Project

## 1. Project overview

This project explores automatic image classification of tomatoes into two classes:

- **Fresh**
- **Defect** (the original `Rotten` class was mapped to `Defect` during preparation)

The baseline uses a pretrained **Ultralytics YOLO11n-cls** image-classification model and fine-tunes it on the prepared tomato dataset. A small synthetic-data experiment was then performed by composing multiple tomato crops onto generated belt-like backgrounds.

**Important:** this is currently an image-classification prototype. It does not yet detect individual tomatoes in a live conveyor-belt scene, draw bounding boxes, or perform reliable instance segmentation. Object detection and improved cropping/segmentation remain future work.

## 2. Current experimental results

Both models were evaluated on the same original test split of 300 images.

| Metric | Baseline | Synthetic-data experiment |
|---|---:|---:|
| Top-1 test accuracy | 96.67% (290/300) | 97.00% (291/300) |
| Defect recall | 95.83% (92/96) | 90.63% (87/96) |
| Fresh recall | 97.06% (198/204) | 100.00% (204/204) |
| Defect images predicted as Fresh | 4 | 9 |
| Fresh images predicted as Defect | 6 | 0 |

The experimental model improved overall test accuracy by one correct prediction, but it missed more Defect tomatoes. Therefore, the 0.33-percentage-point accuracy increase is not enough to conclude that synthetic data improved the system overall. For a sorting application, Defect recall is especially important.

The synthetic-data experiment is a small pilot, not a final result. Repeatability across multiple training seeds and a more realistic synthetic-image pipeline have not yet been established.

## 3. Repository and dataset layout

The project was developed on Windows. The paths below describe the current development layout; change them if the project is moved to another computer.

```text
project/
├── 01_prepare_data.py
├── 02_train_baseline.py
├── 03_evaluate_baseline.py
├── 04_create_synthetic_data.py
├── 05_create_synthetic_scenes.py
├── 06_prepare_experiment_data.py
├── 07_train_synthetic_experiment.py
├── 08_compare_test_results.py
├── 09_analyze_model_predictions.py
├── 10_review_changed_predictions.py
├── 11_analyze_prediction_confidence.py
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
└── tomato_experiment_v1/
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
| `04_create_synthetic_data.py` | Generates prototype belt-like background images for synthetic-scene composition. |
| `05_create_synthetic_scenes.py` | Creates pilot scenes by extracting tomato crops and placing multiple crops on generated backgrounds. Cropping is approximate and may leave background fragments. |
| `06_prepare_experiment_data.py` | Builds the experimental dataset by adding synthetic scenes to the training data while retaining original validation and test images. |
| `07_train_synthetic_experiment.py` | Trains a separate YOLO11n-cls model using the experimental dataset. It does not overwrite the baseline weights. |
| `08_compare_test_results.py` | Evaluates the baseline and experimental models on the same original test split and prints their Top-1 accuracy comparison. |
| `09_analyze_model_predictions.py` | Computes per-class precision, recall, F1-score, confusion matrices, and the list of images where model predictions differ. |
| `10_review_changed_predictions.py` | Creates a contact sheet of disagreement images, showing the actual class and both models' predictions. |
| `11_analyze_prediction_confidence.py` | Compares the predicted class and confidence values for disagreement images and saves a CSV report. |

## 5. Important output files and folders

The current runs are saved under `project/runs/`.

- Baseline weights: `runs/classify/runs/baseline/weights/best.pt`
- Baseline last checkpoint: `runs/classify/runs/baseline/weights/last.pt`
- Synthetic experiment best weights: `runs/experiment_synthetic/yolo11n_synthetic_pilot/weights/best.pt`
- Test comparison outputs: `runs/comparison_test/`
- Prediction analysis CSV: `runs/prediction_analysis/changed_predictions.csv`
- Disagreement contact sheet: `runs/prediction_analysis/changed_predictions_contact_sheet.jpg`
- Confidence analysis CSV: `runs/prediction_analysis/prediction_confidence.csv`

Ultralytics may create nested run directories depending on the `project` setting and installed version. If a path is missing, inspect the actual run output printed by the script rather than assuming the folder layout.

## 6. Requirements and installation

Use **Python 3.10 or 3.11** as a practical starting point, and use the same Python environment for all scripts. Exact compatibility depends on the installed PyTorch and Ultralytics versions.

### Main libraries

- **ultralytics** — YOLO model loading, training, prediction, and evaluation.
- **torch** — model training and inference backend; normally installed as an Ultralytics dependency.
- **opencv-python** — image loading, resizing, image composition, and contact-sheet generation.
- **Pillow** — creation and editing of synthetic background images.
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


## 7. How to run the workflow

Run the scripts from the `project` directory, with `.venv` activated.

```powershell
python 01_prepare_data.py
python 02_train_baseline.py
python 03_evaluate_baseline.py
python 04_create_synthetic_data.py
python 05_create_synthetic_scenes.py
python 06_prepare_experiment_data.py
python 07_train_synthetic_experiment.py
python 08_compare_test_results.py
python 09_analyze_model_predictions.py
python 10_review_changed_predictions.py
python 11_analyze_prediction_confidence.py
```

**Do not blindly rerun all scripts in order.** Some scripts may refuse to overwrite existing output folders, retrain models, or regenerate datasets. For the current project, the baseline and experimental models have already been trained and evaluated. For analysis only, scripts 08–11 can be run against the existing weights and outputs, provided the expected paths exist.

If a script reports that a dataset or model path cannot be found, compare the path in the error with the actual folder layout on your machine.

## 8. Methodology

1. Prepare a class-folder dataset with train, validation, and test splits.
2. Train a YOLO11n-cls baseline on the original training images.
3. Evaluate the baseline on the original held-out test set.
4. Generate prototype conveyor-belt-like backgrounds.
5. Compose multiple tomato crops onto backgrounds to produce a small synthetic-scene pilot.
6. Add synthetic scenes to the training set only.
7. Train a separate experimental classifier.
8. Evaluate both models on the same original test images.
9. Inspect per-class errors, disagreement images, and confidence values.

The scene-generation step uses approximate foreground extraction, not a trained tomato detector or a validated instance-segmentation model. Consequently, synthetic crops can contain unwanted background and may create unrealistic examples.

## 9. Limitations and next steps

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

## 10. References

- Dataset: Tomato Maturity Detection and Quality Grading, Kaggle  
  https://www.kaggle.com/datasets/sujaykapadnis/tomato-maturity-detection-and-quality-grading
- Ultralytics documentation: https://docs.ultralytics.com/
- PyTorch installation guide: https://pytorch.org/get-started/locally/

---
NOTE 
- you will find a text called requirements-lock will have all the instualtion you need.
- there is a folder called otehrs this is a screenshots from the VC in order to track the model.

**Project status:** baseline classification and a small synthetic-data experiment have been completed. The README should be updated whenever paths, dependencies, scripts, or results change.
