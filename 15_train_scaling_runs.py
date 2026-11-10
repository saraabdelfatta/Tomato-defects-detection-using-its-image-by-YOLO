"""Trains one YOLO11n-cls per dataset size (~30 min each on your laptop).
IMPORTANT: EPOCHS / IMGSZ / BATCH / SEED must be IDENTICAL to 02_train_baseline.py."""
from ultralytics import YOLO
from tomato_utils import DATASETS, PROJECT, IMGSZ

SIZES = [100, 250, 500, 1000]
EPOCHS, BATCH, PATIENCE, SEED = 30, 32, 10, 42
RUNS = PROJECT / "runs" / "experiment_scaling"

for n in SIZES:
    dataset = DATASETS / f"tomato_scaling_N{n}"
    if not dataset.is_dir():
        raise FileNotFoundError(
            f"Scaling dataset not found: {dataset}\n"
            "Run 13_build_scaling_datasets.py first."
        )
    best = RUNS / f"N{n}" / "weights" / "best.pt"
    if best.exists():
        print(f"skip N={n} (already trained)")
        continue
    model = YOLO(str(PROJECT / "yolo11n-cls.pt"))
    model.train(
        data=str(dataset),
        epochs=EPOCHS,
        imgsz=IMGSZ,
        batch=BATCH,
        patience=PATIENCE,
        seed=SEED,
        project=str(RUNS),
        name=f"N{n}",
        exist_ok=True,
    )
