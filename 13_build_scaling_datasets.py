"""Builds tomato_scaling_N{100,250,500,1000}: all real train images + N synthetic v2 scenes
(train only). Subsets are NESTED (N=100 is inside N=250, ...) so only N changes between runs.
Val/test stay real-only, identical to the baseline."""
import os
import random
import shutil
from pathlib import Path

PROJECT = Path(__file__).resolve().parent
DATASETS = PROJECT.parent / "datasets"
REAL = DATASETS / "tomato"
SYN_DIR = DATASETS / "synthetic_scenes_v2"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SIZES = [100, 250, 500, 1000]
SEED = 42


def link(src, dst):
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(src, dst)                       # hard link: no extra disk space
    except OSError:
        shutil.copy2(src, dst)


def images(d):
    if not d.is_dir():
        return []
    return sorted(
        p for p in d.rglob("*")
        if p.is_file() and p.suffix.lower() in IMG_EXT
    )


def is_complete_dataset(out, n):
    """Check that an existing output contains the requested real and synthetic images."""
    total_synthetic = 0
    for split in ("train", "val", "test"):
        for class_name in ("Fresh", "Defect"):
            output_images = images(out / split / class_name)
            real_images = images(REAL / split / class_name)
            real_names = {path.name for path in real_images}
            output_real_names = {
                path.name for path in output_images
                if not path.name.startswith("syn_")
            }
            if output_real_names != real_names:
                return False
            synthetic_count = sum(
                path.name.startswith("syn_") for path in output_images
            )
            if split == "train":
                total_synthetic += synthetic_count
            elif synthetic_count:
                return False
    return total_synthetic == n


def main():
    if not SYN_DIR.is_dir():
        raise FileNotFoundError(
            f"Synthetic scene folder not found: {SYN_DIR}\n"
            "Generate 500 scenes per class first with:\n"
            "python 05_create_synthetic_scenes.py --scenes-per-class 500 "
            "--output-dir ..\\datasets\\synthetic_scenes_v2"
        )

    pool = [
        (path, class_name)
        for class_name in ("Fresh", "Defect")
        for path in images(SYN_DIR / class_name)
    ]
    required = max(SIZES)
    if len(pool) < required:
        class_counts = {
            class_name: len(images(SYN_DIR / class_name))
            for class_name in ("Fresh", "Defect")
        }
        raise ValueError(
            f"Need at least {required} synthetic scenes in {SYN_DIR}, "
            f"found {len(pool)} "
            f"(Fresh: {class_counts['Fresh']}, "
            f"Defect: {class_counts['Defect']}). "
            "The existing conveyor pilot has only 10 scenes per class. "
            "Generate the larger set using the command shown above."
        )

    missing_real = [
        REAL / split / class_name
        for split in ("train", "val", "test")
        for class_name in ("Fresh", "Defect")
        if not images(REAL / split / class_name)
    ]
    if missing_real:
        raise FileNotFoundError(
            "Missing real dataset images in:\n"
            + "\n".join(str(path) for path in missing_real)
        )

    random.Random(SEED).shuffle(pool)

    for n in SIZES:
        out = DATASETS / f"tomato_scaling_N{n}"
        if out.exists():
            if not is_complete_dataset(out, n):
                raise RuntimeError(
                    f"Existing dataset is incomplete or does not match N={n}: "
                    f"{out}. Move it aside before rebuilding."
                )
            print(f"skip {out} (already complete)")
            continue
        for split in ("train", "val", "test"):
            for class_name in ("Fresh", "Defect"):
                for path in images(REAL / split / class_name):
                    link(path, out / split / class_name / path.name)
        for path, class_name in pool[:n]:
            link(
                path,
                out / "train" / class_name / f"syn_{path.name}",
            )
        defect_count = sum(
            1 for _, class_name in pool[:n] if class_name == "Defect"
        )
        print(
            f"{out.name}: +{n} synthetic "
            f"({defect_count} Defect, {n - defect_count} Fresh)"
        )


if __name__ == "__main__":
    main()
