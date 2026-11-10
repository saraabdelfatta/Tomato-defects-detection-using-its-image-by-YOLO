
"""
FILE: 05_create_synthetic_scenes.py

PURPOSE:
    Generate a conveyor-photo pilot by extracting tomato groups
    from training photos and composing them onto a real belt.

INPUT:
    datasets/tomato/train/Fresh
    datasets/tomato/train/Defect
    assets/industrial_conveyor.png

OUTPUT:
    datasets/synthetic_conveyor_scenes_pilot/train/Fresh
    datasets/synthetic_conveyor_scenes_pilot/train/Defect
    datasets/tomato_conveyor_crop_cache_v7/Fresh
    datasets/tomato_conveyor_crop_cache_v7/Defect

IMPORTANT:
    - Uses only the existing training images.
    - Processes a limited sample to reduce runtime.
    - Uses tomato-color segmentation to remove the pale tabletop.
    - Keeps each synthetic scene within one class.
    - Does not modify the original dataset or test set.

NEXT STEP:
    Inspect scenes, then run scripts 06-08 to prepare, train,
    and compare the conveyor experiment.
"""

from pathlib import Path
import argparse
import random
import cv2
import numpy as np

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_PATH.parent / "datasets"

SOURCE_ROOT = DATASET_ROOT / "tomato" / "train"
BACKGROUND_PATH = PROJECT_PATH / "assets" / "industrial_conveyor.png"

OUTPUT_ROOT = (
    DATASET_ROOT / "synthetic_conveyor_scenes_pilot" / "train"
)
CACHE_ROOT = DATASET_ROOT / "tomato_conveyor_crop_cache_v7"

CLASSES = ["Fresh", "Defect"]

# ------------------------------------------------------------
# 2. SPEED AND IMAGE SETTINGS
# ------------------------------------------------------------

# Start with a manageable crop sample for visual review.
MAX_SOURCE_IMAGES_PER_CLASS = 30

# Number of scenes generated for each class.
SCENES_PER_CLASS = 10

# Number of tomatoes placed in each scene.
MIN_TOMATOES = 5
MAX_TOMATOES = 9

SCENE_WIDTH = 640
SCENE_HEIGHT = 480
BELT_TOP = 108
BELT_BOTTOM = 375

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


# ------------------------------------------------------------
# 3. FIND IMAGES
# ------------------------------------------------------------

def find_images(folder):
    """Return image files in a folder, sorted for reproducibility."""
    if not folder.exists():
        return []

    return sorted(
        path for path in folder.rglob("*")
        if path.is_file()
        and path.suffix.lower() in IMAGE_EXTENSIONS
    )


# ------------------------------------------------------------
# 4. EXTRACT A TOMATO CROP FROM ITS COLOR MASK
# ------------------------------------------------------------

def extract_crop(image_path):
    """
    Separate the saturated tomato group from the pale tabletop.

    The source photos have a low-saturation wood background and
    saturated tomato skins. Morphological closing fills small
    holes in the color mask; small disconnected specks are removed.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        raise OSError(f"Could not read source image: {image_path}")

    height, width = image.shape[:2]

    if width < 40 or height < 40:
        return None

    scale = min(1.0, 900 / max(width, height))
    if scale < 1:
        image = cv2.resize(
            image,
            (int(width * scale), int(height * scale)),
            interpolation=cv2.INTER_AREA,
        )
        height, width = image.shape[:2]

    saturation = cv2.cvtColor(
        image, cv2.COLOR_BGR2HSV
    )[:, :, 1]

    alpha = (saturation >= 75).astype(np.uint8) * 255
    alpha = cv2.morphologyEx(
        alpha,
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (5, 5),
        ),
    )
    alpha = cv2.morphologyEx(
        alpha,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (21, 21),
        ),
    )

    foreground_fraction = np.count_nonzero(alpha) / alpha.size

    if foreground_fraction < 0.03 or foreground_fraction > 0.90:
        return None

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        (alpha > 0).astype(np.uint8),
        connectivity=8,
    )
    if component_count > 1:
        largest_area = int(stats[1:, cv2.CC_STAT_AREA].max())
        keep = np.zeros_like(alpha)
        for component in range(1, component_count):
            area = int(stats[component, cv2.CC_STAT_AREA])
            if area >= max(100, largest_area * 0.05):
                keep[labels == component] = 255
        alpha = keep

    points = cv2.findNonZero(alpha)

    if points is None:
        return None

    x, y, crop_width, crop_height = cv2.boundingRect(points)

    if crop_width < 30 or crop_height < 30:
        return None

    padding_x = max(2, int(crop_width * 0.025))
    padding_y = max(2, int(crop_height * 0.025))
    x1 = max(0, x - padding_x)
    y1 = max(0, y - padding_y)
    x2 = min(width, x + crop_width + padding_x)
    y2 = min(height, y + crop_height + padding_y)

    image_crop = image[y1:y2, x1:x2]
    alpha_crop = alpha[y1:y2, x1:x2]
    alpha_crop = cv2.GaussianBlur(
        alpha_crop, (5, 5), 1.0
    )

    rgba = cv2.cvtColor(image_crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha_crop

    return rgba


# ------------------------------------------------------------
# 5. BUILD OR REUSE THE CROP CACHE
# ------------------------------------------------------------

def prepare_crops(class_name):
    """
    Reuse or extract a class-specific sample of tomato groups.
    """
    cache_folder = CACHE_ROOT / class_name
    cache_folder.mkdir(parents=True, exist_ok=True)

    source_folder = SOURCE_ROOT / class_name
    source_images = find_images(source_folder)

    if not source_images:
        raise FileNotFoundError(
            f"No training images found in: {source_folder}"
        )

    source_images = random.sample(
        source_images,
        min(MAX_SOURCE_IMAGES_PER_CLASS, len(source_images)),
    )

    cached_crops = sorted(cache_folder.glob("*.png"))

    if cached_crops:
        print(
            f"  Reusing {len(cached_crops)} cached "
            f"{class_name} crops."
        )
        return cached_crops

    print(
        f"  Processing {len(source_images)} source images "
        f"for {class_name}..."
    )

    saved = []

    for index, image_path in enumerate(source_images, start=1):
        crop = extract_crop(image_path)

        if crop is not None:
            output_path = cache_folder / f"crop_{index:04d}.png"

            if cv2.imwrite(str(output_path), crop):
                saved.append(output_path)

        # Visible progress instead of a long silent wait.
        if index % 5 == 0 or index == len(source_images):
            print(
                f"    {class_name}: checked "
                f"{index}/{len(source_images)}, "
                f"saved {len(saved)} crops."
            )

    if len(saved) < 5:
        raise RuntimeError(
            f"Only {len(saved)} usable crops were extracted for "
            f"{class_name}. Inspect the source images and masks."
        )

    print(f"  Cached {len(saved)} crops for {class_name}.")
    return saved


# ------------------------------------------------------------
# 6. LOAD A BACKGROUND AND RESIZE IT
# ------------------------------------------------------------

def load_background(background_path):
    """Load the supplied conveyor photo and resize it to scene size."""
    background = cv2.imread(str(background_path))

    if background is None:
        return None

    return cv2.resize(
        background,
        (SCENE_WIDTH, SCENE_HEIGHT),
        interpolation=cv2.INTER_AREA,
    )


# ------------------------------------------------------------
# 7. COMPOSITE ONE TRANSPARENT TOMATO CROP
# ------------------------------------------------------------

def paste_crop(scene, crop_path, x, y, target_width):
    """
    Resize a transparent crop and alpha-blend it into the scene.
    The crop is placed only if it fits within the scene.
    """
    crop = cv2.imread(str(crop_path), cv2.IMREAD_UNCHANGED)

    if crop is None or crop.ndim != 3 or crop.shape[2] != 4:
        return False

    original_height, original_width = crop.shape[:2]

    if original_width == 0 or original_height == 0:
        return False

    target_width = max(20, min(target_width, SCENE_WIDTH // 2))
    target_height = int(
        original_height * target_width / original_width
    )

    if target_height > SCENE_HEIGHT // 2:
        target_height = SCENE_HEIGHT // 2
        target_width = int(
            original_width * target_height / original_height
        )

    crop = cv2.resize(
        crop,
        (target_width, target_height),
        interpolation=cv2.INTER_AREA,
    )

    height, width = scene.shape[:2]

    # Do not allow a crop to extend outside the scene.
    x = max(0, min(x, width - target_width))
    y = max(0, min(y, height - target_height))

    roi = scene[
        y:y + target_height,
        x:x + target_width
    ]

    alpha = crop[:, :, 3].astype(np.float32) / 255.0
    alpha = alpha[:, :, np.newaxis]

    shadow = cv2.GaussianBlur(
        alpha[:, :, 0],
        (0, 0),
        sigmaX=7,
    )[:, :, np.newaxis]
    shadow_x = min(x + 4, width - target_width)
    shadow_y = min(y + 7, height - target_height)
    shadow_roi = scene[
        shadow_y:shadow_y + target_height,
        shadow_x:shadow_x + target_width,
    ].astype(np.float32)
    scene[
        shadow_y:shadow_y + target_height,
        shadow_x:shadow_x + target_width,
    ] = np.clip(
        shadow_roi * (1.0 - shadow * 0.20),
        0,
        255,
    ).astype(np.uint8)

    foreground = crop[:, :, :3].astype(np.float32)
    background = roi.astype(np.float32)

    blended = (
        foreground * alpha + background * (1.0 - alpha)
    )

    scene[
        y:y + target_height,
        x:x + target_width
    ] = np.clip(blended, 0, 255).astype(np.uint8)

    return True


# ------------------------------------------------------------
# 8. GENERATE SYNTHETIC SCENES
# ------------------------------------------------------------

def generate_scenes(class_name, crop_paths, backgrounds, output_root,
                    scenes_per_class):
    """Create synthetic scenes containing only one class."""
    output_folder = output_root / class_name
    output_folder.mkdir(parents=True, exist_ok=True)

    for scene_index in range(scenes_per_class):
        background_path = random.choice(backgrounds)
        scene = load_background(background_path)

        if scene is None:
            print(f"  Warning: could not read {background_path}")
            continue

        number_of_tomatoes = random.randint(
            MIN_TOMATOES, MAX_TOMATOES
        )

        placed = 0

        for _ in range(number_of_tomatoes):
            crop_path = random.choice(crop_paths)

            target_width = random.randint(68, 128)
            crop = cv2.imread(
                str(crop_path),
                cv2.IMREAD_UNCHANGED,
            )
            if crop is None:
                raise OSError(f"Could not read cached crop: {crop_path}")
            crop_height, crop_width = crop.shape[:2]
            target_height = min(
                int(crop_height * target_width / crop_width),
                SCENE_HEIGHT // 2,
            )
            target_width = min(target_width, SCENE_WIDTH // 3)
            x = random.randint(0, SCENE_WIDTH - target_width)
            max_y = max(
                BELT_TOP,
                min(BELT_BOTTOM - target_height, SCENE_HEIGHT - target_height),
            )
            y = random.randint(BELT_TOP, max_y)

            if paste_crop(scene, crop_path, x, y, target_width):
                placed += 1

        if placed < MIN_TOMATOES:
            raise RuntimeError(
                f"Only {placed} tomatoes were placed in "
                f"{class_name} scene {scene_index + 1}."
            )

        output_path = (
            output_folder / f"synthetic_{scene_index + 1:03d}.jpg"
        )

        if cv2.imwrite(str(output_path), scene):
            print(
                f"  {class_name} scene "
                f"{scene_index + 1}/{scenes_per_class} "
                f"saved ({placed} tomatoes): {output_path.name}"
            )


# ------------------------------------------------------------
# 9. MAIN
# ------------------------------------------------------------

def main(scenes_per_class=SCENES_PER_CLASS, output_root=OUTPUT_ROOT):
    if scenes_per_class < 1:
        raise ValueError("scenes_per_class must be at least 1")

    print("=" * 60)
    print("CONVEYOR-PHOTO SYNTHETIC SCENE GENERATION")
    print("=" * 60)
    print(f"Project folder: {PROJECT_PATH}")
    print(f"Source dataset: {SOURCE_ROOT}")
    print(f"Background photo: {BACKGROUND_PATH}")
    print(f"Output folder: {output_root}")
    print(f"Crop cache: {CACHE_ROOT}")
    print()

    if not BACKGROUND_PATH.is_file():
        raise FileNotFoundError(
            f"Conveyor background image not found: {BACKGROUND_PATH}"
        )
    backgrounds = [BACKGROUND_PATH]

    print(f"Background image: {BACKGROUND_PATH.name}")
    print(
        "Maximum source images per class: "
        f"{MAX_SOURCE_IMAGES_PER_CLASS}"
    )
    print()

    # Prepare crops for both classes before generating scenes.
    crop_cache = {}

    for class_name in CLASSES:
        print(f"Preparing {class_name} tomato crops...")
        crop_cache[class_name] = prepare_crops(class_name)
        print()

    for class_name in CLASSES:
        print(f"Generating scenes for {class_name}...")
        generate_scenes(
            class_name,
            crop_cache[class_name],
            backgrounds,
            output_root,
            scenes_per_class,
        )
        print()

    print("=" * 60)
    print("GENERATION FINISHED")
    print(f"Scenes saved under: {output_root}")
    print(f"Reusable crops saved under: {CACHE_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate class-specific synthetic conveyor scenes."
    )
    parser.add_argument(
        "--scenes-per-class",
        type=int,
        default=SCENES_PER_CLASS,
        help=f"Scenes to generate for each class (default: {SCENES_PER_CLASS}).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_ROOT,
        help=f"Output directory (default: {OUTPUT_ROOT}).",
    )
    args = parser.parse_args()
    main(args.scenes_per_class, args.output_dir)
