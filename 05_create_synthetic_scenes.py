
"""
FILE: 05_create_synthetic_scenes.py

PURPOSE:
    Generate multi-tomato synthetic conveyor-belt scenes for the
    Fresh-versus-Defect classification project.

INPUT:
    datasets/tomato/train/Fresh
    datasets/tomato/train/Defect
    datasets/synthetic_backgrounds

OUTPUT:
    datasets/synthetic_multi_scenes_pilot/train/Fresh
    datasets/synthetic_multi_scenes_pilot/train/Defect
    datasets/tomato_crop_cache/Fresh
    datasets/tomato_crop_cache/Defect

IMPORTANT:
    - Uses only the existing training images.
    - Processes a limited sample to reduce runtime.
    - Caches tomato crops so later runs can reuse them.
    - Keeps each synthetic scene within one class.
    - Does not modify the original dataset or test set.

NEXT STEP:
    Inspect the generated crops and scenes before using them
    to train an experimental classifier.
"""

from pathlib import Path
import random
import cv2
import numpy as np

# ------------------------------------------------------------
# 1. PATHS
# ------------------------------------------------------------

PROJECT_PATH = Path(__file__).resolve().parent
DATASET_ROOT = PROJECT_PATH.parent / "datasets"

SOURCE_ROOT = DATASET_ROOT / "tomato" / "train"
BACKGROUND_ROOT = DATASET_ROOT / "synthetic_backgrounds"

OUTPUT_ROOT = (
    DATASET_ROOT / "synthetic_multi_scenes_pilot" / "train"
)
CACHE_ROOT = DATASET_ROOT / "tomato_crop_cache"

CLASSES = ["Fresh", "Defect"]

# ------------------------------------------------------------
# 2. SPEED AND IMAGE SETTINGS
# ------------------------------------------------------------

# Start small. Increase to 40 or 60 after checking the results.
MAX_SOURCE_IMAGES_PER_CLASS = 30

# Number of scenes generated for each class.
SCENES_PER_CLASS = 10

# Number of tomatoes placed in each scene.
MIN_TOMATOES = 5
MAX_TOMATOES = 12

SCENE_WIDTH = 640
SCENE_HEIGHT = 480

# Fewer GrabCut iterations reduce processing time.
GRABCUT_ITERATIONS = 2

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
# 4. EXTRACT A TOMATO CROP USING GRABCUT
# ------------------------------------------------------------

def extract_crop(image_path):
    """
    Attempt to extract the main foreground object.

    IMPORTANT:
        GrabCut is an approximate segmentation method.
        It may retain leaves or remove parts of a tomato.
        Review the resulting crops before relying on them.
    """
    image = cv2.imread(str(image_path))

    if image is None:
        return None

    height, width = image.shape[:2]

    # Very small images are unlikely to provide useful crops.
    if width < 40 or height < 40:
        return None

    # A rectangle inset from the edges gives GrabCut a rough
    # estimate of the foreground region.
    margin_x = max(2, int(width * 0.05))
    margin_y = max(2, int(height * 0.05))

    rect_width = width - 2 * margin_x
    rect_height = height - 2 * margin_y

    if rect_width <= 1 or rect_height <= 1:
        return None

    mask = np.zeros((height, width), dtype=np.uint8)
    background_model = np.zeros((1, 65), dtype=np.float64)
    foreground_model = np.zeros((1, 65), dtype=np.float64)

    try:
        cv2.grabCut(
            image,
            mask,
            (
                margin_x,
                margin_y,
                rect_width,
                rect_height,
            ),
            background_model,
            foreground_model,
            GRABCUT_ITERATIONS,
            cv2.GC_INIT_WITH_RECT,
        )
    except cv2.error:
        return None

    # Definite and probable foreground pixels become opaque.
    alpha = np.where(
        (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
        255,
        0,
    ).astype(np.uint8)

    # Reject masks that select almost nothing or almost the
    # entire image; these often indicate poor segmentation.
    foreground_fraction = np.count_nonzero(alpha) / alpha.size

    if foreground_fraction < 0.01 or foreground_fraction > 0.95:
        return None

    points = cv2.findNonZero(alpha)

    if points is None:
        return None

    x, y, crop_width, crop_height = cv2.boundingRect(points)

    if crop_width < 15 or crop_height < 15:
        return None

    # Crop the image and its transparency mask together.
    image_crop = image[
        y:y + crop_height,
        x:x + crop_width
    ]
    alpha_crop = alpha[
        y:y + crop_height,
        x:x + crop_width
    ]

    # Slightly soften the mask boundary.
    alpha_crop = cv2.GaussianBlur(
        alpha_crop, (3, 3), 0.8
    )

    # Store a four-channel PNG so the crop has transparency.
    rgba = cv2.cvtColor(image_crop, cv2.COLOR_BGR2BGRA)
    rgba[:, :, 3] = alpha_crop

    return rgba


# ------------------------------------------------------------
# 5. BUILD OR REUSE THE CROP CACHE
# ------------------------------------------------------------

def prepare_crops(class_name):
    """
    Reuse cached crops when available.

    Otherwise, process only a small sample of source images.
    Each successfully extracted crop is saved as a PNG.
    """
    cache_folder = CACHE_ROOT / class_name
    cache_folder.mkdir(parents=True, exist_ok=True)

    cached_crops = sorted(cache_folder.glob("*.png"))

    if cached_crops:
        print(
            f"  Reusing {len(cached_crops)} cached "
            f"{class_name} crops."
        )
        return cached_crops

    source_folder = SOURCE_ROOT / class_name
    source_images = find_images(source_folder)

    if not source_images:
        raise FileNotFoundError(
            f"No training images found in: {source_folder}"
        )

    random.shuffle(source_images)
    source_images = source_images[:MAX_SOURCE_IMAGES_PER_CLASS]

    print(
        f"  Processing {len(source_images)} source images "
        f"for {class_name}..."
    )

    saved = []

    for index, image_path in enumerate(source_images, start=1):
        crop = extract_crop(image_path)

        if crop is not None:
            output_path = (
                cache_folder / f"crop_{len(saved):04d}.png"
            )

            if cv2.imwrite(str(output_path), crop):
                saved.append(output_path)

        # Visible progress instead of a long silent wait.
        if index % 5 == 0 or index == len(source_images):
            print(
                f"    {class_name}: checked "
                f"{index}/{len(source_images)}, "
                f"saved {len(saved)} crops."
            )

    if not saved:
        raise RuntimeError(
            f"No usable crops were extracted for {class_name}. "
            "Inspect the source images and segmentation settings."
        )

    print(f"  Cached {len(saved)} crops for {class_name}.")
    return saved


# ------------------------------------------------------------
# 6. LOAD A BACKGROUND AND RESIZE IT
# ------------------------------------------------------------

def load_background(background_path):
    """Load and resize one conveyor-belt background."""
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

    # Keep the resized tomato within reasonable dimensions.
    target_height = max(20, min(target_height, SCENE_HEIGHT // 2))

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

def generate_scenes(class_name, crop_paths, backgrounds):
    """Create synthetic scenes containing only one class."""
    output_folder = OUTPUT_ROOT / class_name
    output_folder.mkdir(parents=True, exist_ok=True)

    for scene_index in range(SCENES_PER_CLASS):
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

            # Vary sizes to simulate different distances and sizes.
            target_width = random.randint(45, 105)

            x = random.randint(
                0, SCENE_WIDTH - target_width
            )
            y = random.randint(
                0, SCENE_HEIGHT - 40
            )

            if paste_crop(scene, crop_path, x, y, target_width):
                placed += 1

        output_path = (
            output_folder / f"synthetic_{scene_index + 1:03d}.jpg"
        )

        if cv2.imwrite(str(output_path), scene):
            print(
                f"  {class_name} scene "
                f"{scene_index + 1}/{SCENES_PER_CLASS} "
                f"saved ({placed} tomatoes): {output_path.name}"
            )


# ------------------------------------------------------------
# 9. MAIN
# ------------------------------------------------------------

def main():
    print("=" * 60)
    print("FAST MULTI-TOMATO SYNTHETIC SCENE GENERATION")
    print("=" * 60)
    print(f"Project folder: {PROJECT_PATH}")
    print(f"Source dataset: {SOURCE_ROOT}")
    print(f"Background folder: {BACKGROUND_ROOT}")
    print(f"Output folder: {OUTPUT_ROOT}")
    print(f"Crop cache: {CACHE_ROOT}")
    print()

    backgrounds = find_images(BACKGROUND_ROOT)

    if not backgrounds:
        raise FileNotFoundError(
            f"No background images found in {BACKGROUND_ROOT}"
        )

    print(f"Background images found: {len(backgrounds)}")
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
        )
        print()

    print("=" * 60)
    print("GENERATION FINISHED")
    print(f"Scenes saved under: {OUTPUT_ROOT}")
    print(f"Reusable crops saved under: {CACHE_ROOT}")
    print("=" * 60)


if __name__ == "__main__":
    main()
