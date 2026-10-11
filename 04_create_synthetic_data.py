
# ============================================================
# FILE: 04_create_synthetic_data.py
#
# PURPOSE:
# Explain clearly what this file does and why we need it.
#
# INPUT:
# List the images, datasets, model weights, or other files
# that this script uses.
#
# OUTPUT:
# List the files, images, models, metrics, or results
# that this script creates.
#
# NEXT STEP:
# Explain which project step will use these outputs.
# ============================================================


from pathlib import Path
import random

from PIL import Image, ImageDraw, ImageFilter


# ------------------------------------------------------------
# CONFIGURATION
# ------------------------------------------------------------

# This script is being run from the project folder.
PROJECT_PATH = Path(__file__).resolve().parent

# Save generated backgrounds outside the original dataset.
OUTPUT_PATH = (
    PROJECT_PATH.parent
    / "datasets"
    / "synthetic_backgrounds"
)

# Number of background images to generate.
NUMBER_OF_BACKGROUNDS = 20

# Image dimensions.
IMAGE_WIDTH = 640
IMAGE_HEIGHT = 480

# Fixed seed makes the experiment reproducible.
RANDOM_SEED = 42


def create_belt_background(seed):
    """
    Create one artificial conveyor-belt background.

    The background includes:
    - A dark gray belt surface
    - Subtle surface texture
    - Horizontal belt markings
    - Slight lighting variation
    """

    random.seed(seed)

    # Start with a dark gray conveyor surface.
    image = Image.new(
        "RGB",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        (65, 65, 65),
    )

    draw = ImageDraw.Draw(image)

    # Add many small variations to imitate surface texture.
    for _ in range(12000):
        x = random.randrange(IMAGE_WIDTH)
        y = random.randrange(IMAGE_HEIGHT)

        brightness = random.randint(-12, 12)
        gray = max(0, min(255, 65 + brightness))

        draw.point((x, y), fill=(gray, gray, gray))

    # Draw subtle horizontal conveyor-belt lines.
    for y in range(30, IMAGE_HEIGHT, 60):
        line_color = random.choice([
            (52, 52, 52),
            (75, 75, 75),
            (83, 83, 83),
        ])

        draw.line(
            (0, y, IMAGE_WIDTH, y),
            fill=line_color,
            width=random.randint(1, 3),
        )

    # Apply a small blur so the texture is less artificial.
    image = image.filter(
        ImageFilter.GaussianBlur(radius=0.6)
    )

    # Add a translucent lighting gradient.
    # The overlay makes one side slightly brighter.
    overlay = Image.new(
        "RGBA",
        (IMAGE_WIDTH, IMAGE_HEIGHT),
        (0, 0, 0, 0),
    )

    overlay_draw = ImageDraw.Draw(overlay)

    for x in range(IMAGE_WIDTH):
        distance = abs(x - IMAGE_WIDTH / 2)
        alpha = int(
            35 * distance / (IMAGE_WIDTH / 2)
        )

        overlay_draw.line(
            (x, 0, x, IMAGE_HEIGHT),
            fill=(0, 0, 0, alpha),
        )

    image = Image.alpha_composite(
        image.convert("RGBA"),
        overlay,
    ).convert("RGB")

    return image


def main():
    # Create the output folder if it does not exist.
    OUTPUT_PATH.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("Generating synthetic conveyor backgrounds...")
    print(f"Output folder: {OUTPUT_PATH}")

    # Generate backgrounds using different random seeds.
    for index in range(NUMBER_OF_BACKGROUNDS):
        background = create_belt_background(
            RANDOM_SEED + index
        )

        output_file = (
            OUTPUT_PATH / f"belt_{index + 1:03d}.jpg"
        )

        background.save(
            output_file,
            quality=95,
        )

        print(f"Saved: {output_file.name}")

    print("\nBackground generation completed.")
    print(
        f"Generated {NUMBER_OF_BACKGROUNDS} "
        "background images."
    )


if __name__ == "__main__":
    main()
