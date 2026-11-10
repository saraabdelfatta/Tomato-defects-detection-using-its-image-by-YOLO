"""Example synthetic scenes for slide 6: M1 pilot (top row) vs M2 v2 (bottom row)."""
import random
import matplotlib.pyplot as plt
from PIL import Image
from tomato_utils import DATASETS, OUT, IMG_EXT

V2 = DATASETS / "synthetic_scenes_v2"
PILOT = DATASETS / "old synthetic" / "synthetic_multi_scenes_pilot" / "train"
K, SEED = 4, 7


def pick(d, k):
    imgs = sorted(p for p in d.rglob("*") if p.suffix.lower() in IMG_EXT)
    return random.Random(SEED).sample(imgs, min(k, len(imgs))) if imgs else []


rows = [("M2 v2: lighting, shadows,\nocclusion, blur", pick(V2, K))]
if PILOT.exists() and pick(PILOT, 1):
    rows.insert(0, ("M1 pilot", pick(PILOT, K)))

fig, axes = plt.subplots(len(rows), K, figsize=(3 * K, 3 * len(rows)), squeeze=False)
for r, (label, imgs) in enumerate(rows):
    for c in range(K):
        ax = axes[r][c]; ax.axis("off")
        if c < len(imgs):
            ax.imshow(Image.open(imgs[c]).convert("RGB"))
            ax.set_title(f"{imgs[c].parent.name} scene", fontsize=9)
    axes[r][0].text(-0.05, 0.5, label, transform=axes[r][0].transAxes, rotation=90,
                    va="center", ha="right", fontsize=10, fontweight="bold")
OUT.mkdir(parents=True, exist_ok=True)
fig.tight_layout(); fig.savefig(OUT / "synthetic_examples.png", dpi=200)
print("saved", OUT / "synthetic_examples.png")
