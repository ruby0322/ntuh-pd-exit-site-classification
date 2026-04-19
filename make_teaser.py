from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
DATASET_ROOT = Path("dataset")
OUTPUT_PATH = Path("teaser.png")
SAMPLES_PER_CLASS = 5
RANDOM_SEED = 7


def discover_class_images(dataset_root: Path) -> dict[str, list[Path]]:
    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset directory: {dataset_root}")

    class_map: dict[str, list[Path]] = {}
    for class_dir in sorted(p for p in dataset_root.iterdir() if p.is_dir()):
        images = sorted(
            p
            for p in class_dir.iterdir()
            if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
        )
        if images:
            class_map[class_dir.name] = images

    if not class_map:
        raise ValueError(f"No class folders with images found under: {dataset_root}")

    return class_map


def sample_class_images(
    class_map: dict[str, list[Path]],
    samples_per_class: int = SAMPLES_PER_CLASS,
    seed: int = RANDOM_SEED,
) -> dict[str, list[Path]]:
    rng = random.Random(seed)
    sampled: dict[str, list[Path]] = {}

    for class_name in sorted(class_map):
        images = class_map[class_name]
        if len(images) < samples_per_class:
            raise ValueError(
                f"Class {class_name!r} has only {len(images)} images; "
                f"need at least {samples_per_class}"
            )
        sampled[class_name] = rng.sample(images, samples_per_class)

    return sampled


def render_teaser(sampled: dict[str, list[Path]], output_path: Path) -> None:
    class_names = list(sampled)
    rows = max(len(paths) for paths in sampled.values())
    cols = len(class_names)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * 2.2, rows * 2.2),
        dpi=200,
        squeeze=False,
    )

    for col, class_name in enumerate(class_names):
        for row, image_path in enumerate(sampled[class_name]):
            ax = axes[row, col]
            with Image.open(image_path) as img:
                ax.imshow(img.convert("RGB"))
            ax.set_axis_off()
            if row == 0:
                ax.set_title(class_name, fontsize=11, pad=10)

    fig.suptitle("NTUH PD exit-site dataset samples", fontsize=14, y=0.995)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    class_map = discover_class_images(DATASET_ROOT)
    sampled = sample_class_images(class_map)
    render_teaser(sampled, OUTPUT_PATH)


if __name__ == "__main__":
    main()
