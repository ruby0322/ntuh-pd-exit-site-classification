from __future__ import annotations

import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from PIL import Image, ImageOps

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp"}
DATASET_ROOT = Path("dataset")
OUTPUT_PATH = Path("sample.png")
SAMPLES_PER_CLASS = 5
RANDOM_SEED = 7
CELL_SIZE = 2.2 * 0.8


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
    rows = len(class_names)
    cols = max(len(paths) for paths in sampled.values())

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=(cols * CELL_SIZE, rows * CELL_SIZE),
        dpi=200,
        squeeze=False,
        gridspec_kw={"hspace": 0.03, "wspace": 0.03},
    )

    for row, class_name in enumerate(class_names):
        for col, image_path in enumerate(sampled[class_name]):
            ax = axes[row, col]
            with Image.open(image_path) as img:
                square = ImageOps.fit(
                    img.convert("RGB"),
                    (256, 256),
                    method=Image.Resampling.LANCZOS,
                    centering=(0.5, 0.5),
                )
                ax.imshow(square)
            ax.set_axis_off()
            if col == 0:
                ax.text(
                    -0.08,
                    0.5,
                    class_name,
                    transform=ax.transAxes,
                    ha="right",
                    va="center",
                    fontsize=11,
                    fontweight="bold",
                )

    fig.suptitle("NTUH PD exit-site dataset samples", fontsize=14, y=0.995)
    fig.subplots_adjust(left=0.14, right=0.995, top=0.96, bottom=0.01)
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    class_map = discover_class_images(DATASET_ROOT)
    sampled = sample_class_images(class_map)
    render_teaser(sampled, OUTPUT_PATH)


if __name__ == "__main__":
    main()
