"""
Dataset preparation and shared constants for NTUH PD exit-site classification.

Usage:
    python prepare.py                     # verify dataset, print stats
    python prepare.py --dataset ./dataset # override dataset root

Shared constants (imported by train.py):
    DATASET_ROOT, TIME_BUDGET, BINARY_INFECTION_CLASS
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from collections import Counter
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Constants (imported by train.py)
# ---------------------------------------------------------------------------

DATASET_ROOT = Path("./dataset")

# Wall-clock training budget in seconds (5 minutes). train.py enforces this.
TIME_BUDGET = 300

# Index of the infection-positive class in ImageFolder sorted order.
# ImageFolder sorts class subdirectories alphabetically:
#   class_0 → 0, class_1 → 1, ..., class_4 → 4
# class_4 is the infection-positive class; all others (0–3) are negative.
BINARY_INFECTION_CLASS = 4

# Canonical image size used for train and test transforms.
IMAGE_SIZE = 384

# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def find_images(class_dir: Path) -> list[Path]:
    return [p for p in class_dir.iterdir() if p.suffix.lower() in SUPPORTED_EXTENSIONS]


def verify_dataset(root: Path) -> dict[str, list[Path]]:
    """Check dataset structure and return {class_name: [image_paths]}."""
    if not root.exists():
        print(f"ERROR: dataset root not found: {root.resolve()}", file=sys.stderr)
        sys.exit(1)

    class_dirs = sorted(p for p in root.iterdir() if p.is_dir())
    if not class_dirs:
        print(f"ERROR: no class subdirectories found in {root.resolve()}", file=sys.stderr)
        sys.exit(1)

    data: dict[str, list[Path]] = {}
    for cd in class_dirs:
        images = find_images(cd)
        data[cd.name] = images

    return data


def check_readable(images: list[Path], sample: int = 5) -> list[str]:
    """Try to open a sample of images; return list of paths that failed."""
    from PIL import Image

    failed = []
    for p in images[:sample]:
        try:
            with Image.open(p) as img:
                img.verify()
        except Exception as e:
            failed.append(f"{p}: {e}")
    return failed


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def print_stats(data: dict[str, list[Path]], binary_class: int) -> None:
    total = sum(len(v) for v in data.values())
    class_names = list(data.keys())

    print(f"\nDataset: {len(class_names)} classes, {total} images total")
    print(f"{'Class':<12}  {'Count':>6}  {'%':>6}  {'Role'}")
    print("-" * 42)
    for idx, name in enumerate(class_names):
        n = len(data[name])
        pct = 100.0 * n / total if total else 0.0
        role = "infection-positive (class 4)" if idx == binary_class else "infection-negative"
        print(f"  {name:<10}  {n:>6}  {pct:>5.1f}%  {role}")

    # Binary split
    neg_total = sum(len(data[n]) for i, n in enumerate(class_names) if i != binary_class)
    pos_total = len(data[class_names[binary_class]]) if binary_class < len(class_names) else 0
    print()
    print(f"Binary split:")
    print(f"  Negative (classes 0–{binary_class - 1}):  {neg_total}  ({100.0*neg_total/total:.1f}%)")
    print(f"  Positive (class {binary_class}):          {pos_total}  ({100.0*pos_total/total:.1f}%)")

    # Class imbalance warning
    counts = [len(v) for v in data.values()]
    ratio = max(counts) / min(counts) if min(counts) > 0 else float("inf")
    if ratio > 3:
        print(f"\n  ⚠  Max/min class ratio = {ratio:.1f}x — class-weighted loss is recommended.")

    # Extension breakdown
    ext_counter: Counter = Counter()
    for images in data.values():
        for p in images:
            ext_counter[p.suffix.lower()] += 1
    print(f"\nImage formats: {dict(ext_counter)}")

    # Config constants
    print(f"\nActive constants (imported by train.py):")
    print(f"  TIME_BUDGET            = {TIME_BUDGET} s")
    print(f"  IMAGE_SIZE             = {IMAGE_SIZE}")
    print(f"  BINARY_INFECTION_CLASS = {BINARY_INFECTION_CLASS}")
    print()


# ---------------------------------------------------------------------------
# Evaluation (fixed ground truth — do not change the metric logic)
# ---------------------------------------------------------------------------

def evaluate(model: nn.Module, test_loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Evaluate model on test_loader; returns (multiclass_accuracy, binary_accuracy).

    Multiclass accuracy: prediction must exactly match the 5-class ground truth.

    Binary accuracy (infection screening):
      - label == BINARY_INFECTION_CLASS → correct only if pred == BINARY_INFECTION_CLASS
      - label != BINARY_INFECTION_CLASS → correct if pred is any non-infection class

    This is the fixed evaluation harness. Do not modify.
    """
    import torch

    model.eval()
    correct = 0
    bin_correct = 0
    total = 0
    C = BINARY_INFECTION_CLASS
    with torch.no_grad():
        for x, t in test_loader:
            x = x.to(device).float()
            t = t.to(device)
            pred = model(x).argmax(dim=1)
            correct += (pred == t).sum().item()
            bin_ok = ((t == C) & (pred == C)) | ((t != C) & (pred != C))
            bin_correct += bin_ok.sum().item()
            total += t.size(0)
    acc = correct / total if total else 0.0
    bin_acc = bin_correct / total if total else 0.0
    print("Accuracy:        %6.4f" % acc, flush=True)
    print("Binary accuracy: %6.4f" % bin_acc, flush=True)
    return acc, bin_acc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_ROOT,
        help="Path to dataset root (ImageFolder layout, default: ./dataset)",
    )
    parser.add_argument(
        "--check-images",
        type=int,
        default=5,
        metavar="N",
        help="Number of images per class to open and verify (0 = skip, default: 5)",
    )
    args = parser.parse_args()

    print(f"Dataset root: {args.dataset.resolve()}")
    data = verify_dataset(args.dataset)

    print_stats(data, BINARY_INFECTION_CLASS)

    if args.check_images > 0:
        print("Checking image readability (PIL open+verify)...")
        errors: list[str] = []
        for cls_name, images in data.items():
            errs = check_readable(images, sample=args.check_images)
            if errs:
                errors.extend(errs)
                print(f"  {cls_name}: {len(errs)} unreadable")
            else:
                print(f"  {cls_name}: OK ({min(args.check_images, len(images))} sampled)")
        if errors:
            print("\nUnreadable images:")
            for e in errors:
                print(f"  {e}")
            sys.exit(1)
        else:
            print("\nAll sampled images readable. Dataset looks good.")
    else:
        print("Image readability check skipped (--check-images 0).")
