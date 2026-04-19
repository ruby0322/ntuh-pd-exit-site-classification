"""
NTUH PD exit-site image classification training (from ntuh_pd_exit_site.ipynb).

Usage (after `python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt`):

    python3 train.py
    python3 train.py --dataset ./dataset/ --epochs 100 --max-train-seconds 300
    python3 train.py --max-train-seconds 0 # no wall-clock limit (only --epochs)
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as trns
from torchvision import datasets
from torch.utils.data import DataLoader

from prepare import TIME_BUDGET, BINARY_INFECTION_CLASS, IMAGE_SIZE, DATASET_ROOT, evaluate


# --- Hyperparameters ---

BATCH_SIZE = 4
INPUT_CHANNEL = 3
NUM_EPOCH = 100
MAX_TRAIN_SECONDS_DEFAULT = float(TIME_BUDGET)

# ResNet-style presets for exit-site classification (small data → prefer regularisation over raw width).
ARCH_CONFIGS: dict[str, dict] = {
    "baseline": {
        "stem_ch": 32,
        "stages": (64, 128, 256, 256),
        "blocks": (1, 1, 1, 1),
        "dropout": 0.4,
    },
    "wide": {
        "stem_ch": 40,
        "stages": (80, 160, 320, 320),
        "blocks": (1, 1, 1, 1),
        "dropout": 0.45,
    },
    # Wider than `wide` (~1.4× params); same block pattern — pair with same SGD / aug / early-stop as e11.
    "wide_xl": {
        "stem_ch": 48,
        "stages": (96, 192, 384, 384),
        "blocks": (1, 1, 1, 1),
        "dropout": 0.5,
    },
    "deep": {
        "stem_ch": 32,
        "stages": (64, 128, 256, 256),
        "blocks": (1, 2, 2, 2),
        "dropout": 0.35,
    },
    "lite": {
        "stem_ch": 24,
        "stages": (48, 96, 160, 160),
        "blocks": (1, 1, 1, 1),
        "dropout": 0.35,
    },
}

TRANSFER_BACKBONES = ("none", "resnet50", "efficientnet_b3", "mobilenet_v3_large", "convnext_tiny")
CLASSIFIER_HEAD_BACKBONES = {"efficientnet_b3", "mobilenet_v3_large", "convnext_tiny"}


class myBatchNorm(nn.Module):
    def __init__(self, input_channel, eps=1e-4, momentum=0.1):
        super().__init__()
        self.eps = eps
        self.momentum = momentum
        shape = (1, input_channel, 1, 1)
        self.gamma = nn.Parameter(torch.ones(shape))
        self.beta = nn.Parameter(torch.zeros(shape))
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)

    def forward(self, x):
        if self.moving_mean.device != x.device:
            self.moving_mean = self.moving_mean.to(x.device)
            self.moving_var = self.moving_var.to(x.device)
        y, self.moving_mean, self.moving_var = self.batch_norm(
            x,
            self.gamma,
            self.beta,
            self.moving_mean,
            self.moving_var,
            self.eps,
            self.momentum,
        )
        return y

    def batch_norm(self, x, gamma, beta, moving_mean, moving_var, eps, momentum):
        if not torch.is_grad_enabled():
            x_hat = (x - moving_mean) / torch.sqrt(moving_var + eps)
        else:
            batch_mean = torch.mean(x, dim=(0, 2, 3), keepdim=True)
            batch_var = torch.var(x, dim=(0, 2, 3), keepdim=True, unbiased=False)
            x_hat = (x - batch_mean) / torch.sqrt(batch_var + eps)
            moving_mean.data = momentum * moving_mean.data + (1.0 - momentum) * batch_mean.data
            moving_var.data = momentum * moving_var.data + (1.0 - momentum) * batch_var.data
        y = gamma * x_hat + beta
        return y, moving_mean, moving_var


class myConvolution(nn.Module):
    def __init__(self, input_channel, output_channel, kernel_size=1, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(input_channel, output_channel, kernel_size, stride, padding)

    def forward(self, x):
        return self.conv(x)


class myActivation(nn.Module):
    def __init__(self):
        super().__init__()
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x)


class myMaxPooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.pool = nn.MaxPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pool(x)


class myAvgPooling(nn.Module):
    def __init__(self, kernel_size=2, stride=2, padding=0):
        super().__init__()
        self.pool = nn.AvgPool2d(kernel_size, stride, padding)

    def forward(self, x):
        return self.pool(x)


class myResBlock(nn.Module):
    def __init__(self, input_channel, med_channel, stride=1, padding=1):
        super().__init__()
        self.conv1 = myConvolution(input_channel, med_channel, kernel_size=3, stride=stride, padding=padding)
        self.bn1 = myBatchNorm(med_channel)
        self.relu = myActivation()
        self.conv2 = myConvolution(med_channel, med_channel, kernel_size=3, stride=1, padding=padding)
        self.bn2 = myBatchNorm(med_channel)
        self.shortcut = nn.Sequential()
        if stride != 1 or input_channel != med_channel:
            self.shortcut = nn.Sequential(
                myConvolution(input_channel, med_channel, kernel_size=1, stride=stride),
                myBatchNorm(med_channel),
            )

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += identity
        out = self.relu(out)
        return out


def _make_stage(in_ch: int, out_ch: int, n_blocks: int) -> nn.Module:
    layers: list[nn.Module] = [myResBlock(in_ch, out_ch)]
    for _ in range(n_blocks - 1):
        layers.append(myResBlock(out_ch, out_ch))
    return nn.Sequential(*layers)


class myCNN(nn.Module):
    def __init__(self, input_channel: int = 3, num_classes: int = 5, arch: str = "baseline"):
        super().__init__()
        if arch not in ARCH_CONFIGS:
            raise ValueError(f"Unknown arch {arch!r}; choose from {list(ARCH_CONFIGS)}")
        cfg = ARCH_CONFIGS[arch]
        stem_ch = cfg["stem_ch"]
        stages: tuple[int, int, int, int] = cfg["stages"]
        blocks: tuple[int, int, int, int] = cfg["blocks"]
        dropout_p = cfg["dropout"]

        self.stem = nn.Sequential(
            myConvolution(input_channel, stem_ch, kernel_size=7, stride=2, padding=3),
            myBatchNorm(stem_ch),
            myActivation(),
            myMaxPooling(kernel_size=3, stride=2, padding=1),
        )
        s1, s2, s3, s4 = stages
        b1, b2, b3, b4 = blocks
        self.stage1 = _make_stage(stem_ch, s1, b1)
        self.pool1 = myMaxPooling(2, 2)
        self.stage2 = _make_stage(s1, s2, b2)
        self.pool2 = myMaxPooling(2, 2)
        self.stage3 = _make_stage(s2, s3, b3)
        self.pool3 = myMaxPooling(2, 2)
        self.stage4 = _make_stage(s3, s4, b4)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(p=dropout_p)
        self.classifier = nn.Linear(s4, num_classes)
        self._arch_name = arch

    def forward(self, x):
        x = self.stem(x)
        x = self.pool1(self.stage1(x))
        x = self.pool2(self.stage2(x))
        x = self.pool3(self.stage3(x))
        x = self.stage4(x)
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x


class EvalTTAWrapper(nn.Module):
    """Wrap a model and average logits over horizontal-flip TTA at eval time."""

    def __init__(self, model: nn.Module, *, hflip: bool = False):
        super().__init__()
        self.model = model
        self.hflip = hflip

    def forward(self, x):
        logits = self.model(x)
        if self.training or not self.hflip:
            return logits
        flipped_logits = self.model(torch.flip(x, dims=[3]))
        return 0.5 * (logits + flipped_logits)


class BinaryScreeningLoss(nn.Module):
    """Collapse 5-class logits into infection-vs-noninfection grouped logits."""

    def __init__(self, infection_class: int, binary_class_weights: torch.Tensor | None = None):
        super().__init__()
        self.infection_class = infection_class
        if binary_class_weights is not None:
            self.register_buffer("binary_class_weights", binary_class_weights.clone().detach())
        else:
            self.binary_class_weights = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        neg_logits = torch.cat(
            [logits[:, : self.infection_class], logits[:, self.infection_class + 1 :]],
            dim=1,
        )
        grouped_logits = torch.stack(
            [
                torch.logsumexp(neg_logits, dim=1),
                logits[:, self.infection_class],
            ],
            dim=1,
        )
        binary_targets = (targets == self.infection_class).long()
        return F.cross_entropy(grouped_logits, binary_targets, weight=self.binary_class_weights)


class MulticlassWithBinaryAuxLoss(nn.Module):
    """Primary 5-class CE with a small auxiliary binary screening term."""

    def __init__(
        self,
        multiclass_loss: nn.Module,
        binary_loss: nn.Module,
        aux_weight: float,
    ):
        super().__init__()
        self.multiclass_loss = multiclass_loss
        self.binary_loss = binary_loss
        self.aux_weight = aux_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mc = self.multiclass_loss(logits, targets)
        bin_loss = self.binary_loss(logits, targets)
        return mc + self.aux_weight * bin_loss


def get_model_size(model: nn.Module) -> tuple[int, int, float]:
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    buffer_size = 0
    for buffer in model.buffers():
        buffer_size += buffer.nelement() * buffer.element_size()
    size_all_mb = (param_size + buffer_size) / 1024**2
    return param_size, buffer_size, size_all_mb


def build_transfer_model(backbone: str, num_classes: int, dropout_p: float) -> nn.Module:
    """Return a torchvision pretrained backbone with a fresh classification head."""
    from torchvision import models

    if backbone == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        in_features = base.fc.in_features
        base.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "efficientnet_b3":
        base = models.efficientnet_b3(weights=models.EfficientNet_B3_Weights.DEFAULT)
        in_features = base.classifier[1].in_features
        base.classifier = nn.Sequential(
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    elif backbone == "mobilenet_v3_large":
        base = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
        in_features = base.classifier[-1].in_features
        base.classifier[-2] = nn.Dropout(p=dropout_p, inplace=True)
        base.classifier[-1] = nn.Linear(in_features, num_classes)
    elif backbone == "convnext_tiny":
        base = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.DEFAULT)
        in_features = base.classifier[-1].in_features
        base.classifier = nn.Sequential(
            base.classifier[0],
            base.classifier[1],
            nn.Dropout(p=dropout_p, inplace=True),
            nn.Linear(in_features, num_classes),
        )
    else:
        raise ValueError(
            "Unknown backbone %r; choose from 'resnet50', 'efficientnet_b3', "
            "'mobilenet_v3_large', 'convnext_tiny'" % (backbone,)
        )
    return base


# ImageNet statistics for pretrained backbones.
_IMAGENET_MEAN = [0.485, 0.456, 0.406]
_IMAGENET_STD  = [0.229, 0.224, 0.225]


def _build_transforms(
    *,
    imagenet_norm: bool = False,
    train_vertical_flip: bool = True,
    rotation_degrees: float = 15.0,
    jitter_brightness: float = 0.2,
    jitter_contrast: float = 0.2,
    jitter_saturation: float = 0.1,
) -> tuple[nn.Module, nn.Module]:
    """Build train/eval transforms used throughout the research loop."""
    _norm = [trns.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)] if imagenet_norm else []
    train_tfms: list[nn.Module] = [
        trns.Resize(IMAGE_SIZE),
        trns.CenterCrop(IMAGE_SIZE),
        trns.RandomHorizontalFlip(),
    ]
    if train_vertical_flip:
        train_tfms.append(trns.RandomVerticalFlip())
    train_tfms.extend(
        [
            trns.RandomRotation(rotation_degrees),
            trns.ColorJitter(
                brightness=jitter_brightness,
                contrast=jitter_contrast,
                saturation=jitter_saturation,
            ),
            trns.ToTensor(),
            *_norm,
        ]
    )
    train_transform = trns.Compose(train_tfms)
    eval_transform = trns.Compose(
        [
            trns.Resize(IMAGE_SIZE),
            trns.CenterCrop(IMAGE_SIZE),
            trns.ToTensor(),
            *_norm,
        ]
    )
    return train_transform, eval_transform


def _build_imagefolder_datasets(
    dataset_root: Path,
    *,
    imagenet_norm: bool = False,
    train_vertical_flip: bool = True,
    rotation_degrees: float = 15.0,
    jitter_brightness: float = 0.2,
    jitter_contrast: float = 0.2,
    jitter_saturation: float = 0.1,
) -> tuple[datasets.ImageFolder, datasets.ImageFolder]:
    """Build paired train/eval ImageFolder datasets with identical class ordering."""
    train_transform, eval_transform = _build_transforms(
        imagenet_norm=imagenet_norm,
        train_vertical_flip=train_vertical_flip,
        rotation_degrees=rotation_degrees,
        jitter_brightness=jitter_brightness,
        jitter_contrast=jitter_contrast,
        jitter_saturation=jitter_saturation,
    )
    full_train_ds = datasets.ImageFolder(root=str(dataset_root), transform=train_transform)
    full_eval_ds = datasets.ImageFolder(root=str(dataset_root), transform=eval_transform)
    return full_train_ds, full_eval_ds


def _split_train_test_indices(num_items: int, train_fraction: float, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Deterministically split dataset indices into train/test subsets."""
    train_size = int(train_fraction * num_items)
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_items, generator=generator)
    return indices[:train_size], indices[train_size:]


def build_dataloaders(
    dataset_root: Path,
    batch_size: int,
    train_fraction: float,
    seed: int,
    *,
    imagenet_norm: bool = False,
    train_vertical_flip: bool = True,
    rotation_degrees: float = 15.0,
    jitter_brightness: float = 0.2,
    jitter_contrast: float = 0.2,
    jitter_saturation: float = 0.1,
) -> tuple[DataLoader, DataLoader, int, torch.Tensor, torch.Tensor]:
    # Augmented transform for training only — critical for generalisation on ~2k images.
    # Flips and colour jitter are safe for wound photography; rotation ±15° mimics
    # real capture-angle variation.
    # Load the dataset twice with different transforms; use identical split indices
    # so train subset uses augmentation while test subset does not.
    full_train_ds, full_test_ds = _build_imagefolder_datasets(
        dataset_root,
        imagenet_norm=imagenet_norm,
        train_vertical_flip=train_vertical_flip,
        rotation_degrees=rotation_degrees,
        jitter_brightness=jitter_brightness,
        jitter_contrast=jitter_contrast,
        jitter_saturation=jitter_saturation,
    )
    num_classes = len(full_train_ds.classes)
    print(f"Detected {num_classes} classes: {full_train_ds.classes}")

    n = len(full_train_ds)
    train_idx, test_idx = _split_train_test_indices(n, train_fraction, seed)

    data_train = torch.utils.data.Subset(full_train_ds, train_idx)
    data_test  = torch.utils.data.Subset(full_test_ds,  test_idx)

    # Per-class weights to counteract heavy class imbalance (class_0 ≈ 57%)
    targets = torch.tensor(full_train_ds.targets)
    counts  = torch.bincount(targets, minlength=num_classes).float()
    class_weights = (n / (num_classes * counts))
    binary_counts = torch.tensor(
        [
            counts.sum() - counts[BINARY_INFECTION_CLASS],
            counts[BINARY_INFECTION_CLASS],
        ],
        dtype=torch.float32,
    )
    binary_class_weights = n / (2 * binary_counts)

    train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=True)
    test_loader  = DataLoader(data_test,  batch_size=batch_size, shuffle=False)
    return train_loader, test_loader, num_classes, class_weights, binary_class_weights


def _binary_accuracy_from_probs(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    threshold: float,
    infection_class: int = BINARY_INFECTION_CLASS,
) -> float:
    """Binary screening accuracy from infection probabilities and multiclass labels."""
    if labels.numel() == 0:
        return 0.0
    binary_labels = labels == infection_class
    binary_preds = probs >= threshold
    return (binary_preds == binary_labels).float().mean().item()


def _legacy_binary_accuracy_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    infection_class: int = BINARY_INFECTION_CLASS,
) -> float:
    """Binary screening accuracy induced by the legacy multiclass argmax rule."""
    if labels.numel() == 0:
        return 0.0
    preds = logits.argmax(dim=1)
    binary_ok = ((labels == infection_class) & (preds == infection_class)) | (
        (labels != infection_class) & (preds != infection_class)
    )
    return binary_ok.float().mean().item()


def _find_best_binary_threshold(
    probs: torch.Tensor,
    labels: torch.Tensor,
    *,
    num_steps: int = 1001,
    infection_class: int = BINARY_INFECTION_CLASS,
) -> tuple[float, float]:
    """Grid-search a threshold that maximizes binary screening accuracy."""
    best_threshold = 0.5
    best_acc = -1.0
    candidates = torch.linspace(0.0, 1.0, steps=num_steps)
    for candidate in candidates.tolist():
        acc = _binary_accuracy_from_probs(
            probs,
            labels,
            threshold=candidate,
            infection_class=infection_class,
        )
        if (
            acc > best_acc
            or (acc == best_acc and abs(candidate - 0.5) < abs(best_threshold - 0.5))
        ):
            best_threshold = float(candidate)
            best_acc = float(acc)
    return best_threshold, best_acc


def _split_indices_for_oof(num_items: int, num_folds: int, seed: int) -> list[torch.Tensor]:
    """Split `[0, num_items)` into deterministic OOF folds."""
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(num_items, generator=generator)
    return [chunk.clone() for chunk in torch.tensor_split(indices, num_folds)]


def _collect_logits_and_labels(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Collect logits and labels from a loader without changing caller state."""
    was_training = model.training
    model.eval()
    logits_parts: list[torch.Tensor] = []
    label_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for x, y in loader:
            logits_parts.append(model(x.to(device).float()).cpu())
            label_parts.append(y.cpu())
    if was_training:
        model.train()
    if not logits_parts:
        return torch.empty((0, 5)), torch.empty((0,), dtype=torch.long)
    return torch.cat(logits_parts, dim=0), torch.cat(label_parts, dim=0)


def _infection_probs_from_logits(
    logits: torch.Tensor,
    *,
    infection_class: int = BINARY_INFECTION_CLASS,
) -> torch.Tensor:
    """Extract infection-class probabilities from multiclass logits."""
    if logits.numel() == 0:
        return torch.empty((0,), dtype=torch.float32)
    return torch.softmax(logits, dim=1)[:, infection_class]


def _collect_binary_calibration_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    *,
    infection_class: int = BINARY_INFECTION_CLASS,
) -> dict[str, float]:
    """Summarize argmax and threshold-optimized binary screening stats."""
    logits, labels = _collect_logits_and_labels(model, loader, device)
    probs = _infection_probs_from_logits(logits, infection_class=infection_class)
    best_threshold, best_bin_acc = _find_best_binary_threshold(
        probs,
        labels,
        infection_class=infection_class,
    )
    argmax_bin_acc = _legacy_binary_accuracy_from_logits(
        logits,
        labels,
        infection_class=infection_class,
    )
    return {
        "argmax_bin_acc": argmax_bin_acc,
        "best_threshold": best_threshold,
        "best_bin_acc": best_bin_acc,
    }


def _val_metrics(model: nn.Module, loader: DataLoader, device: torch.device) -> tuple[float, float]:
    """Return (multiclass_acc, binary_acc) on *loader* (no grad)."""
    model.eval()
    correct = bin_correct = total = 0
    infection_class = BINARY_INFECTION_CLASS
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device).float(), y.to(device)
            preds = model(x).argmax(dim=1)
            correct += (preds == y).sum().item()
            bin_ok = ((y == infection_class) & (preds == infection_class)) | (
                (y != infection_class) & (preds != infection_class)
            )
            bin_correct += bin_ok.sum().item()
            total += y.size(0)
    model.train()
    if total == 0:
        return 0.0, 0.0
    return correct / total, bin_correct / total


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    device: torch.device,
    num_epoch: int,
    *,
    max_train_seconds: float | None,
    log_every: int,
    patience: int,
    checkpoint_path: Path,
    monitor_metric: str,
) -> tuple[float, bool, int]:
    """Train until ``num_epoch`` epochs or ``max_train_seconds`` elapses.

    Applies early stopping using ``monitor_metric`` while always retaining
    the checkpoint with the best validation binary screening accuracy.
    Restores that best-bin_acc checkpoint before returning.

    Returns ``(wall_seconds, stopped_by_time_budget, best_epoch)``.
    """
    model.train()
    t_start = time.perf_counter()
    deadline = (t_start + max_train_seconds) if max_train_seconds and max_train_seconds > 0 else None
    stopped_by_budget = False

    best_metric: float = -1.0
    best_epoch: int = 0
    best_checkpoint_metric: float = -1.0
    best_checkpoint_epoch: int = 0
    epochs_no_improve: int = 0

    for epoch in range(num_epoch):
        if deadline and time.perf_counter() >= deadline:
            stopped_by_budget = True
            print(
                "\nStopped: reached max train time (%.1f s) before epoch %d"
                % (max_train_seconds, epoch),
                flush=True,
            )
            break

        print("Epoch %d / %d — starting" % (epoch, num_epoch - 1), flush=True)
        losses: list[float] = []
        for batch_num, input_data in enumerate(train_loader):
            if deadline and time.perf_counter() >= deadline:
                stopped_by_budget = True
                print(
                    "\nStopped: max train time (%.1f s) after epoch %d batch %d"
                    % (max_train_seconds, epoch, batch_num),
                    flush=True,
                )
                break
            optimizer.zero_grad()
            x, y = input_data
            x = x.to(device).float()
            y = y.to(device)
            output = model(x)
            loss = criterion(output, y)
            loss.backward()
            losses.append(loss.item())
            optimizer.step()
            if log_every > 0 and batch_num % log_every == 0:
                print(
                    "\tEpoch %d | Batch %d | Loss %6.4f" % (epoch, batch_num, loss.item()),
                    flush=True,
                )

        avg_loss = sum(losses) / len(losses) if losses else float("nan")
        val_acc, val_bin_acc = _val_metrics(model, val_loader, device)
        monitored_value = val_bin_acc if monitor_metric == "bin_acc" else val_acc

        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(monitored_value)
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            "Epoch %d | Loss %6.4f | Val-Acc %6.4f | Val-Bin %6.4f | LR %.2e"
            % (epoch, avg_loss, val_acc, val_bin_acc, current_lr),
            flush=True,
        )

        if val_bin_acc > best_checkpoint_metric:
            best_checkpoint_metric = val_bin_acc
            best_checkpoint_epoch = epoch
            torch.save(model.state_dict(), checkpoint_path)
            print(
                "  ↑ New best checkpoint bin_acc %.4f — checkpoint saved to %s"
                % (val_bin_acc, checkpoint_path),
                flush=True,
            )
        if monitored_value > best_metric:
            best_metric = monitored_value
            best_epoch = epoch
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if patience > 0 and epochs_no_improve >= patience:
                print(
                    "Early stopping: no %s improvement for %d epochs (best %.4f at epoch %d)"
                    % (monitor_metric, patience, best_metric, best_epoch),
                    flush=True,
                )
                break

        if stopped_by_budget:
            break

    if checkpoint_path.exists():
        model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
        print(
            "Restored best checkpoint from epoch %d (bin_acc %.4f)"
            % (best_checkpoint_epoch, best_checkpoint_metric),
            flush=True,
        )

    return time.perf_counter() - t_start, stopped_by_budget, best_checkpoint_epoch



def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=Path, default=DATASET_ROOT, help="ImageFolder root")
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--epochs", type=int, default=NUM_EPOCH, help="Max full epochs (may stop earlier due to --max-train-seconds)")
    p.add_argument(
        "--max-train-seconds",
        type=float,
        default=MAX_TRAIN_SECONDS_DEFAULT,
        help="Wall-clock training cap in seconds (default 300). Use 0 to disable.",
    )
    p.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Print every N batches (0 = only epoch start/end)",
    )
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0001)
    p.add_argument(
        "--transfer-dropout",
        type=float,
        default=0.4,
        help="Dropout probability used in pretrained backbone classification head.",
    )
    p.add_argument(
        "--loss-mode",
        type=str,
        default="multiclass",
        choices=("multiclass", "binary_grouped", "multiclass_binary_aux"),
        help=(
            "Training loss: standard 5-class cross entropy, grouped infection-vs-noninfection "
            "loss, or CE plus a small grouped-binary auxiliary loss."
        ),
    )
    p.add_argument(
        "--binary-aux-weight",
        type=float,
        default=0.25,
        help="Weight for grouped binary auxiliary loss when --loss-mode multiclass_binary_aux.",
    )
    p.add_argument(
        "--pos-weight-boost",
        type=float,
        default=1.0,
        help=(
            "Extra multiplier applied to the infection-positive class weight. "
            "Useful when transfer learning improves mc_acc but misses the 9% positive class."
        ),
    )
    p.add_argument(
        "--disable-vertical-flip",
        action="store_true",
        help="Disable RandomVerticalFlip in training augmentation.",
    )
    p.add_argument(
        "--rotation-degrees",
        type=float,
        default=15.0,
        help="RandomRotation magnitude for training augmentation.",
    )
    p.add_argument(
        "--jitter-brightness",
        type=float,
        default=0.2,
        help="ColorJitter brightness strength for training augmentation.",
    )
    p.add_argument(
        "--jitter-contrast",
        type=float,
        default=0.2,
        help="ColorJitter contrast strength for training augmentation.",
    )
    p.add_argument(
        "--jitter-saturation",
        type=float,
        default=0.1,
        help="ColorJitter saturation strength for training augmentation.",
    )
    p.add_argument("--train-fraction", type=float, default=0.8)
    p.add_argument("--seed", type=int, default=42, help="RNG seed for split, torch, cuda")
    p.add_argument(
        "--arch",
        type=str,
        default="baseline",
        choices=tuple(ARCH_CONFIGS.keys()),
        help="ResNet preset (see ARCH_CONFIGS in train.py)",
    )
    p.add_argument("--optimizer", type=str, default="sgd", choices=("sgd", "adamw"))
    p.add_argument(
        "--nesterov",
        action="store_true",
        help="Use Nesterov momentum for SGD (ignored for AdamW)",
    )
    p.add_argument("--model-out", type=Path, default=Path("model.pt"))
    p.add_argument(
        "--backbone",
        type=str,
        default="none",
        choices=TRANSFER_BACKBONES,
        help=(
            "Pretrained torchvision backbone (default: none = use custom myCNN). "
            "When set, --arch is ignored and ImageNet normalisation is applied automatically."
        ),
    )
    p.add_argument(
        "--freeze-epochs",
        type=int,
        default=0,
        help=(
            "Two-phase fine-tuning: freeze backbone for this many epochs (linear probe), "
            "then unfreeze the full model. 0 = full fine-tune from the start."
        ),
    )
    p.add_argument(
        "--backbone-lr",
        type=float,
        default=0.0,
        help=(
            "Differential LR for pretrained backbone layers (head uses --lr). "
            "0 (default) = same LR for all params. Ignored when --backbone none or --freeze-epochs > 0."
        ),
    )
    p.add_argument(
        "--eval-hflip-tta",
        action="store_true",
        help="At evaluation time, average logits over original and horizontally flipped views.",
    )
    p.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early-stopping patience in epochs (0 = disabled)",
    )
    p.add_argument(
        "--ckpt-dir",
        type=Path,
        default=None,
        help="Directory for best-checkpoint file (default: same dir as --model-out)",
    )
    p.add_argument(
        "--scheduler",
        type=str,
        default="plateau",
        choices=("none", "plateau", "cosine"),
        help=(
            "LR scheduler: 'plateau' = ReduceLROnPlateau (factor 0.5, patience 5), "
            "'cosine' = CosineAnnealingLR (T_max=epochs), 'none' = constant LR"
        ),
    )
    p.add_argument(
        "--enable-oof-calibration",
        action="store_true",
        help="Fit an infection threshold from out-of-fold predictions on the training split.",
    )
    p.add_argument(
        "--oof-folds",
        type=int,
        default=3,
        help="Number of folds used when --enable-oof-calibration is enabled.",
    )
    return p.parse_args(argv)


def _freeze_backbone(model: nn.Module, backbone: str) -> None:
    """Freeze all backbone parameters; leave only the head trainable."""
    model = _unwrap_model(model)
    if backbone == "resnet50":
        for name, param in model.named_parameters():
            param.requires_grad = "fc" in name
    elif backbone in CLASSIFIER_HEAD_BACKBONES:
        for name, param in model.named_parameters():
            param.requires_grad = "classifier" in name


def _unfreeze_all(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.model if isinstance(model, EvalTTAWrapper) else model


def _build_model(args: argparse.Namespace, num_classes: int, device: torch.device) -> nn.Module:
    """Construct the requested model on the target device."""
    if args.backbone != "none":
        model = build_transfer_model(args.backbone, num_classes, args.transfer_dropout).to(device)
    else:
        model = myCNN(INPUT_CHANNEL, num_classes, arch=args.arch).to(device)
    if args.eval_hflip_tta:
        model = EvalTTAWrapper(model, hflip=True).to(device)
    return model


def _build_loss(
    args: argparse.Namespace,
    class_weights: torch.Tensor,
    binary_class_weights: torch.Tensor,
    device: torch.device,
    *,
    announce: bool = True,
) -> tuple[nn.Module, str]:
    """Construct the training criterion and its monitored validation metric."""
    multiclass_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    binary_criterion = BinaryScreeningLoss(BINARY_INFECTION_CLASS, binary_class_weights.to(device))

    if args.loss_mode == "binary_grouped":
        if announce:
            print("Training objective: grouped binary infection-vs-noninfection loss", flush=True)
        return binary_criterion, "bin_acc"
    if args.loss_mode == "multiclass_binary_aux":
        if announce:
            print(
                "Training objective: multiclass CE + %.3f * grouped binary auxiliary loss"
                % args.binary_aux_weight,
                flush=True,
            )
        return (
            MulticlassWithBinaryAuxLoss(
                multiclass_criterion,
                binary_criterion,
                args.binary_aux_weight,
            ),
            "mc_acc",
        )
    return multiclass_criterion, "mc_acc"


def _head_params_for_model(model: nn.Module, backbone: str) -> tuple[list, list]:
    """Return (backbone_params, head_params) for differential LR."""
    base_model = _unwrap_model(model)
    if backbone == "resnet50":
        head = set(base_model.fc.parameters())
    elif backbone in CLASSIFIER_HEAD_BACKBONES:
        head = set(base_model.classifier.parameters())
    else:
        return list(model.parameters()), []
    backbone_p = [p for p in model.parameters() if p not in head]
    head_p = list(head)
    return backbone_p, head_p


def _make_optimizer_from_args(args: argparse.Namespace, params) -> torch.optim.Optimizer:
    """Build a standard optimizer from CLI args."""
    if args.optimizer == "adamw":
        return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        params,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )


def _make_diff_optimizer_from_args(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    """Build an optimizer with backbone_lr / head_lr parameter groups."""
    bb_params, hd_params = _head_params_for_model(model, args.backbone)
    param_groups = [
        {"params": bb_params, "lr": args.backbone_lr},
        {"params": hd_params, "lr": args.lr},
    ]
    if args.optimizer == "adamw":
        return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    return torch.optim.SGD(
        param_groups,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        nesterov=args.nesterov,
    )


def _make_scheduler_from_args(
    args: argparse.Namespace,
    optimizer: torch.optim.Optimizer,
    n_epochs: int,
) -> torch.optim.lr_scheduler.LRScheduler | None:
    """Build the requested LR scheduler."""
    if args.scheduler == "plateau":
        return torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=5)
    if args.scheduler == "cosine":
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(n_epochs, 1))
    return None


def _run_training_pipeline(
    args: argparse.Namespace,
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    criterion: nn.Module,
    monitor_metric: str,
    device: torch.device,
    checkpoint_path: Path,
    *,
    announce: bool = True,
    max_train_seconds_override: float | None = None,
) -> tuple[float, bool, int, int]:
    """Train a model with the same optimizer/scheduler policy as the main loop."""
    freeze_epochs = args.freeze_epochs if args.backbone != "none" else 0
    if freeze_epochs > 0:
        _freeze_backbone(model, args.backbone)
        if announce:
            print(f"Phase 1: backbone frozen for {freeze_epochs} epoch(s) (linear probe)", flush=True)

    if args.backbone != "none" and args.backbone_lr > 0 and freeze_epochs == 0:
        optimizer = _make_diff_optimizer_from_args(args, model)
        if announce:
            print(
                f"Differential LR: backbone={args.backbone_lr:.2e}, head={args.lr:.2e}",
                flush=True,
            )
    else:
        optimizer = _make_optimizer_from_args(args, filter(lambda p: p.requires_grad, model.parameters()))

    max_train = args.max_train_seconds if args.max_train_seconds > 0 else None
    if max_train_seconds_override is not None:
        max_train = max_train_seconds_override

    if freeze_epochs > 0:
        sched1 = _make_scheduler_from_args(args, optimizer, freeze_epochs)
        sec1, budget_hit1, _ = train(
            model,
            train_loader,
            val_loader,
            criterion,
            optimizer,
            sched1,
            device,
            freeze_epochs,
            max_train_seconds=max_train,
            log_every=args.log_every,
            patience=0,
            checkpoint_path=checkpoint_path,
            monitor_metric=monitor_metric,
        )
        remaining = (max_train - sec1) if max_train else None
        if not budget_hit1:
            if announce:
                print("Phase 2: unfreezing backbone for full fine-tune", flush=True)
            _unfreeze_all(model)
            if args.backbone != "none" and args.backbone_lr > 0:
                optimizer2 = _make_diff_optimizer_from_args(args, model)
                if announce:
                    print(
                        f"Phase 2 differential LR: backbone={args.backbone_lr:.2e}, head={args.lr:.2e}",
                        flush=True,
                    )
            else:
                optimizer2 = _make_optimizer_from_args(args, model.parameters())
            sched2 = _make_scheduler_from_args(args, optimizer2, args.epochs - freeze_epochs)
            sec2, budget_hit2, best_epoch = train(
                model,
                train_loader,
                val_loader,
                criterion,
                optimizer2,
                sched2,
                device,
                args.epochs - freeze_epochs,
                max_train_seconds=remaining,
                log_every=args.log_every,
                patience=args.patience,
                checkpoint_path=checkpoint_path,
                monitor_metric=monitor_metric,
            )
            return sec1 + sec2, budget_hit1 or budget_hit2, best_epoch, freeze_epochs
        return sec1, budget_hit1, 0, freeze_epochs

    sched = _make_scheduler_from_args(args, optimizer, args.epochs)
    train_seconds, stopped_by_budget, best_epoch = train(
        model,
        train_loader,
        val_loader,
        criterion,
        optimizer,
        sched,
        device,
        args.epochs,
        max_train_seconds=max_train,
        log_every=args.log_every,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
        monitor_metric=monitor_metric,
    )
    return train_seconds, stopped_by_budget, best_epoch, freeze_epochs


def _fit_oof_binary_threshold(
    args: argparse.Namespace,
    device: torch.device,
    num_classes: int,
    class_weights: torch.Tensor,
    binary_class_weights: torch.Tensor,
) -> float:
    """Fit a binary threshold from OOF predictions on the training split."""
    use_pretrained = args.backbone != "none"
    full_train_ds, full_eval_ds = _build_imagefolder_datasets(
        args.dataset,
        imagenet_norm=use_pretrained,
        train_vertical_flip=not args.disable_vertical_flip,
        rotation_degrees=args.rotation_degrees,
        jitter_brightness=args.jitter_brightness,
        jitter_contrast=args.jitter_contrast,
        jitter_saturation=args.jitter_saturation,
    )
    train_idx, _ = _split_train_test_indices(len(full_train_ds), args.train_fraction, args.seed)
    folds = _split_indices_for_oof(train_idx.numel(), args.oof_folds, args.seed)
    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else args.model_out.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    per_fold_budget = None
    if args.max_train_seconds > 0:
        per_fold_budget = max(args.max_train_seconds / max(args.oof_folds, 1), 1.0)

    oof_probs: list[torch.Tensor] = []
    oof_labels: list[torch.Tensor] = []
    for fold_idx, fold_positions in enumerate(folds, start=1):
        if fold_positions.numel() == 0:
            continue
        keep_mask = torch.ones(train_idx.numel(), dtype=torch.bool)
        keep_mask[fold_positions] = False
        fold_train_idx = train_idx[keep_mask]
        fold_val_idx = train_idx[fold_positions]
        fold_train_loader = DataLoader(
            torch.utils.data.Subset(full_train_ds, fold_train_idx),
            batch_size=args.batch_size,
            shuffle=True,
        )
        fold_val_loader = DataLoader(
            torch.utils.data.Subset(full_eval_ds, fold_val_idx),
            batch_size=args.batch_size,
            shuffle=False,
        )
        torch.manual_seed(args.seed + fold_idx)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed + fold_idx)
        fold_model = _build_model(args, num_classes, device)
        fold_criterion, fold_monitor = _build_loss(
            args,
            class_weights,
            binary_class_weights,
            device,
            announce=False,
        )
        fold_ckpt = ckpt_dir / f"{args.model_out.stem}_oof_fold{fold_idx}.pt"
        print(
            f"OOF fold {fold_idx}/{len(folds)} — train={len(fold_train_idx)} val={len(fold_val_idx)}",
            flush=True,
        )
        _run_training_pipeline(
            args,
            fold_model,
            fold_train_loader,
            fold_val_loader,
            fold_criterion,
            fold_monitor,
            device,
            fold_ckpt,
            announce=False,
            max_train_seconds_override=per_fold_budget,
        )
        fold_logits, fold_labels = _collect_logits_and_labels(fold_model, fold_val_loader, device)
        oof_probs.append(_infection_probs_from_logits(fold_logits))
        oof_labels.append(fold_labels)

    best_threshold, best_oof_acc = _find_best_binary_threshold(
        torch.cat(oof_probs, dim=0),
        torch.cat(oof_labels, dim=0),
    )
    print(
        "OOF calibration complete | best threshold %.4f | OOF bin_acc %.4f"
        % (best_threshold, best_oof_acc),
        flush=True,
    )
    return best_threshold


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    use_pretrained = args.backbone != "none"

    train_loader, test_loader, num_classes, class_weights, binary_class_weights = build_dataloaders(
        args.dataset,
        args.batch_size,
        args.train_fraction,
        args.seed,
        imagenet_norm=use_pretrained,
        train_vertical_flip=not args.disable_vertical_flip,
        rotation_degrees=args.rotation_degrees,
        jitter_brightness=args.jitter_brightness,
        jitter_contrast=args.jitter_contrast,
        jitter_saturation=args.jitter_saturation,
    )
    if args.pos_weight_boost != 1.0:
        class_weights[BINARY_INFECTION_CLASS] *= args.pos_weight_boost
        binary_class_weights[1] *= args.pos_weight_boost
        print(
            "Adjusted class-%d weight by x%.3f for binary screening emphasis"
            % (BINARY_INFECTION_CLASS, args.pos_weight_boost),
            flush=True,
        )

    model = _build_model(args, num_classes, device)
    param_size, buffer_size, model_size_mb = get_model_size(model)
    print(f"Model parameters size: {param_size / 1024**2:.4f} MB", flush=True)
    print(f"Model buffers size: {buffer_size / 1024**2:.4f} MB", flush=True)
    print(f"Model total size: {model_size_mb:.4f} MB", flush=True)
    if args.max_train_seconds > 0:
        print(
            "Training budget: %.1f s wall-clock (stops early if epochs finish first)"
            % args.max_train_seconds,
            flush=True,
        )
    else:
        print("Training budget: no wall-clock limit (--max-train-seconds 0)", flush=True)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else args.model_out.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / (args.model_out.stem + "_best.pt")
    criterion, monitor_metric = _build_loss(args, class_weights, binary_class_weights, device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    train_seconds, stopped_by_budget, best_epoch, freeze_epochs = _run_training_pipeline(
        args,
        model,
        train_loader,
        test_loader,
        criterion,
        monitor_metric,
        device,
        checkpoint_path,
    )

    torch.save(model, args.model_out)
    print(f"Saved model to {args.model_out}", flush=True)

    mc_acc, bin_acc = evaluate(model, test_loader, device)
    test_logits, test_labels = _collect_logits_and_labels(model, test_loader, device)
    bin_acc_argmax = _legacy_binary_accuracy_from_logits(test_logits, test_labels)
    bin_acc_calibrated = bin_acc_argmax
    best_oof_threshold = 0.5
    if args.enable_oof_calibration:
        best_oof_threshold = _fit_oof_binary_threshold(
            args,
            device,
            num_classes,
            class_weights,
            binary_class_weights,
        )
        test_probs = _infection_probs_from_logits(test_logits)
        bin_acc_calibrated = _binary_accuracy_from_probs(
            test_probs,
            test_labels,
            threshold=best_oof_threshold,
        )

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    print("---", flush=True)
    print(f"mc_acc:               {mc_acc:.6f}", flush=True)
    print(f"bin_acc:              {bin_acc:.6f}", flush=True)
    print(f"bin_acc_argmax:       {bin_acc_argmax:.6f}", flush=True)
    print(f"bin_acc_calibrated:   {bin_acc_calibrated:.6f}", flush=True)
    print(f"best_oof_threshold:   {best_oof_threshold:.6f}", flush=True)
    print(f"train_seconds:        {train_seconds:.1f}", flush=True)
    print(f"train_stopped_budget: {str(stopped_by_budget).lower()}", flush=True)
    print(f"peak_vram_mb:         {peak_vram_mb:.1f}", flush=True)
    print(f"arch:                 {args.arch}", flush=True)
    print(f"optimizer:            {args.optimizer}", flush=True)
    print(f"best_epoch:           {best_epoch}", flush=True)
    print(f"scheduler:            {args.scheduler}", flush=True)
    print(f"patience:             {args.patience}", flush=True)
    print(f"backbone:             {args.backbone}", flush=True)
    print(f"freeze_epochs:        {freeze_epochs}", flush=True)
    print(f"backbone_lr:          {args.backbone_lr}", flush=True)
    print(f"pos_weight_boost:     {args.pos_weight_boost}", flush=True)
    print(f"transfer_dropout:     {args.transfer_dropout}", flush=True)
    print(f"loss_mode:            {args.loss_mode}", flush=True)
    print(f"binary_aux_weight:    {args.binary_aux_weight}", flush=True)
    print(f"train_vflip:          {str(not args.disable_vertical_flip).lower()}", flush=True)
    print(f"rotation_degrees:     {args.rotation_degrees}", flush=True)
    print(f"jitter_brightness:    {args.jitter_brightness}", flush=True)
    print(f"jitter_contrast:      {args.jitter_contrast}", flush=True)
    print(f"jitter_saturation:    {args.jitter_saturation}", flush=True)
    print(f"eval_hflip_tta:       {str(args.eval_hflip_tta).lower()}", flush=True)
    print(f"enable_oof_calibration: {str(args.enable_oof_calibration).lower()}", flush=True)
    print(f"oof_folds:            {args.oof_folds}", flush=True)
    return

    if use_pretrained:
        model = build_transfer_model(args.backbone, num_classes, args.transfer_dropout).to(device)
    else:
        model = myCNN(INPUT_CHANNEL, num_classes, arch=args.arch).to(device)
    if args.eval_hflip_tta:
        model = EvalTTAWrapper(model, hflip=True).to(device)
    param_size, buffer_size, model_size_mb = get_model_size(model)
    print(f"Model parameters size: {param_size / 1024**2:.4f} MB", flush=True)
    print(f"Model buffers size: {buffer_size / 1024**2:.4f} MB", flush=True)
    print(f"Model total size: {model_size_mb:.4f} MB", flush=True)
    if args.max_train_seconds > 0:
        print(
            "Training budget: %.1f s wall-clock (stops early if epochs finish first)"
            % args.max_train_seconds,
            flush=True,
        )
    else:
        print("Training budget: no wall-clock limit (--max-train-seconds 0)", flush=True)

    ckpt_dir = args.ckpt_dir if args.ckpt_dir is not None else args.model_out.parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / (args.model_out.stem + "_best.pt")

    # Two-phase setup: during freeze_epochs only the head trains (lower VRAM, faster convergence).
    # After that the backbone is unfrozen and a new optimizer is swapped in at full LR.
    freeze_epochs = args.freeze_epochs if use_pretrained else 0
    if freeze_epochs > 0:
        _freeze_backbone(model, args.backbone)
        print(f"Phase 1: backbone frozen for {freeze_epochs} epoch(s) (linear probe)", flush=True)

    def _head_params(backbone: str) -> tuple[list, list]:
        """Return (backbone_params, head_params) for differential LR."""
        base_model = _unwrap_model(model)
        if backbone == "resnet50":
            head = set(base_model.fc.parameters())
        elif backbone in {"efficientnet_b3", "mobilenet_v3_large"}:
            head = set(base_model.classifier.parameters())
        else:
            return list(model.parameters()), []
        backbone_p = [p for p in model.parameters() if p not in head]
        head_p = list(head)
        return backbone_p, head_p

    def _make_optimizer(params):
        if args.optimizer == "adamw":
            return torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
        return torch.optim.SGD(
            params,
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

    def _make_diff_optimizer():
        """SGD/AdamW with backbone_lr and head_lr = args.lr."""
        bb_params, hd_params = _head_params(args.backbone)
        param_groups = [
            {"params": bb_params, "lr": args.backbone_lr},
            {"params": hd_params, "lr": args.lr},
        ]
        if args.optimizer == "adamw":
            return torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
        return torch.optim.SGD(
            param_groups,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
            nesterov=args.nesterov,
        )

    multiclass_criterion = nn.CrossEntropyLoss(weight=class_weights.to(device))
    binary_criterion = BinaryScreeningLoss(BINARY_INFECTION_CLASS, binary_class_weights.to(device))

    if args.loss_mode == "binary_grouped":
        criterion = binary_criterion
        monitor_metric = "bin_acc"
        print("Training objective: grouped binary infection-vs-noninfection loss", flush=True)
    elif args.loss_mode == "multiclass_binary_aux":
        criterion = MulticlassWithBinaryAuxLoss(
            multiclass_criterion,
            binary_criterion,
            args.binary_aux_weight,
        )
        monitor_metric = "mc_acc"
        print(
            "Training objective: multiclass CE + %.3f * grouped binary auxiliary loss"
            % args.binary_aux_weight,
            flush=True,
        )
    else:
        criterion = multiclass_criterion
        monitor_metric = "mc_acc"
    if use_pretrained and args.backbone_lr > 0 and freeze_epochs == 0:
        optimizer = _make_diff_optimizer()
        print(
            f"Differential LR: backbone={args.backbone_lr:.2e}, head={args.lr:.2e}", flush=True
        )
    else:
        optimizer = _make_optimizer(filter(lambda p: p.requires_grad, model.parameters()))

    def _make_scheduler(opt, n_epochs):
        if args.scheduler == "plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
        if args.scheduler == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(n_epochs, 1))
        return None

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    max_train = args.max_train_seconds if args.max_train_seconds > 0 else None

    if freeze_epochs > 0:
        # Phase 1: linear probe (backbone frozen)
        sched1 = _make_scheduler(optimizer, freeze_epochs)
        sec1, budget_hit1, _ = train(
            model, train_loader, test_loader, criterion, optimizer, sched1, device, freeze_epochs,
            max_train_seconds=max_train, log_every=args.log_every,
            patience=0, checkpoint_path=checkpoint_path, monitor_metric=monitor_metric,
        )
        remaining = (max_train - sec1) if max_train else None
        if not budget_hit1:
            # Phase 2: unfreeze backbone, new optimizer at full LR
            print("Phase 2: unfreezing backbone for full fine-tune", flush=True)
            _unfreeze_all(model)
            optimizer2 = _make_optimizer(model.parameters())
            sched2 = _make_scheduler(optimizer2, args.epochs - freeze_epochs)
            sec2, budget_hit2, best_epoch = train(
                model, train_loader, test_loader, criterion, optimizer2, sched2, device,
                args.epochs - freeze_epochs,
                max_train_seconds=remaining, log_every=args.log_every,
                patience=args.patience, checkpoint_path=checkpoint_path, monitor_metric=monitor_metric,
            )
            train_seconds = sec1 + sec2
            stopped_by_budget = budget_hit1 or budget_hit2
        else:
            train_seconds = sec1
            stopped_by_budget = budget_hit1
            best_epoch = 0
    else:
        sched = _make_scheduler(optimizer, args.epochs)
        train_seconds, stopped_by_budget, best_epoch = train(
            model, train_loader, test_loader, criterion, optimizer, sched, device, args.epochs,
            max_train_seconds=max_train, log_every=args.log_every,
            patience=args.patience, checkpoint_path=checkpoint_path, monitor_metric=monitor_metric,
        )

    torch.save(model, args.model_out)
    print(f"Saved model to {args.model_out}", flush=True)

    mc_acc, bin_acc = evaluate(model, test_loader, device)

    peak_vram_mb = 0.0
    if device.type == "cuda":
        peak_vram_mb = torch.cuda.max_memory_allocated(device) / (1024**2)

    # Machine-readable footer (research loop: grep mc_acc / bin_acc / peak_vram_mb)
    print("---", flush=True)
    print(f"mc_acc:               {mc_acc:.6f}", flush=True)
    print(f"bin_acc:              {bin_acc:.6f}", flush=True)
    print(f"train_seconds:        {train_seconds:.1f}", flush=True)
    print(f"train_stopped_budget: {str(stopped_by_budget).lower()}", flush=True)
    print(f"peak_vram_mb:         {peak_vram_mb:.1f}", flush=True)
    print(f"arch:                 {args.arch}", flush=True)
    print(f"optimizer:            {args.optimizer}", flush=True)
    print(f"best_epoch:           {best_epoch}", flush=True)
    print(f"scheduler:            {args.scheduler}", flush=True)
    print(f"patience:             {args.patience}", flush=True)
    print(f"backbone:             {args.backbone}", flush=True)
    print(f"freeze_epochs:        {freeze_epochs}", flush=True)
    print(f"backbone_lr:          {args.backbone_lr}", flush=True)
    print(f"pos_weight_boost:     {args.pos_weight_boost}", flush=True)
    print(f"transfer_dropout:     {args.transfer_dropout}", flush=True)
    print(f"loss_mode:            {args.loss_mode}", flush=True)
    print(f"binary_aux_weight:    {args.binary_aux_weight}", flush=True)
    print(f"train_vflip:          {str(not args.disable_vertical_flip).lower()}", flush=True)
    print(f"rotation_degrees:     {args.rotation_degrees}", flush=True)
    print(f"jitter_brightness:    {args.jitter_brightness}", flush=True)
    print(f"jitter_contrast:      {args.jitter_contrast}", flush=True)
    print(f"jitter_saturation:    {args.jitter_saturation}", flush=True)
    print(f"eval_hflip_tta:       {str(args.eval_hflip_tta).lower()}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
