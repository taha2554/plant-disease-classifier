"""Dataset loading.

Expects an ImageFolder layout – one sub-folder per class:

    data/plantvillage/
        Tomato___Early_blight/  img001.jpg ...
        Tomato___healthy/       ...

The PlantVillage dataset (54k leaf images, 38 classes) can be downloaded from Kaggle or the
original authors; see the README. `split_dataset` makes a reproducible stratified
train/val/test split without copying any files.
"""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def train_transform(size: int = 224) -> transforms.Compose:
    # Leaves photographed on a plain background: flips and rotations are safe,
    # colour jitter is kept small because colour is a real symptom (yellowing, spots).
    return transforms.Compose(
        [
            transforms.RandomResizedCrop(size, scale=(0.7, 1.0)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def eval_transform(size: int = 224) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize(int(size * 1.14)),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def split_indices(targets: list[int], val: float, test: float, seed: int) -> tuple[list[int], list[int], list[int]]:
    """Stratified split: each class is divided with the same proportions."""
    by_class: dict[int, list[int]] = defaultdict(list)
    for idx, label in enumerate(targets):
        by_class[label].append(idx)

    rng = random.Random(seed)
    train_idx, val_idx, test_idx = [], [], []
    for indices in by_class.values():
        rng.shuffle(indices)
        n_test = max(1, round(len(indices) * test)) if test > 0 else 0
        n_val = max(1, round(len(indices) * val)) if val > 0 else 0
        test_idx += indices[:n_test]
        val_idx += indices[n_test : n_test + n_val]
        train_idx += indices[n_test + n_val :]
    return sorted(train_idx), sorted(val_idx), sorted(test_idx)


@dataclass
class Loaders:
    train: DataLoader
    val: DataLoader
    test: DataLoader
    classes: list[str]
    class_weights: torch.Tensor


def make_loaders(
    root: Path | str,
    image_size: int = 224,
    batch_size: int = 32,
    val: float = 0.1,
    test: float = 0.1,
    seed: int = 42,
    workers: int = 2,
) -> Loaders:
    # Two views over the same folder: one with augmentation (train), one without (val/test).
    train_view = datasets.ImageFolder(root, transform=train_transform(image_size))
    eval_view = datasets.ImageFolder(root, transform=eval_transform(image_size))
    train_idx, val_idx, test_idx = split_indices(train_view.targets, val, test, seed)

    # Inverse-frequency weights so rare diseases still matter in the loss.
    counts = Counter(train_view.targets[i] for i in train_idx)
    n_classes = len(train_view.classes)
    weights = torch.tensor([len(train_idx) / (n_classes * counts.get(c, 1)) for c in range(n_classes)], dtype=torch.float)

    def loader(view, idx, shuffle):
        return DataLoader(Subset(view, idx), batch_size=batch_size, shuffle=shuffle, num_workers=workers)

    return Loaders(
        train=loader(train_view, train_idx, True),
        val=loader(eval_view, val_idx, False),
        test=loader(eval_view, test_idx, False),
        classes=train_view.classes,
        class_weights=weights,
    )


def pretty_label(name: str) -> str:
    """'Tomato___Early_blight' -> 'Tomato – Early blight'"""
    plant, _, disease = name.partition("___")
    plant = plant.replace("_", " ").replace(",", "").strip()
    disease = disease.replace("_", " ").strip()
    return f"{plant} – {disease}" if disease else plant
