#!/usr/bin/env python3
"""
Defines on-the-fly preprocessing and augmentation pipelines
for train / val / test / inference.
"""

from albumentations import (
    Compose,
    Resize,
    Rotate,
    Perspective,
    CLAHE,
    RandomBrightnessContrast,
    HorizontalFlip,
    Normalize,
)
from albumentations.pytorch import ToTensorV2

# ─── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
# ────────────────────────────────────────────────────────────────────────────────

def _train_augs():
    return Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Rotate(limit=5, border_mode=0, p=0.5),
        Perspective(scale=(0.02, 0.05), p=0.3),
        CLAHE(p=0.3),
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.3),
        HorizontalFlip(p=0.5),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def _eval_augs():
    return Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def build_transforms(mode: str = "train"):
    """
    mode: "train", "val", "test", or "infer"
    Returns an Albumentations Compose object.
    """
    if mode == "train":
        return _train_augs()
    else:
        return _eval_augs()
