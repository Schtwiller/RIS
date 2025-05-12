#!/usr/bin/env python3
"""
Defines on-the-fly preprocessing and augmentation pipelines
for train / val / test / inference.
"""

from albumentations import (
    Compose, Resize, Rotate, Perspective, CLAHE, RandomBrightnessContrast,
    HorizontalFlip, Normalize, ElasticTransform, GridDistortion,
)
from albumentations.augmentations.transforms import GaussNoise
from albumentations.pytorch import ToTensorV2
import numpy as np
import cv2
import random

# ─── Config ────────────────────────────────────────────────────────────────────
IMG_SIZE = 224
MEAN = (0.485, 0.456, 0.406)
STD  = (0.229, 0.224, 0.225)
# ────────────────────────────────────────────────────────────────────────────────

def _train_augs(rotation_limit=5, contrast_limit=0.2, elastic_p=0.2, noise_p=0.3):
    return Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Rotate(limit=rotation_limit, border_mode=0, p=0.5),
        Perspective(scale=(0.02, 0.05), p=0.3),
        CLAHE(p=0.3),
        RandomBrightnessContrast(brightness_limit=contrast_limit, contrast_limit=contrast_limit, p=0.3),
        HorizontalFlip(p=0.5),
        ElasticTransform(alpha=1.0, sigma=50, p=elastic_p),  # Elastic transform
        GaussNoise(std_range=[0.2, 0.44], p=noise_p),   # Gaussian noise
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ], additional_targets={'image': _dynamic_resize})

def _eval_augs():
    return Compose([
        Resize(IMG_SIZE, IMG_SIZE),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def _dynamic_resize(image):
    """Resize image to maintain aspect ratio, pad to IMG_SIZE."""
    h, w = image.shape[:2]
    scale = IMG_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padding = ((0, IMG_SIZE - new_h), (0, IMG_SIZE - new_w), (0, 0))  # Padding for aspect ratio
    padded_resized = np.pad(resized, padding, mode='constant', constant_values=255)  # White padding
    return padded_resized

def build_transforms(mode: str = "train"):
    if mode == "train":
        return _train_augs()
    else:
        return _eval_augs()
