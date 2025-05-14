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

from src.config import IMAGE_SIZE, AUGMENTATION

# ─── Config ────────────────────────────────────────────────────────────────────
MEAN = AUGMENTATION["mean"]
STD  = AUGMENTATION["std"]
ROTATE_LIMIT = AUGMENTATION["rotate_limit"]
CONTRAST_LIMIT = AUGMENTATION["contrast_limit"]
ELASTIC_P = AUGMENTATION["elastic_p"]
NOISE_P = AUGMENTATION["noise_p"]
# ────────────────────────────────────────────────────────────────────────────────

def _train_augs(rotation_limit=ROTATE_LIMIT, contrast_limit=CONTRAST_LIMIT, elastic_p=ELASTIC_P, noise_p=NOISE_P):
    return Compose([
        Resize(IMAGE_SIZE, IMAGE_SIZE),
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
        Resize(IMAGE_SIZE, IMAGE_SIZE),
        Normalize(mean=MEAN, std=STD),
        ToTensorV2(),
    ])

def _dynamic_resize(image):
    """Resize image to maintain aspect ratio, pad to IMAGE_SIZE."""
    h, w = image.shape[:2]
    scale = IMAGE_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))
    padding = ((0, IMAGE_SIZE - new_h), (0, IMAGE_SIZE - new_w), (0, 0))  # Padding for aspect ratio
    padded_resized = np.pad(resized, padding, mode='constant', constant_values=255)  # White padding
    return padded_resized

def build_transforms(mode: str = "train"):
    if mode == "train":
        return _train_augs()
    else:
        return _eval_augs()

def get_train_transforms():
    return build_transforms("train")

def get_val_transforms():
    return build_transforms("val")

def get_test_transforms():
    return build_transforms("test")
