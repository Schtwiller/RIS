# src/datasets/transforms.py
from torchvision import transforms
from albumentations import (
    Compose, Resize, RandomBrightnessContrast, Rotate, Perspective,
    CLAHE, HorizontalFlip
)
from albumentations.pytorch import ToTensorV2
import random

IMG_SIZE = 224  # keep in sync with model

def _albumentations_train():
    return Compose(
        [
            Resize(IMG_SIZE, IMG_SIZE),
            Rotate(limit=5, border_mode=0, p=0.5),
            Perspective(scale=(0.02, 0.05), p=0.3),
            CLAHE(p=0.3),
            RandomBrightnessContrast(0.1, 0.1, p=0.3),
            HorizontalFlip(p=0.5),
            ToTensorV2(),
        ]
    )

def _albumentations_eval():
    return Compose(
        [
            Resize(IMG_SIZE, IMG_SIZE),
            ToTensorV2(),
        ]
    )

def build_transforms(split: str = "train"):
    if split == "train":
        return _albumentations_train()
    else:                 # "val", "test", "infer"
        return _albumentations_eval()
