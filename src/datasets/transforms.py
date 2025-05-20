#!/usr/bin/env python3
"""
On‑the‑fly preprocessing & augmentation pipelines
(train / val / test / inference) that work as regular
torchvision transforms thanks to AlbumentationsWrapper.
"""

from pathlib import Path
import numpy as np
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2
import torchvision.transforms.functional as F  # only used as a fallback

# --------------------------------------------------------------------- #
# Config (pulled from src/config)                                       #
# --------------------------------------------------------------------- #
from src.config import IMAGE_SIZE, AUGMENTATION

MEAN           = AUGMENTATION["mean"]
STD            = AUGMENTATION["std"]
ROTATE_LIMIT   = AUGMENTATION["rotate_limit"]
CONTRAST_LIMIT = AUGMENTATION["contrast_limit"]
ELASTIC_P      = AUGMENTATION["elastic_p"]
NOISE_P        = AUGMENTATION["noise_p"]

# --------------------------------------------------------------------- #
# Albumentations <‑‑> torchvision adapter                               #
# --------------------------------------------------------------------- #
class AlbumentationsWrapper:
    """
    Wrap an Albumentations.Compose so it behaves like a torchvision
    transform (i.e. accepts a single PIL.Image or np.ndarray and
    returns a torch.Tensor).
    """
    def __init__(self, aug: A.Compose):
        self.aug = aug

    def __call__(self, img):
        # torchvision supplies PIL.Image; Albumentations wants np.ndarray
        if not isinstance(img, np.ndarray):
            img = np.array(img)

        augmented = self.aug(image=img)["image"]

        # If the pipeline ends with ToTensorV2, we already have a tensor.
        # Otherwise, convert explicitly.
        if isinstance(augmented, np.ndarray):
            augmented = F.to_tensor(augmented)

        return augmented

# --------------------------------------------------------------------- #
# Helper – dynamic resize that preserves aspect ratio                   #
# --------------------------------------------------------------------- #
def _dynamic_resize(image, **kwargs):
    """Resize image to fit inside IMAGE_SIZE, pad with white to square."""
    h, w = image.shape[:2]
    scale = IMAGE_SIZE / max(h, w)
    new_h, new_w = int(h * scale), int(w * scale)
    resized = cv2.resize(image, (new_w, new_h))

    pad_h, pad_w = IMAGE_SIZE - new_h, IMAGE_SIZE - new_w
    padded = cv2.copyMakeBorder(
        resized,
        top=0,
        bottom=pad_h,
        left=0,
        right=pad_w,
        borderType=cv2.BORDER_CONSTANT,
        value=[255, 255, 255],  # white padding
    )
    return padded

# --------------------------------------------------------------------- #
# Augmentation pipelines                                                #
# --------------------------------------------------------------------- #
def _train_augs():
    return A.Compose(
        [
            A.Lambda(image=_dynamic_resize),
            A.Rotate(limit=ROTATE_LIMIT, border_mode=cv2.BORDER_CONSTANT, p=0.5),
            A.Perspective(scale=(0.02, 0.05), p=0.3),
            A.CLAHE(p=0.3),
            A.RandomBrightnessContrast(
                brightness_limit=CONTRAST_LIMIT,
                contrast_limit=CONTRAST_LIMIT,
                p=0.3,
            ),
            A.HorizontalFlip(p=0.5),
            A.ElasticTransform(alpha=1.0, sigma=50, p=ELASTIC_P),
            A.GaussNoise(std_range=(0.04, 0.2), p=NOISE_P),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )


def _eval_augs():
    return A.Compose(
        [
            A.Lambda(image=_dynamic_resize),
            A.Normalize(mean=MEAN, std=STD),
            ToTensorV2(),
        ]
    )

# --------------------------------------------------------------------- #
# Public API                                                            #
# --------------------------------------------------------------------- #
def build_transforms(mode: str = "train"):
    if mode == "train":
        return AlbumentationsWrapper(_train_augs())
    else:  # "val", "test", or "infer"
        return AlbumentationsWrapper(_eval_augs())


def get_train_transforms():
    return build_transforms("train")


def get_val_transforms():
    return build_transforms("val")


def get_test_transforms():
    return build_transforms("test")
