# src/config.py

import os
from pathlib import Path

# General settings
PROJECT_NAME = "Reverse Image Search"
IMAGE_SIZE = 512  # Default size of images to resize

# Data settings
RAW_DATA_PATH = Path(__file__).parents[2] / "data" / "raw"
PROCESSED_DATA_PATH = Path(__file__).parents[2] / "data" / "processed"
LABELS_CSV = os.path.join(RAW_DATA_PATH, "labels.csv")

# Augmentation parameters
AUGMENTATION = {
    "rotate_limit": 5,
    "contrast_limit": 0.2,
    "elastic_p": 0.2,
    "noise_p": 0.3,
    "mean": (0.485, 0.456, 0.406),
    "std": (0.229, 0.224, 0.225),
}

# Model settings
MODEL = {
    "use_pretrained": True,
    "model_name": "resnet18",  # You can change this to other models (e.g., "efficientnet")
    "dropout_rate": 0.5,
}

# Training settings
TRAINING = {"epochs": 10, "patience": 3, "batch_size": 32}

# Optimizer settings
OPTIMIZER = {
    "name": "adam",  # Optimizer type (e.g., adam, sgd)
    "momentum": 0.9,  # Only for SGD
    "weight_decay": 1e-4,  # Regularization term
    "lr": 1e-3,
}

# Logging and checkpoints
LOGGING = {"log_dir": os.path.join(os.getcwd(), "logs"), "log_level": "INFO"}

CHECKPOINTS = {
    "save_dir": os.path.join(os.getcwd(), "checkpoints"),
    "save_best_only": True,
}

INDEXING = {
    "graduating_threshold": 10,
    "top_k_images": 5,
    "match_threshold": 1.2,
    "n_fallback": 3,
}

# Training/Validation Split
TRAIN_VAL_SPLIT = 0.7
