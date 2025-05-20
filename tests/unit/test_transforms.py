import pytest
import torch
from pathlib import Path
from src.datasets import IDDataset
from src.datasets.transforms import build_transforms
import numpy as np
from src.config import IMAGE_SIZE

# ← absolute path to reverse-image-search-model/data/processed
PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"

@pytest.fixture(scope="session")
def train_ds():
    # Use the train transform pipeline
    transform = build_transforms("train")
    return IDDataset("train", PROCESSED_DIR, transform=transform)

@pytest.fixture(scope="session")
def val_ds():
    # Use the validation transform pipeline
    transform = build_transforms("val")
    return IDDataset("val", PROCESSED_DIR, transform=transform)

def test_train_dataset_length(train_ds):
    # Test that train dataset has samples
    assert len(train_ds) > 0

def test_val_dataset_length(val_ds):
    # Test that val dataset has samples
    assert len(val_ds) > 0

def test_train_transform(train_ds):
    # Get the first sample after applying transformations
    img, lbl = train_ds[0]
    # Check if the image is a tensor of correct shape
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

def test_val_transform(val_ds):
    # Get the first sample from validation dataset
    img, lbl = val_ds[0]
    # Check if the image is a tensor of correct shape
    assert isinstance(img, torch.Tensor)
    assert img.shape == (3, IMAGE_SIZE, IMAGE_SIZE)

def test_augmentation(train_ds):
    # Ensure random augmentations happen for each access
    img1, lbl1 = train_ds[0]
    img2, lbl2 = train_ds[1]
    # Assert that the images are not the same after random transformation
    assert not torch.equal(img1, img2)

