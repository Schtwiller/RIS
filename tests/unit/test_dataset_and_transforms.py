import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

import numpy as np
import pytest
import torch

from src.datasets import IDDataset
from src.datasets.transforms import build_transforms
from src.config import IMAGE_SIZE

# ← absolute path to reverse-image-search-model/data/processed
PROCESSED = Path(__file__).resolve().parents[2] / "data" / "processed"

@pytest.fixture(scope="session")
def train_ds():
    tfm = build_transforms("train")
    return IDDataset("train", root=PROCESSED, transform=tfm)

def test_len_nonzero(train_ds):
    assert len(train_ds) > 0

def test_sample_shape(train_ds):
    x, label = train_ds[0]
    # Albumentations returns a tensor already
    assert isinstance(x, torch.Tensor)
    assert x.shape == (3, IMAGE_SIZE, IMAGE_SIZE)
    # Label is a string for now
    assert isinstance(label, str)

def test_dataloader_batch(train_ds):
    from torch.utils.data import DataLoader

    loader = DataLoader(train_ds, batch_size=4, shuffle=True, num_workers=0)
    x, y = next(iter(loader))
    assert x.shape == (4, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert len(y) == 4
