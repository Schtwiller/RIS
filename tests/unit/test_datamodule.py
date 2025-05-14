"""
tests/unit/test_datamodule.py
-----------------------------
Smoke‑tests DataModule wiring using torchvision.FakeData so we don’t rely on
real images in CI.
"""
import pytest

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

from torchvision.datasets import FakeData
from torchvision import transforms as tvt

from src.datasets.datamodule import DataModule


class TinyDM(DataModule):
    """Override setup() to use a small FakeData split for quick tests."""
    def setup(self):
        tf = tvt.ToTensor()
        self.train_dataset = FakeData(size=20, image_size=(3, 224, 224), num_classes=4, transform=tf)
        self.val_dataset   = FakeData(size=10, image_size=(3, 224, 224), num_classes=4, transform=tf)
        self.test_dataset  = FakeData(size=10, image_size=(3, 224, 224), num_classes=4, transform=tf)

def test_dataloaders_shapes():
    dm = TinyDM(batch_size=4, num_workers=0)
    dm.setup()
    train_loader = dm.train_dataloader()
    images, labels = next(iter(train_loader))
    # basic shape checks
    assert images.shape == (4, 3, 224, 224)
    assert labels.shape == (4,)
    # number of classes inferred
    assert len(dm.train_dataset.classes) == 4 if hasattr(dm.train_dataset, "classes") else True
