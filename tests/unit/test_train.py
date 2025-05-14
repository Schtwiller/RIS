"""
tests/unit/test_train.py
------------------------
Runs a 1‑epoch training pass on FakeData to ensure the full training loop
completes without crashing (CPU‑only, <5 seconds).
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

import torch
from torchvision.datasets import FakeData
from torchvision import transforms as tvt
from torch.utils.data import DataLoader

from src.models.resnet50  import create_resnet50_model
from src.training.trainer import train_model, evaluate_model


def _fake_dataloaders(batch_size=8):
    tf = tvt.ToTensor()
    train_ds = FakeData(size=64, image_size=(3, 224, 224), num_classes=3, transform=tf)
    val_ds   = FakeData(size=32, image_size=(3, 224, 224), num_classes=3, transform=tf)
    test_ds  = FakeData(size=32, image_size=(3, 224, 224), num_classes=3, transform=tf)
    return (
        DataLoader(train_ds, batch_size=batch_size, shuffle=True),
        DataLoader(val_ds,   batch_size=batch_size, shuffle=False),
        DataLoader(test_ds,  batch_size=batch_size, shuffle=False),
        train_ds      # to get .classes (FakeData supplies integers 0..N‑1)
    )

def test_one_epoch_training_cpu():
    train_loader, val_loader, test_loader, train_ds = _fake_dataloaders()
    model = create_resnet50_model(num_classes=3, use_pretrained=False)
    # train for 1 epoch on CPU
    model = train_model(model, train_loader, val_loader, epochs=1, lr=1e-3, device=torch.device("cpu"))
    # quick eval (just to run the code path)
    evaluate_model(model, test_loader, class_names=[str(i) for i in range(3)], device=torch.device("cpu"))
