"""
src/datasets/datamodule.py
--------------------------

A lightweight “data‑module” wrapper (in the spirit of PyTorch‑Lightning) that
handles

    • dataset construction (train / val / test)
    • DataLoader creation with the right transforms
    • convenient helper accessors

It assumes you already have:
    - build_datasets(...)  in  src/datasets/dataset_builder.py
    - get_train_transforms(), get_val_transforms(), get_test_transforms()
      in  src/datasets/transforms.py
If those functions are named differently, tweak the imports below.
"""

from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parents[2]   # .../reverse-image-search-model
DEFAULT_DATA_ROOT = REPO_ROOT / "data" / "processed"
from src.config import TRAINING
from torch.utils.data import DataLoader
from .dataset_builder import build_datasets
from .transforms import (
    get_train_transforms,
    get_val_transforms,
    get_test_transforms,
)

BATCH_SIZE = TRAINING["batch_size"]

class DataModule:
    def __init__(
        self,
        data_root: str | Path = DEFAULT_DATA_ROOT,
        batch_size: int = BATCH_SIZE,
        num_workers: int = 4,
        pin_memory: bool = True,
    ):
        self.data_root   = Path(data_root)
        self.batch_size  = batch_size
        self.num_workers = num_workers
        self.pin_memory  = pin_memory

        # These will be filled in during .setup()
        self.train_dataset = None
        self.val_dataset   = None
        self.test_dataset  = None

    # ------------------------------------------------------------------ #
    # main hook – call once from your training script                     #
    # ------------------------------------------------------------------ #
    def setup(self) -> None:
        """Build datasets with the project’s transform pipeline."""
        train_tf = get_train_transforms()
        val_tf   = get_val_transforms()
        test_tf  = get_test_transforms()

        self.train_dataset, self.val_dataset, self.test_dataset = build_datasets(
            self.data_root,
            train_tf,
            val_tf,
            test_tf,
        )

    # ------------------------------------------------------------------ #
    # DataLoader helpers                                                  #
    # ------------------------------------------------------------------ #
    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
        )

# ---------------------------------------------------------------------- #
# Convenience functional API                                             #
# ---------------------------------------------------------------------- #
def get_dataloaders(
    data_root: str | Path = "data",
    batch_size: int = 32,
    num_workers: int = 4,
    pin_memory: bool = True,
):
    """
    One‑liner helper if you don’t need the full DataModule object.

    Example:
        train_loader, val_loader, test_loader = get_dataloaders(batch_size=64)
    """
    dm = DataModule(
        data_root=data_root,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    dm.setup()
    return dm.train_dataloader(), dm.val_dataloader(), dm.test_dataloader()
