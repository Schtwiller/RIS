#!/usr/bin/env python
"""
scripts/train.py
----------------
CLI entry point for training / evaluating the ResNet‑50 pipeline.

Example:
    python scripts/train.py \
        --data_root data/processed \
        --epochs 10 \
        --batch_size 32 \
        --lr 1e-3
"""
import argparse
from pathlib import Path

import torch

# project imports
from src.datasets.datamodule import DataModule
from src.models.resnet50 import create_resnet50_model
from src.training.trainer import train_model, evaluate_model


def main(args):
    # 1. data
    dm = DataModule(
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=not args.no_pin_memory,
    )
    dm.setup()

    # 2. model
    model = create_resnet50_model(
        num_classes=len(dm.train_dataset.classes),
        use_pretrained=not args.no_pretrained,
        feature_extract=args.feature_extract,
    )

    # 3. train
    model = train_model(
        model,
        dm.train_dataloader(),
        dm.val_dataloader(),
        epochs=args.epochs,
        lr=args.lr,
        device=torch.device(args.device),
    )

    # 4. eval
    evaluate_model(
        model,
        dm.test_dataloader(),
        class_names=dm.train_dataset.classes,
        device=torch.device(args.device),
    )

    # 5. save checkpoint
    ckpt_dir = Path(args.ckpt_dir)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = ckpt_dir / f"resnet50_epoch{args.epochs}.pt"
    torch.save(model.state_dict(), ckpt_path)
    print(f"\n✅  checkpoint saved to {ckpt_path.resolve()}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", default="data/processed")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--feature_extract",
        action="store_true",
        help="freeze conv base and train only the final FC layer",
    )
    p.add_argument(
        "--no_pretrained",
        action="store_true",
        help="start from random weights instead of ImageNet",
    )
    p.add_argument("--no_pin_memory", action="store_true")
    p.add_argument("--ckpt_dir", default="checkpoints")
    main(p.parse_args())
