#!/usr/bin/env python3
"""
Minimal classification trainer using ResNet-18.
Usage:
    python -m src.training.train --epochs 5 --batch_size 32 --lr 1e-3
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet18

from src.datasets import IDDataset
from src.datasets.transforms import build_transforms

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root",  type=str, default="data/processed")
    p.add_argument("--epochs",     type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--lr",         type=float, default=1e-3)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    return p.parse_args()

def encode_vocab(dataset):
    labels = [item["doc_type"] for item in dataset.items]
    vocab  = {lbl: idx for idx, lbl in enumerate(sorted(set(labels)))}
    return vocab

def train_one_epoch(model, loader, vocab, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    for imgs, lbls in loader:
        imgs = imgs.to(device)
        # lbls is a list of strings
        y = torch.tensor([vocab[l] for l in lbls], dtype=torch.long, device=device)

        optimizer.zero_grad()
        logits = model(imgs)
        loss   = criterion(logits, y)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * imgs.size(0)

    return running_loss / len(loader.dataset)

def validate(model, loader, vocab, criterion, device):
    model.eval()
    val_loss = 0.0
    correct  = 0
    total    = 0
    with torch.no_grad():
        for imgs, lbls in loader:
            imgs = imgs.to(device)
            y    = torch.tensor([vocab[l] for l in lbls], dtype=torch.long, device=device)
            logits = model(imgs)
            loss   = criterion(logits, y)
            val_loss += loss.item() * imgs.size(0)

            preds = logits.argmax(dim=1)
            correct += (preds == y).sum().item()
            total   += imgs.size(0)

    return val_loss / len(loader.dataset), correct / total

def main():
    args = parse_args()
    device = args.device

    # ─── Datasets & Loaders ───────────────────────────────────────────────
    train_tf = build_transforms("train")
    val_tf   = build_transforms("val")

    train_ds = IDDataset("train", Path(args.data_root), transform=train_tf)
    val_ds   = IDDataset("val",   Path(args.data_root), transform=val_tf)

    vocab = encode_vocab(train_ds)
    num_classes = len(vocab)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,  num_workers=args.workers
    )
    val_loader   = DataLoader(
        val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=args.workers
    )

    # ─── Model, Loss & Optimizer ─────────────────────────────────────────
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # ─── Training Loop ───────────────────────────────────────────────────
    print(f"Train samples: {len(train_ds)}, Val samples: {len(val_ds)}")
    start_time = time.time()
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, vocab, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, vocab, criterion, device)

        print(
            f"[Epoch {epoch}/{args.epochs}] "
            f"train_loss={train_loss:.4f}  "
            f"val_loss={val_loss:.4f}  "
            f"val_acc={val_acc:.3%}"
        )

    print(f"Total time: {(time.time() - start_time):.1f}s")

if __name__ == "__main__":
    main()
