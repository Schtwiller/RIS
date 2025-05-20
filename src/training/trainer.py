"""
src/training/trainer.py
-----------------------

Reusable training / evaluation utilities for the project.
Everything here is *import‑only* (no side‑effects) so tests, notebooks,
and CLI scripts can call these functions safely.

Exposes
-------
create_resnet50_model() : convenience builder
train_model()           : main training loop
evaluate_model()        : test‑set evaluation + report
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Iterable, List, Union, Tuple
from src.config import OPTIMIZER, TRAINING

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, accuracy_score
from torchvision import models

# --------------------------------------------------------------- #
#  Model helper
# --------------------------------------------------------------- #
def create_resnet50_model(
    num_classes: int,
    *,
    use_pretrained: bool = True,
    feature_extract: bool = False,
) -> nn.Module:
    """
    Build a ResNet‑50, optionally loading ImageNet weights and optionally
    freezing all conv layers (feature‑extract regime).

    Parameters
    ----------
    num_classes : int
        Number of output classes for the final FC layer.
    use_pretrained : bool
        If True, load ImageNet weights.
    feature_extract : bool
        If True, freeze all parameters except the last FC layer.
    """
    # torchvision >=0.13 uses .ResNet50_Weights
    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT if use_pretrained else None
    )

    if feature_extract:
        for param in model.parameters():
            param.requires_grad_(False)

    in_feats = model.fc.in_features  # 2048 for ResNet‑50
    model.fc = nn.Linear(in_feats, num_classes)
    return model


# --------------------------------------------------------------- #
#  Training loop
# --------------------------------------------------------------- #
def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    *,
    epochs: int = TRAINING["epochs"],
    lr: float = OPTIMIZER["lr"],
    weight_decay: float = OPTIMIZER["weight_decay"],
    device: torch.device | str | None = None,
    ckpt_dir: str | Path | None = None,
    scheduler_step: int = 2,      # patience (epochs w/o val‑loss improve)
    scheduler_gamma: float = 0.1, # LR scale factor when plateau
    verbose: bool = True,
) -> nn.Module:
    """
    Fine‑tune `model` on `train_loader`, validating on `val_loader`.
    Adds:
      • ReduceLROnPlateau scheduler   (patience = `scheduler_step`,
                                       factor   = `scheduler_gamma`)
      • Early‑stopping (patience = 3 epochs on val‑loss)
      • Best‑model checkpoint saving.
    Returns the best‑val‑accuracy model.
    """
    device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    print(
        f"\n▶️  Training on: {device}  "
        f"{torch.cuda.get_device_name(0) if device.type == 'cuda' else ''}\n"
    )
    model.to(device)

    criterion  = nn.CrossEntropyLoss()
    optimizer  = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler  = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=scheduler_gamma,
        patience=scheduler_step
    )

    best_acc   = 0.0
    best_state = copy.deepcopy(model.state_dict())

    best_val_loss      = float("inf")
    epochs_no_improve  = 0
    early_stop_patience = 3

    for epoch in range(1, epochs + 1):
        # ── training loop ───────────────────────────────────────────── #
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            out  = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ── validation loop ─────────────────────────────────────────── #
        val_loss, val_acc = _evaluate_loop(model, val_loader, criterion, device)

        if verbose:
            print(
                f"[{epoch:02d}/{epochs}]  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}  "
                f"lr={optimizer.param_groups[0]['lr']:.6f}"
            )

        # ── LR scheduler step (plateau on val‑loss) ─────────────────── #
        scheduler.step(val_loss)

        # ── best‑model tracking (by val‑accuracy) ───────────────────── #
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = copy.deepcopy(model.state_dict())

        # ── early‑stopping on val‑loss ──────────────────────────────── #
        if val_loss + 1e-6 < best_val_loss:        # significant improvement
            best_val_loss = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= early_stop_patience:
            if verbose:
                print(f"\n🛑 Early stopping at epoch {epoch} "
                      f"(no val‑loss improvement for {early_stop_patience} epochs)")
            break

    # ── load & optionally save the best model ───────────────────────── #
    model.load_state_dict(best_state)
    if ckpt_dir is not None:
        ckpt_dir = Path(ckpt_dir)
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        ckpt_path = ckpt_dir / "best_resnet50.pt"
        torch.save(best_state, ckpt_path)
        if verbose:
            print(f"✅  best model checkpoint saved to {ckpt_path.resolve()}")

    return model


# --------------------------------------------------------------- #
#  Evaluation / utility
# --------------------------------------------------------------- #
def evaluate_model(
    model: nn.Module,
    test_loader: torch.utils.data.DataLoader,
    *,
    class_names: List[str] | Iterable[str],
    device: torch.device | str | None = None,
    verbose: bool = True,
) -> dict:
    """
    Evaluate `model` on `test_loader`. Prints accuracy + classification report.

    Returns
    -------
    dict
        {"accuracy": float, "report": str}
    """
    device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)
    model.eval()

    all_preds: list[int] = []
    all_labels: list[int] = []

    with torch.no_grad():
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            out = model(imgs)
            preds = out.argmax(dim=1).cpu().tolist()
            all_preds.extend(preds)
            all_labels.extend(labels.tolist())

    acc = accuracy_score(all_labels, all_preds)
    report = classification_report(all_labels, all_preds, target_names=list(class_names), zero_division=0)

    if verbose:
        print(f"\nTest accuracy: {acc:.4f}\n")
        print(report)

    return {"accuracy": acc, "report": report}


# --------------------------------------------------------------- #
#  Internal helpers
# --------------------------------------------------------------- #
def _evaluate_loop(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Return (val_loss, val_accuracy) for one pass over `dataloader`."""
    model.eval()
    loss_sum, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            loss_sum += loss.item() * imgs.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return loss_sum / total, correct / total

