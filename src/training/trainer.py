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
from pathlib import Path
from typing import Iterable, List, Union

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
    epochs: int = 10,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
    device: torch.device | str | None = None,
    ckpt_dir: str | Path | None = None,
    scheduler_step: int = 7,
    scheduler_gamma: float = 0.1,
    verbose: bool = True,
) -> nn.Module:
    """
    Train `model` on `train_loader`, validating on `val_loader` each epoch.

    Parameters
    ----------
    model : nn.Module
    train_loader : DataLoader
    val_loader   : DataLoader
    epochs       : int
    lr           : float
    weight_decay : float
    device       : torch.device | str | None
        'cuda', 'cpu', or torch.device.  If None → choose automatically.
    ckpt_dir     : str | Path | None
        If provided, saves the *best* validation‑accuracy model to this dir.
    scheduler_step, scheduler_gamma : int, float
        Params for ReduceLROnPlateau‑like step scheduler.
    verbose : bool
        If True, prints per‑epoch stats.

    Returns
    -------
    nn.Module
        The trained *best* model (w. highest val accuracy).
    """
    device = torch.device(
        device if device is not None else ("cuda" if torch.cuda.is_available() else "cpu")
    )
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=lr,
        weight_decay=weight_decay,
    )
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, step_size=scheduler_step, gamma=scheduler_gamma
    )

    best_acc = 0.0
    best_state = None

    for epoch in range(1, epochs + 1):
        # -------------------- train -------------------- #
        model.train()
        running_loss = 0.0
        for imgs, labels in train_loader:
            imgs, labels = imgs.to(device), labels.to(device)

            optimizer.zero_grad(set_to_none=True)
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * imgs.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        # ------------------ validate ------------------- #
        val_loss, val_acc = _evaluate_loop(model, val_loader, criterion, device)
        scheduler.step()

        if verbose:
            print(
                f"[{epoch:02d}/{epochs}]  "
                f"train_loss={train_loss:.4f}  "
                f"val_loss={val_loss:.4f}  "
                f"val_acc={val_acc:.4f}"
            )

        # save best
        if val_acc > best_acc:
            best_acc = val_acc
            best_state = model.state_dict()

    if best_state is not None:
        model.load_state_dict(best_state)
        if ckpt_dir is not None:
            ckpt_dir = Path(ckpt_dir)
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            path = ckpt_dir / "best_resnet50.pt"
            torch.save(best_state, path)
            if verbose:
                print(f"✅  best model checkpoint saved to {path.resolve()}")
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
    loader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device,
):
    """Internal helper to compute loss + accuracy for a given loader."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs, labels = imgs.to(device), labels.to(device)
            out = model(imgs)
            loss = criterion(out, labels)
            running_loss += loss.item() * imgs.size(0)

            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    avg_loss = running_loss / len(loader.dataset)
    acc = correct / total if total else 0.0
    return avg_loss, acc
