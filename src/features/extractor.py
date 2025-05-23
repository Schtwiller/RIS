from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from src.datasets.transforms import get_val_transforms
from PIL import Image
from src.config import TRAINING, AUGMENTATION

MEAN = AUGMENTATION["mean"]
STD = AUGMENTATION["std"]
_DEFAULT_TF = get_val_transforms()  # uses the central definition
BATCH_SIZE = TRAINING["batch_size"]


# ---------- public API ----------------------------------------------- #
@torch.no_grad()
def extract_features(
    model: nn.Module,
    image_paths: List[str],
    batch_size: int = BATCH_SIZE,
    device: torch.device | None = None,
    transform=None,
) -> np.ndarray:
    """
    Return float32 array (N, 2048) of embeddings for given image paths.
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    backbone = nn.Sequential(*list(model.children())[:-1]).to(device)

    tf = transform or _DEFAULT_TF
    imgs = [tf(Image.open(p).convert("RGB")) for p in image_paths]
    loader = DataLoader(imgs, batch_size=batch_size, shuffle=False)

    feats = []
    for batch in loader:
        batch = batch.to(device)
        out = backbone(batch).flatten(1)  # (B, 2048)
        feats.append(out.cpu())
    return torch.cat(feats).numpy()
