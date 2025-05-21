"""
tests/unit/test_extractor.py
----------------------------
Smoke‑tests feature extraction to ensure:

1. The returned embedding array has shape (N, 2048) and dtype float32.
2. No NaNs or Infs are present.
3. Works entirely on CPU with a randomly initialised ResNet‑50.

We generate four synthetic RGB images on the fly, write them
to a temporary directory (pytest's tmp_path fixture), and
run `extract_features`.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

from pathlib import Path
import numpy as np
from PIL import Image
import torch

from src.models.resnet50 import create_resnet50_model
from src.features.extractor import extract_features


def _make_dummy_rgb(path: Path, color):
    img = Image.new("RGB", (224, 224), color=color)
    img.save(path)


def test_extract_features_shapes(tmp_path: Path):
    # --- create 4 dummy images (red, green, blue, gray) -------------
    img_paths = []
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (120, 120, 120)]
    for i, col in enumerate(colors):
        p = tmp_path / f"img_{i}.jpg"
        _make_dummy_rgb(p, col)
        img_paths.append(str(p))

    # --- build a small ResNet‑50 (random weights, cpu) --------------
    model = create_resnet50_model(num_classes=4, use_pretrained=False)
    model.eval()

    # --- run extractor ---------------------------------------------
    feats = extract_features(
        model,
        img_paths,
        batch_size=2,
        device=torch.device("cpu"),
    )

    # --- assertions ------------------------------------------------
    assert feats.shape == (4, 2048), "Unexpected feature shape"
    assert feats.dtype == np.float32, "Features must be float32"
    assert not np.isnan(feats).any(), "NaNs detected in features"
    assert not np.isinf(feats).any(), "Infs detected in features"
