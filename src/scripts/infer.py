#!/usr/bin/env python3
"""
infer.py  –  Fixed-config CNN inference using the checkpoint
checkpoints/resnet50_epoch10.pt. No CLI arguments required.
"""

import torch
from pathlib import Path
from PIL import Image
from torchvision import transforms as T

from src.models.resnet50       import create_resnet50_model
from src.datasets.datamodule   import DataModule, get_dataloaders
from src.config import IMAGE_SIZE

# ────────────────────────────────────────────────────────────────────────────────
# ① CONFIGURATION: adjust these paths & settings as needed
# ────────────────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "checkpoints/resnet50_epoch10.pt"

# Inference on a single image:
QUERY_IMAGE     = "data/inference/NJ_DL1.jpg"
# Or inference on an entire folder (set FOLDER_INFERENCE to True):
FOLDER_INFERENCE = False
QUERY_FOLDER     = "sample_queries/"

BATCH_SIZE       = 32  # for folder inference
# ────────────────────────────────────────────────────────────────────────────────

def load_model():
    # Load the model architecture and weights
    dm = DataModule(data_root="data/processed")
    dm.setup()
    num_classes = len(dm.train_dataset.classes)
    class_names = dm.train_dataset.classes

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = create_resnet50_model(num_classes=num_classes)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device).eval()
    return model, class_names, device

@torch.no_grad()
def preprocess_image(path, tf):
    img = Image.open(path).convert("RGB")
    return tf(img).unsqueeze(0)  # shape (1,3,H,W)

def main():
    model, class_names, device = load_model()

    # Prepare transform (same as validation)
    tf = T.Compose([
        T.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]),
    ])

    # Gather images
    if FOLDER_INFERENCE:
        folder = Path(QUERY_FOLDER)
        images = sorted(folder.rglob("*"))
        images = [p for p in images if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]
    else:
        images = [Path(QUERY_IMAGE)]

    # Run inference
    print(f"[INFO] Running inference on {len(images)} image(s) using {CHECKPOINT_PATH}")
    for idx, img_path in enumerate(images, 1):
        batch = preprocess_image(img_path, tf).to(device)
        logits = model(batch)
        pred = logits.argmax(dim=1).item()
        label = class_names[pred]
        print(f"{idx:>3}. {img_path}  ➜  {label}")

if __name__ == "__main__":
    main()
