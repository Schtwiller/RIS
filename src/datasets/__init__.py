# src/datasets/__init__.py
from pathlib import Path
import json
import numpy as np
from PIL import Image
from torch.utils.data import Dataset

class IDDataset(Dataset):
    def __init__(self, split: str, root: Path = Path("data/processed"), transform=None):
        manifest = root / split / "_manifest.json"
        self.items = json.loads(manifest.read_text())
        self.transform = transform

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        meta = self.items[idx]
        img = Image.open(meta["path"]).convert("RGB")
        if self.transform:
            img = self.transform(image=np.array(img))["image"]
        label = meta["doc_type"]     # or encode to int later
        return img, label
