#!/usr/bin/env python3
"""
Stage 3 – fine‑tune ResNet classifier on HDBSCAN pseudo‑labels
   📥  artifacts/clusters/cluster_map.csv (+ images)
   📤  artifacts/checkpoints/resnet50_clusters.pt
"""
import argparse, yaml, pathlib, torch, pandas as pd, numpy as np, shutil, os, sys
from src.models.resnet50 import create_resnet50_model
from src.datasets.datamodule import DataModule
from src.training.trainer import train_model
from src.config import TRAINING, OPTIMIZER

def _safe_link(src: pathlib.Path, dst: pathlib.Path):
    """Try symlink; fall back to physical copy if not permitted (e.g. Windows)."""
    try:
        dst.symlink_to(src)
    except (OSError, NotImplementedError):
        shutil.copy2(src, dst)          # deep‑copy but always works

def build_image_folders(map_csv: str, root: pathlib.Path):
    df = pd.read_csv(map_csv)
    df = df[df.cluster_id >= 0]

    root.mkdir(parents=True, exist_ok=True)
    for cid, grp in df.groupby("cluster_id"):
        paths = grp.image_path.tolist()
        val_cut = max(1, int(0.2 * len(paths)))
        for i, p in enumerate(paths):
            split = "val" if i < val_cut else "train"
            d = root / split / str(cid)
            d.mkdir(parents=True, exist_ok=True)
            _safe_link(pathlib.Path(p).resolve(), d / pathlib.Path(p).name)

def parse():
    ap = argparse.ArgumentParser()
    ap.add_argument("--map", default="artifacts/clusters/cluster_map.csv")
    ap.add_argument("--checkpoint", default=None, help="backbone weights to start from")
    return ap.parse_args()

def main():
    args = parse()
    folder_root = pathlib.Path("data/clusters_processed")
    if folder_root.exists(): shutil.rmtree(folder_root)
    build_image_folders(args.map, folder_root)

    dm = DataModule(data_root=str(folder_root),
                    batch_size=TRAINING["batch_size"],
                    num_workers=4)
    dm.setup()
    C = len(dm.train_dataset.classes)
    model = create_resnet50_model(num_classes=C, use_pretrained=False, feature_extract=False)

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)

    model = train_model(model,
                        dm.train_dataloader(),
                        dm.val_dataloader(),
                        epochs=TRAINING["epochs"],
                        lr=OPTIMIZER["lr"])
    pathlib.Path("artifacts/checkpoints").mkdir(exist_ok=True, parents=True)
    torch.save(model.state_dict(), f"artifacts/checkpoints/resnet50_clusters.pt")

if __name__ == "__main__":
    main()