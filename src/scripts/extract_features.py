#!/usr/bin/env python3
"""
Extract ResNet‑50 embeddings and save them to .npz
"""
import argparse
from pathlib import Path
import numpy as np
import torch

from src.features.extractor import extract_features
from src.models.resnet50 import create_resnet50_model


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--num_classes", type=int, default=None,
                   help="If omitted, inferred from sub‑folders in data_dir")
    args = p.parse_args()

    data_dir = Path(args.data_dir)
    image_paths, labels = [], []
    for cls_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
        for img in sorted(cls_dir.glob("*")):
            image_paths.append(str(img))
            labels.append(cls_dir.name)

    # infer class count if not supplied
    num_classes = args.num_classes or len(set(labels))

    # load model backbone
    model = create_resnet50_model(num_classes=num_classes)
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt, strict=False)   # ignore FC mismatch

    feats = extract_features(model, image_paths,
                             batch_size=args.batch_size)

    np.savez(args.output,
             features=feats.astype("float32"),
             paths=np.array(image_paths),
             labels=np.array(labels))
    print(f"[✓] saved {len(feats)} embeddings → {args.output}")


if __name__ == "__main__":
    main()
