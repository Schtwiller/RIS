import argparse
from pathlib import Path
import numpy as np
import torch

from src.features.extractor import extract_features
from src.models.resnet50 import create_resnet50_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", required=True, help="Path to class-based directory of images")
    parser.add_argument("--checkpoint", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", required=True, help="Where to save the output .npz file")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_classes", type=int, required=True)
    args = parser.parse_args()

    # Collect image paths and class labels
    data_dir = Path(args.data_dir)
    image_paths, labels = [], []
    for class_dir in sorted(data_dir.iterdir()):
        if not class_dir.is_dir():
            continue
        for img_path in sorted(class_dir.glob("*")):
            image_paths.append(str(img_path))
            labels.append(class_dir.name)

    # Load model
    model = create_resnet50_model(num_classes=args.num_classes)
    model.load_state_dict(torch.load(args.checkpoint, map_location="cpu"))

    # Extract features
    features = extract_features(model, image_paths, batch_size=args.batch_size)

    # Save features, paths, and labels
    np.savez(args.output,
             features=features.astype("float32"),
             paths=np.array(image_paths),
             labels=np.array(labels))
    print(f"[✓] Saved {len(features)} features to {args.output}")


if __name__ == "__main__":
    main()
