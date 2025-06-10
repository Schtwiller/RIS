"""
Stages 1 and 2 - extracting 2048-D ResNet features with PCA-128 and HDBSCAN
Outputs:
    - artifacts/pca/pca_2048to128.pkl
    - artifacts/embeddings/unlabeled_embeddings.npz
    - artifacts/clusters/cluster_map.csv
"""
import argparse, pickle, json, yaml, pathlib, numpy as np, pandas as pd, hdbscan, torch
from sklearn.decomposition import PCA
from src.models.resnet50 import create_resnet50_model
from src.features.extractor import extract_features
from src.config import EMBEDDING, CLUSTERING, GROUPING, TRAINING

def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--images", default="data/raw/unlabeled")
    ap.add_argument("--checkpoint", default=None, help="optional pretrained backbone .pt (loads conv weights)")
    return ap.parse_args()

def main():
    args = parse_args()
    out_dir = pathlib.Path("artifacts")
    out_dir.mkdir(exist_ok=True, parents=True)

    # 1. Load Backbone
    model = create_resnet50_model(num_classes=1000, use_pretrained=True, feature_extract=True)  # freeze conv by default

    if args.checkpoint:
        state = torch.load(args.checkpoint, map_location="cpu")
        model.load_state_dict(state, strict=False)
    model.eval()

    # 2. Enumerate Images
    image_paths = sorted(str(p) for ext in ("*.jpg", "*.png") for p in pathlib.Path(args.images).rglob(ext))
    print(f"[cluster] found {len(image_paths):,} images")

    # 3. Extract 2048-D
    feats = extract_features(model, image_paths, batch_size=TRAINING["batch_size"])
    np.savez(out_dir / "embeddings/unlabeled_embeddings.npz",
             features=feats, paths=np.array(image_paths))

    # 4. PCA
    n_samples = feats.shape[0]
    pca_dim = min(EMBEDDING["pca_dim"], n_samples)
    pca = PCA(n_components=pca_dim, whiten=True, random_state=42)
    feats_128 = pca.fit_transform(feats).astype("float32")
    pathlib.Path("artifacts/pca").mkdir(exist_ok=True, parents=True)
    pickle.dump(pca, open("artifacts/pca/pca_2048to128.pkl", "wb"))

    # 5. HDBSCAN
    clust = hdbscan.HDBSCAN(min_cluster_size=CLUSTERING["min_cluster_size"],
                            min_samples=CLUSTERING["min_samples"],
                            metric=CLUSTERING["metric"])
    labels = clust.fit_predict(feats_128)
    df = pd.DataFrame({"image_path": image_paths, "cluster_id": labels})
    pathlib.Path("artifacts/clusters").mkdir(parents=True, exist_ok=True)
    df.to_csv("artifacts/clusters/cluster_map.csv", index=False)
    print(df["cluster_id"].value_counts())

    # optional grouping trigger
    if (labels >= 0).sum() > GROUPING["trigger_clusters"]:
        print("[cluster] > trigger_clusters - consider running group_clusters.py")

if __name__ == "__main__":
    main()