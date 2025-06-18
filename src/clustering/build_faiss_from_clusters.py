#!/usr/bin/env python3
"""
Stage 4 – build FAISS indices for each cluster + leftovers.

Inputs
------
artifacts/clusters/cluster_map.csv
artifacts/pca/pca_2048to128.pkl
artifacts/checkpoints/resnet50_clusters.pt
Outputs
-------
artifacts/indices/<cluster>.index  (+ .paths)
artifacts/indices/-1.index         (+ -1.paths)   # leftovers
"""
import pathlib, pickle, numpy as np, pandas as pd, torch, faiss
from src.models.resnet50 import create_resnet50_model
from src.features.extractor import extract_features
from src.indexing.faiss_index import build_class_indices

def main() -> None:
    # ------------------------------------------------------------------ I/O
    cluster_csv = pathlib.Path("artifacts/clusters/cluster_map.csv")
    pca_path    = pathlib.Path("artifacts/pca/pca_2048to128.pkl")
    ckpt_path   = pathlib.Path("artifacts/checkpoints/resnet50_clusters.pt")
    out_dir     = pathlib.Path("artifacts/cluster-indices")
    out_dir.mkdir(parents=True, exist_ok=True)

    paths_df = pd.read_csv(cluster_csv)
    pca      = pickle.load(open(pca_path, "rb"))

    # ---------------------------------------------------------------- model
    model = create_resnet50_model(num_classes=1)          # dummy FC
    state = torch.load(ckpt_path, map_location="cpu")
    model.load_state_dict(state, strict=False)
    model.eval()

    # ---------------------------------------------------------- embeddings
    img_paths = paths_df.image_path.tolist()
    feats2048 = extract_features(model, img_paths, batch_size=32)
    feats128  = pca.transform(feats2048).astype("float32")
    dim       = feats128.shape[1]                         # e.g. 128

    # ---------------------------------------------------- per-cluster build
    labels    = paths_df.cluster_id.to_numpy()
    clustered = labels >= 0
    indices   = build_class_indices(
        feats128[clustered],
        labels[clustered].astype(str),
        np.array(img_paths)[clustered].tolist(),
    )

    # ------------------------------------------------ leftovers / noise
    noise_vecs  = feats128[~clustered]
    noise_paths = np.array(img_paths)[~clustered].tolist()
    leftovers   = faiss.IndexFlatL2(dim)
    leftovers.paths = noise_paths                 # always assign (even empty)
    if len(noise_vecs):
        leftovers.add(noise_vecs)

    # ----------------------------------------------------------- save all
    for lab, idx in indices.items():
        faiss.write_index(idx, str(out_dir / f"{lab}.index"))
        pickle.dump(idx.paths, open(out_dir / f"{lab}.paths", "wb"))

    faiss.write_index(leftovers, str(out_dir / "-1.index"))
    pickle.dump(leftovers.paths, open(out_dir / "-1.paths", "wb"))

    print(f"[faiss] wrote {len(indices)} cluster indices + leftovers")

if __name__ == "__main__":
    main()
