import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

import pathlib, pickle, types, numpy as np, pandas as pd
import pytest

@pytest.fixture
def dummy_embeddings(monkeypatch, tmp_path):
    # minimal PCA & checkpoint to satisfy script
    import pickle, torch
    from sklearn.decomposition import PCA
    # make sure n_samples >= n_components
    X = np.random.rand(12, 2048).astype("float32")
    pca = PCA(n_components=8).fit(X)
    pathlib.Path("artifacts/pca").mkdir(parents=True, exist_ok=True)
    pickle.dump(pca, open("artifacts/pca/pca_2048to128.pkl", "wb"))

    # fake model checkpoint
    torch.save({}, "artifacts/checkpoints/resnet50_clusters.pt")

    # cluster map (2 clusters + 1 noise)
    img_dir = tmp_path / "img"
    img_dir.mkdir()
    paths, labels = [], []
    for i in range(3):
        p = img_dir / f"{i}.jpg"; p.touch()
        paths.append(str(p)); labels.append(0 if i < 1 else (1 if i < 2 else -1))
    pd.DataFrame({"image_path": paths, "cluster_id": labels}).to_csv(
        "artifacts/clusters/cluster_map.csv", index=False)

    # stub extractor (returns deterministic rand)
    def fake_extract(*a, **k):
        return np.stack([np.ones(2048)*i for i in range(3)]).astype("float32")
    monkeypatch.setattr("src.features.extractor.extract_features",
                        fake_extract, raising=True)

    # stub model builder
    import torch.nn as nn
    monkeypatch.setattr("src.models.resnet50.create_resnet50_model",
                        lambda *a, **k: nn.Identity(), raising=True)

def test_build_indices(monkeypatch, dummy_embeddings):
    import src.clustering.build_faiss_from_clusters as bf
    bf.main()
    out = pathlib.Path("artifacts/indices")
    files = {p.name for p in out.glob("*")}
    assert "0.index" in files and "1.index" in files and "-1.index" in files
