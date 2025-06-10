import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

import numpy as np, pandas as pd, pytest, shutil, types, pathlib, builtins
from PIL import Image
from importlib import reload

@pytest.fixture
def dummy_images(tmp_path):
    img_dir = tmp_path / "unlabeled"
    img_dir.mkdir()
    for i in range(10):
        img = Image.new("RGB", (64, 64), (i*25, i*25, i*25))
        img.save(img_dir / f"img_{i}.jpg")
    return img_dir

def test_cluster_unlabeled(monkeypatch, tmp_path, dummy_images):
    # --- monkey‑patch heavy modules -----------------------------------------
    import src.config as CFG
    CFG.EMBEDDING["pca_dim"] = 8          # shrink for speed
    CFG.CLUSTERING["min_cluster_size"] = 2

    # dummy resnet feature extractor returns fixed random vectors
    def fake_extract(model, paths, batch_size=32):
        rng = np.random.default_rng(0)
        return rng.random((len(paths), 2048)).astype("float32")
    monkeypatch.setattr("src.features.extractor.extract_features",
                        fake_extract, raising=True)

    # dummy model builder (identity)
    import torch, torch.nn as nn
    def fake_model(*a, **k): return nn.Identity()
    monkeypatch.setattr("src.models.resnet50.create_resnet50_model",
                        fake_model, raising=True)

    # import & run
    from src.features import cluster_unlabeled as cu
    reload(cu)  # ensure patching visible
    monkeypatch.setattr(cu, "parse_args",
                        lambda: types.SimpleNamespace(
                            images=str(dummy_images),
                            checkpoint=None,
                            config=None), raising=False)

    cu.main()

    # --- asserts ------------------------------------------------------------
    pca_path = pathlib.Path("artifacts/pca/pca_2048to128.pkl")
    assert pca_path.exists()

    cmap = pd.read_csv("artifacts/clusters/cluster_map.csv")
    assert len(cmap) == 10                         # all images listed
    assert cmap.cluster_id.isin([-1, 0, 1]).all()  # some clustering happened
