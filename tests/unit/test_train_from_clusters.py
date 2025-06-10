import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

import pandas as pd, pathlib, numpy as np, types, torch, pytest, shutil

@pytest.fixture
def mock_cluster_map(tmp_path):
    df = pd.DataFrame({
        "image_path": [],
        "cluster_id": []
    })
    img_root = tmp_path / "imgs"
    img_root.mkdir()
    # two clusters, 3 imgs each
    for cid in (0, 1):
        for i in range(3):
            p = img_root / f"{cid}_{i}.jpg"
            torch.zeros(1, dtype=torch.uint8).numpy().tofile(p)
            df.loc[len(df)] = [str(p), cid]
    df.to_csv("artifacts/clusters/cluster_map.csv", index=False)
    return df

def test_train_from_clusters(monkeypatch, tmp_path, mock_cluster_map):
    import src.config as CFG
    CFG.TRAINING.update(dict(epochs=1, batch_size=2, lr=1e-3))

    # stub DataModule to skip real images
    class FakeDM:
        def __init__(self, **kw): self.train_dataset = types.SimpleNamespace(classes=["0", "1"])
        def setup(self): pass
        def train_dataloader(self): return [torch.rand(2, 3, 64, 64)]
        def val_dataloader(self):   return [torch.rand(2, 3, 64, 64)]
    monkeypatch.setattr("src.datasets.datamodule.DataModule", FakeDM, raising=True)

    # stub train_model
    def fake_train(model, *a, **k): return model
    monkeypatch.setattr("src.training.trainer.train_model", fake_train, raising=True)

    # dummy model builder
    import torch.nn as nn
    monkeypatch.setattr("src.models.resnet50.create_resnet50_model",
                        lambda *a, **k: nn.Linear(4, 2), raising=True)

    import src.clustering.train_from_clusters as tfc
    tfc.parse = lambda: types.SimpleNamespace(map="artifacts/clusters/cluster_map.csv",
                                              checkpoint=None, config=None)
    tfc.main()

    ckpt = pathlib.Path("artifacts/checkpoints/resnet50_clusters.pt")
    assert ckpt.exists()