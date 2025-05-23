import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

import numpy as np
import torch
import pytest
from PIL import Image

import src.retrieval.search as search_mod
from src.retrieval.search import retrieve_similar_images
from src.indexing.faiss_index import ClassIndexStore

# ------------------------------------------------------------------ #
# Fixtures
# ------------------------------------------------------------------ #
@pytest.fixture
def dummy_image(tmp_path: Path) -> str:
    """Create a 10 × 10 black RGB image and return its path."""
    img = np.zeros((10, 10, 3), dtype=np.uint8)
    p = tmp_path / "query.jpg"
    Image.fromarray(img).save(p)
    return str(p)


@pytest.fixture
def index_store() -> ClassIndexStore:
    """2‑D toy index: class A → [0,0], class B → [1,1]."""
    store = ClassIndexStore(dim=2)

    idxA = store.leftovers.__class__(2)
    idxA.add(np.array([[0.0, 0.0]], dtype="float32"), ["a.jpg"])

    idxB = store.leftovers.__class__(2)
    idxB.add(np.array([[1.0, 1.0]], dtype="float32"), ["b.jpg"])

    store.class_indices = {"A": idxA, "B": idxB}
    return store


class DummyModel(torch.nn.Module):
    """Returns the same logits tensor no matter the input."""
    def __init__(self, logits: torch.Tensor):
        super().__init__()
        self._logits = logits

    def forward(self, x):
        return self._logits.repeat(x.size(0), 1)


# Monkey‑patch extractor so we control embeddings
@pytest.fixture(autouse=True)
def patch_extract(monkeypatch):
    def fake_extract(model, img_paths, batch_size=1, device=None):
        """
        ➜ [0,0] if filename contains 'a'
        ➜ [2,2] otherwise
        """
        out = []
        for p in img_paths:
            name = Path(p).name.lower()
            out.append([0.0, 0.0] if "a" in name else [2.0, 2.0])
        return np.array(out, dtype="float32")

    monkeypatch.setattr(search_mod, "extract_features", fake_extract)
    yield

# ------------------------------------------------------------------ #
# Tests
# ------------------------------------------------------------------ #
def test_retrieve_exact_match(dummy_image, index_store):
    """
    Query filename contains 'a' → embedding [0,0].
    Distance to class A vector is 0 (< threshold) so we expect A.
    """
    # Rename query so basename contains 'a'
    img_path = Path(dummy_image)
    new_path = img_path.with_name("a_query.jpg")
    img_path.rename(new_path)

    model = DummyModel(torch.tensor([[5.0, 0.0]]))  # predicts A

    results, pred_class = retrieve_similar_images(
        img_path=str(new_path),
        model=model,
        index_store=index_store,
        class_names=["A", "B"],
        top_k=1,
        n_fallback=1,
        match_threshold=0.5,
        device=torch.device("cpu"),
    )

    assert pred_class == "A"
    assert results == [("a.jpg", pytest.approx(0.0, abs=1e-6))]


def test_retrieve_fallback(dummy_image, index_store):
    """
    Threshold 0.0 forces fallback: A fails (dist 8.0 > 0), B returned.
    Embedding [2,2] vs B vector [1,1] → squared distance = 2.
    """
    model = DummyModel(torch.tensor([[0.0, 0.0]]))  # ties → order A then B

    results, pred_class = retrieve_similar_images(
        img_path=dummy_image,
        model=model,
        index_store=index_store,
        class_names=["A", "B"],
        top_k=1,
        n_fallback=2,
        match_threshold=0.0,
        device=torch.device("cpu"),
    )

    assert pred_class == "A"               # classifier top‑1
    assert results[0][0] == "b.jpg"        # retrieved from class B
    assert results[0][1] == pytest.approx(2.0, rel=1e-3)  # squared L2


def test_query_no_valid_class(dummy_image, index_store):
    """If supplied labels have no index, expect ValueError."""
    model = DummyModel(torch.tensor([[0.0, 0.0, 5.0]]))  # predicts class C

    with pytest.raises(ValueError):
        retrieve_similar_images(
            img_path=dummy_image,
            model=model,
            index_store=index_store,
            class_names=["A", "B", "C"],
            top_k=1,
            n_fallback=1,
            match_threshold=1.0,
            device=torch.device("cpu"),
        )