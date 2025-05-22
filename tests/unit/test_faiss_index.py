import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[2]))  # allow imports from src/

import numpy as np
import pytest

import faiss
import src.indexing.faiss_index as fi
from src.indexing.faiss_index import ClassIndexStore, _IndexWithPaths

@pytest.fixture(autouse=True)
def reset_threshold(monkeypatch):
    # Use small thresholds for testing
    monkeypatch.setattr(fi, "MIN_SAMPLES_FOR_NEW_INDEX", 3)
    yield
    # no teardown needed


def test_index_with_paths_basic():
    """_IndexWithPaths should store vectors and return correct distances & paths."""
    idx = _IndexWithPaths(dim=2)
    # three orthonormal basis vectors
    vecs = np.eye(2, dtype="float32")
    paths = ["p0", "p1"]
    # add two vectors
    idx.add(vecs, paths)
    # search the first vector against itself
    D, P = idx.search(vecs[0:1], k=2)
    assert isinstance(D, np.ndarray)
    assert isinstance(P, list)
    # we added exactly 2 vectors, so ntotal=2
    assert idx.index.ntotal == 2
    # best match is itself: distance approx 0
    assert pytest.approx(0.0, abs=1e-6) == D[0]
    # the corresponding path is correct
    assert P[0] == "p0"
    # second-best match is the other vector
    assert P[1] == "p1"


def test_build_creates_and_leaves_leftovers():
    """
    build() should:
      - create a class index if #samples >= threshold
      - otherwise put them in leftovers
    """
    store = ClassIndexStore(dim=2)
    # prepare features and labels
    feats = np.array([
        [1.0, 0.0],  # A
        [1.1, 0.1],  # A
        [0.0, 1.0],  # B
    ], dtype="float32")
    labels = ["A", "A", "B"]
    paths  = ["a1", "a2", "b1"]

    # threshold is 3 (via fixture), so A(2)<3→leftovers, B(1)<3→leftovers
    store.build(feats, labels, paths)
    assert "A" not in store.class_indices
    assert "B" not in store.class_indices
    # leftovers should have 3 vectors
    assert store.leftovers.index.ntotal == 3
    # all paths should be in leftovers, in the same order
    assert store.leftovers.paths == ["A|a1", "A|a2", "B|b1"]

    # now build with lower threshold: monkeypatch to 2
    fi.MIN_SAMPLES_FOR_NEW_INDEX = 2
    store2 = ClassIndexStore(dim=2)
    store2.build(feats, labels, paths)
    # A has 2 samples => class index; B has 1 => leftovers
    assert "A" in store2.class_indices
    assert "B" not in store2.class_indices
    assert store2.leftovers.index.ntotal == 1
    assert store2.leftovers.paths == ["B|b1"]


def test_add_and_graduation(monkeypatch):
    """
    add() should:
      - keep adding to leftovers until sample count >= threshold
      - then create a new class index and remove those from leftovers
    """
    store = ClassIndexStore(dim=2)
    v = np.array([1.0, 1.0], dtype="float32")

    # add 2 items: threshold=3, so still in leftovers
    store.add("X", v, "x1")
    store.add("X", v, "x2")
    assert "X" not in store.class_indices
    assert store.leftovers.index.ntotal == 2
    assert all(p.startswith("X|") for p in store.leftovers.paths)

    # 3rd addition triggers graduation
    store.add("X", v, "x3")
    # class index must now exist, with 3 vectors
    assert "X" in store.class_indices
    idx = store.class_indices["X"]
    assert idx.index.ntotal == 3
    # leftovers should be empty
    assert store.leftovers.index.ntotal == 0

    # subsequent adds go directly to the class index
    store.add("X", v, "x4")
    assert idx.index.ntotal == 4


def test_query_and_fallback():
    """
    query() should:
      - search each class in order
      - return the first whose top1 distance < threshold
      - otherwise fallback to best among them
    """
    store = ClassIndexStore(dim=2)

    # build two small indices manually
    idxA = _IndexWithPaths(2)
    idxB = _IndexWithPaths(2)
    idxA.add(np.array([[10,10]], dtype="float32"), ["pA"])
    idxB.add(np.array([[0,0]],   dtype="float32"), ["pB"])
    store.class_indices = {"A": idxA, "B": idxB}

    q = np.array([0,0], dtype="float32")

    # Case 1: match_threshold low → fallback to B
    lab, dists, paths = store.query(["A","B"], q, k=1, match_threshold=1.0)
    assert lab == "B"
    assert paths == ["pB"]
    assert pytest.approx(0.0) == dists[0]

    # Case 2: threshold high enough → pick A for a query near A
    qA = np.array([10,10], dtype="float32")
    lab2, d2, p2 = store.query("A", qA, k=1, match_threshold=100.0)
    assert lab2 == "A"
    assert p2 == ["pA"]
    assert pytest.approx(0.0) == d2[0]

    # Case 3: no valid labels leads to ValueError
    with pytest.raises(ValueError):
        store.query(["C"], q, k=1, match_threshold=1.0)
