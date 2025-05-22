"""
src/indexing/faiss_index.py
===========================

Light‑weight wrapper around FAISS that

1.  Builds **one FAISS index per class** from an initial batch of
    (feature‑vector, label) pairs.
2.  Maintains a special *“leftovers”* index that stores embeddings whose
    class is still unknown / too rare.
3.  Dynamically **“graduates”** a label out of the leftovers pool** once it
    accumulates `MIN_SAMPLES_FOR_NEW_INDEX` samples – at that point a brand‑new
    per‑class FAISS index is created and those vectors are migrated.

The code is intentionally simple – it uses `faiss.IndexFlatL2` (exact L2
search).  For large datasets you can swap in IVF / PQ variants with only a
few lines changed.

Typical usage
-------------
```python
from src.indexing.faiss_index import ClassIndexStore

store = ClassIndexStore(dim=2048)

# 1) bulk‑build from training set
store.build(initial_feats, initial_labels, initial_paths)

# 2) add new items online
store.add(class_name, feat_vec, img_path)

# 3) query
vec = extract_features(model, [query_path])[0]

matched_class, dists, paths = index_store.query(
    labels=["TX_DL", "CA_DL", "NY_DL"],
    vec=vec,
    k=5,
    match_threshold=1.1,
)

"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple
from src.config import INDEXING

import faiss
import numpy as np

MIN_SAMPLES_FOR_NEW_INDEX = INDEXING["graduating_threshold"] # “graduation” threshold
TOP_K_IMAGES = INDEXING["top_k_images"]
MATCH_THRESHOLD = INDEXING["match_threshold"]

class _IndexWithPaths:
    """Tiny helper to carry a FAISS index and the parallel list of paths."""
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.paths: List[str] = []

    # ------------------------------------------------------------------ #
    def add(self, vecs: np.ndarray, img_paths: List[str]) -> None:
        """Add N vectors + their paths to this index."""
        assert vecs.shape[0] == len(img_paths), "vectors / paths length mismatch"
        self.index.add(vecs.astype("float32"))
        self.paths.extend(img_paths)

    def search(self, vec: np.ndarray, k: int = TOP_K_IMAGES) -> Tuple[np.ndarray, List[str]]:
        """Return (distances, paths) for the k nearest neighbours."""
        D, I = self.index.search(vec.astype("float32"), k)
        return D[0], [self.paths[i] for i in I[0]]

class ClassIndexStore:
    """
    Maintains one FAISS index per class + an overflow 'leftovers' index.
    • `build()`  -> bulk‑construct all indices from arrays
    • `add()`    -> online insertion, with automatic graduation
    • `query()`  -> nearest neighbours within a class
    """

    def __init__(self, dim: int):
        self.dim = dim
        self.class_indices: Dict[str, _IndexWithPaths] = {}
        self.leftovers = _IndexWithPaths(dim)

    # ------------------------------------------------------------------ #
    # Bulk build                                                         #
    # ------------------------------------------------------------------ #
    def build(
            self,
            features: np.ndarray,
            labels: List[str],
            paths: List[str],
    ) -> None:
        """Create indices from a dataset in one go."""
        grouped = defaultdict(list)
        for vec, lab, p in zip(features, labels, paths):
            grouped[str(lab)].append((vec, p))

        for lab, items in grouped.items():
            vecs = np.vstack([v for v, _ in items])
            ps = [p for _, p in items]
            if len(items) < MIN_SAMPLES_FOR_NEW_INDEX:
                self.leftovers.add(vecs, [f"{lab}|{p}" for p in ps])
            else:
                idx = _IndexWithPaths(self.dim)
                idx.add(vecs, ps)
                self.class_indices[lab] = idx

    # ------------------------------------------------------------------ #
    # Online add                                                         #
    # ------------------------------------------------------------------ #
    def add(self, label: str, vec: np.ndarray, img_path: str) -> None:
        """
        Add a single image embedding.

        • If `label` already has its own index  → add there.
        • Else  add to leftovers; if that label now has ≥ threshold samples,
          migrate them into a brand‑new per‑class index.
        """
        label = str(label)
        vec = vec.astype("float32").reshape(1, -1)

        if label in self.class_indices:
            self.class_indices[label].add(vec, [img_path])
            return

        # add to leftovers, but tag with label via paths metadata
        self.leftovers.add(vec, [f"{label}|{img_path}"])

        # check if enough samples of this label exist inside leftovers
        label_mask = [p.startswith(f"{label}|") for p in self.leftovers.paths]
        if sum(label_mask) >= MIN_SAMPLES_FOR_NEW_INDEX:
            # graduate: create new index and migrate
            new_idx = _IndexWithPaths(self.dim)
            to_keep_vecs, to_keep_paths = [], []
            for keep, (path, v) in zip(label_mask, zip(self.leftovers.paths, self.leftovers.index.reconstruct_n(0,
                                                                                                                len(self.leftovers.paths)))):
                if keep:
                    new_idx.add(v.reshape(1, -1), [path.split("|", 1)[1]])
                else:
                    to_keep_vecs.append(v)
                    to_keep_paths.append(path)

            # rebuild leftovers with remaining items
            self.leftovers = _IndexWithPaths(self.dim)
            if to_keep_vecs:
                self.leftovers.add(np.vstack(to_keep_vecs), to_keep_paths)

            self.class_indices[label] = new_idx
            print(f"[faiss_index]  Graduated new class index '{label}' "
                  f"with {new_idx.index.ntotal} vectors.")

    # ------------------------------------------------------------------ #
    # Search                                                             #
    # ------------------------------------------------------------------ #
    def query(
        self,
        labels: str | list[str],
        vec: np.ndarray,
        k: int = TOP_K_IMAGES,
        match_threshold: float = MATCH_THRESHOLD,
    ) -> Tuple[str, np.ndarray, List[str]]:
        """
        Try searching each label in order. Return:
            (matched_label, distances, paths)

        If none of the labels yield a good match (based on top-1 distance),
        fall back to the best among them.
        """
        vec = vec.astype("float32").reshape(1, -1)

        if isinstance(labels, str):
            labels = [labels]

        best_result = None
        best_dist = float("inf")
        best_label = None

        for lab in labels:
            lab = str(lab)
            if lab not in self.class_indices:
                continue

            try:
                dists, paths = self.class_indices[lab].search(vec, k)
                top1_dist = dists[0]

                if top1_dist < match_threshold:
                    return lab, dists, paths

                if top1_dist < best_dist:
                    best_dist = top1_dist
                    best_result = (dists, paths)
                    best_label = lab

            except Exception as e:
                print(f"[WARN] Failed search on class '{lab}': {e}")

        if best_result:
            print(f"[Fallback] No match passed threshold {match_threshold:.2f}, "
                  f"returning closest from '{best_label}' (dist={best_dist:.4f})")
            return best_label, *best_result
        else:
            raise ValueError("No valid indices to search among the given labels.")

