"""
src/indexing/faiss_index.py
===========================

Light‑weight FAISS helper:

1.  One FAISS index per class  (exact L2 via `IndexFlatL2`)
2.  “Leftovers” pool for rare / unknown labels
3.  Automatic graduation to a dedicated index when a label reaches
    `MIN_SAMPLES_FOR_NEW_INDEX`
4.  Simple bulk builder (`build_class_indices`) + online `ClassIndexStore`
5.  Convenient save/load utilities

You can replace `IndexFlatL2` with IVF/PQ for large datasets.
"""

from __future__ import annotations

import pickle
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import faiss
import numpy as np

# ------------------------------------------------------------------ #
# Config values (fail loud if missing)                               #
# ------------------------------------------------------------------ #
from src.config import INDEXING  # noqa: E402

try:
    MIN_SAMPLES_FOR_NEW_INDEX = INDEXING["graduating_threshold"]
    TOP_K_IMAGES = INDEXING["top_k_images"]
    MATCH_THRESHOLD = INDEXING["match_threshold"]
except KeyError as e:
    raise KeyError(f"config.INDEXING missing key {e!s}") from None


# ------------------------------------------------------------------ #
# Bulk helper                                                        #
# ------------------------------------------------------------------ #
def build_class_indices(
    features: np.ndarray,
    labels: np.ndarray,
    paths: List[str] | None = None,
) -> Dict[str, faiss.IndexFlatL2]:
    """
    Quickly build one `IndexFlatL2` per class from arrays.

    Parameters
    ----------
    features : (N, D) float32 array
    labels   : (N,)   iterable of str / int
    paths    : optional list of image paths (length N).  If supplied, each
               resulting index gets a `.paths` attribute with its list.

    Returns
    -------
    dict {class_label: faiss.IndexFlatL2}
    """
    group_vecs = defaultdict(list)
    group_paths = defaultdict(list)

    for vec, lab, p in zip(features, labels, paths or [None] * len(labels)):
        lab = str(lab)
        group_vecs[lab].append(vec)
        if p is not None:
            group_paths[lab].append(p)

    label_to_index = {}
    for lab, vecs in group_vecs.items():
        vecs = np.vstack(vecs).astype("float32")
        d = vecs.shape[1]
        idx = faiss.IndexFlatL2(d)
        idx.add(vecs)
        if paths is not None:
            idx.paths = group_paths[lab]
        label_to_index[lab] = idx

    return label_to_index


# ------------------------------------------------------------------ #
# Tiny wrapper to keep paths next to a FAISS index                   #
# ------------------------------------------------------------------ #
class _IndexWithPaths:
    def __init__(self, dim: int):
        self.index = faiss.IndexFlatL2(dim)
        self.paths: List[str] = []

    def add(self, vecs: np.ndarray, img_paths: List[str]) -> None:
        assert vecs.shape[0] == len(img_paths), "vectors / paths mismatch"
        self.index.add(vecs.astype("float32"))
        self.paths.extend(img_paths)

    def search(
        self, vec: np.ndarray, k: int = TOP_K_IMAGES
    ) -> Tuple[np.ndarray, List[str]]:
        D, I = self.index.search(vec.astype("float32"), k)
        return D[0], [self.paths[i] for i in I[0]]


# ------------------------------------------------------------------ #
# Online store with leftovers & graduation                           #
# ------------------------------------------------------------------ #
class ClassIndexStore:
    """Holds per‑class indices and a leftovers pool."""

    def __init__(self, dim: int):
        self.dim = dim
        self.class_indices: Dict[str, _IndexWithPaths] = {}
        self.leftovers = _IndexWithPaths(dim)

    # ----- bulk ------------------------------------------------------ #
    def build(self, features: np.ndarray, labels: List[str], paths: List[str]) -> None:
        """
        Build indices from full arrays in one shot.
        """
        grouped = defaultdict(list)
        for v, lab, p in zip(features, labels, paths):
            grouped[str(lab)].append((v, p))

        for lab, items in grouped.items():
            vecs = np.vstack([v for v, _ in items])
            ps = [p for _, p in items]
            if len(items) < MIN_SAMPLES_FOR_NEW_INDEX:
                self.leftovers.add(vecs, [f"{lab}|{p}" for p in ps])
            else:
                idx = _IndexWithPaths(self.dim)
                idx.add(vecs, ps)
                self.class_indices[lab] = idx

    # ----- online add ------------------------------------------------ #
    def add(self, label: str, vec: np.ndarray, img_path: str) -> None:
        """
        Add one vector online; graduate label when it hits threshold.
        """
        label = str(label)
        vec = vec.astype("float32").reshape(1, -1)

        if label in self.class_indices:
            self.class_indices[label].add(vec, [img_path])
            return

        # add to leftovers
        self.leftovers.add(vec, [f"{label}|{img_path}"])

        # check graduation
        mask = np.array([p.startswith(f"{label}|") for p in self.leftovers.paths])
        if mask.sum() >= MIN_SAMPLES_FOR_NEW_INDEX:
            # migrate
            all_vecs = self.leftovers.index.reconstruct_n(0, len(self.leftovers.paths))
            new_idx = _IndexWithPaths(self.dim)

            keep_vecs, keep_paths = [], []
            for flag, path, v in zip(mask, self.leftovers.paths, all_vecs):
                if flag:
                    new_idx.add(v.reshape(1, -1), [path.split("|", 1)[1]])
                else:
                    keep_vecs.append(v)
                    keep_paths.append(path)

            # rebuild leftovers
            self.leftovers = _IndexWithPaths(self.dim)
            if keep_vecs:
                self.leftovers.add(np.vstack(keep_vecs), keep_paths)

            self.class_indices[label] = new_idx
            print(f"[faiss] graduated '{label}' with {new_idx.index.ntotal} vectors")

    # ----- search ---------------------------------------------------- #
    def query(
        self,
        labels: str | list[str],
        vec: np.ndarray,
        k: int = TOP_K_IMAGES,
        match_threshold: float = MATCH_THRESHOLD,
    ) -> Tuple[str, np.ndarray, List[str]]:
        """
        Search labels in order.  Return first label whose top‑1 distance
        < `match_threshold`; otherwise return label with smallest top‑1
        distance.

        Returns (matched_label, squared_distances, paths)
        """
        vec = vec.astype("float32").reshape(1, -1)
        labs = [labels] if isinstance(labels, str) else labels

        best = None  # (lab, dists, paths)

        for lab in labs:
            idx = self.class_indices.get(lab)
            if idx is None:
                continue

            dists, paths = idx.search(vec, k)
            if best is None or dists[0] < best[1][0]:
                best = (lab, dists, paths)

            if dists[0] < match_threshold:
                return lab, dists, paths

        if best:
            return best
        raise ValueError("No valid indices among supplied labels.")

    # ----- save / load convenience ---------------------------------- #
    def save(self, out_dir: Path) -> None:
        out_dir.mkdir(parents=True, exist_ok=True)
        for lab, idx in self.class_indices.items():
            faiss.write_index(idx.index, str(out_dir / f"{lab}.index"))
            pickle.dump(idx.paths, open(out_dir / f"{lab}.paths", "wb"))

    @classmethod
    def load(cls, dim: int, from_dir: Path) -> "ClassIndexStore":
        store = cls(dim)
        for idx_file in Path(from_dir).glob("*.index"):
            lab = idx_file.stem
            idx = _IndexWithPaths(dim)
            idx.index = faiss.read_index(str(idx_file))
            idx.paths = pickle.load(open(idx_file.with_suffix(".paths"), "rb"))
            store.class_indices[lab] = idx
        return store


# what `from src.indexing.faiss_index import *` exports
__all__ = [
    "build_class_indices",
    "ClassIndexStore",
    "_IndexWithPaths",
]
