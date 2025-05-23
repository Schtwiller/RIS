"""
Fast image–similarity retrieval
===============================

Given a **trained model** and a **ClassIndexStore** (per‑class FAISS
indices), this helper:

1. Pre‑processes a query image with the same resize / normalization used
   in validation.
2. **Classifies** the image to obtain an *ordered* list of candidate
   classes (top‑N soft‑max scores).
3. Extracts the **2048‑D embedding** from the model backbone.
4. Calls `ClassIndexStore.query()` with that list so it can search each
   class index in order, returning the first whose best match distance is
   below `match_threshold`.

Typical usage inside `scripts/query.py`::

    from src.retrieval.search import retrieve_similar_images
    results, pred_class = retrieve_similar_images(
        "query.jpg",
        model,
        index_store,
        class_names,
        top_k=5,
        n_fallback=3,
    )

`results` is a list [(img_path, distance), …].
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as T

from src.config import INDEXING
from src.features.extractor import extract_features
from src.indexing.faiss_index import ClassIndexStore
from src.datasets.transforms import get_val_transforms

_DEFAULT_TF = get_val_transforms()

TOP_K = INDEXING["top_k_images"]
MATCH_THRESHOLD = INDEXING["match_threshold"]
N_FALLBACK = INDEXING["n_fallback"]


# --------------------------------------------------------------------- #
# Core retrieve function                                                #
# --------------------------------------------------------------------- #
@torch.no_grad()
def retrieve_similar_images(
    img_path: str | Path,
    model: torch.nn.Module,
    index_store: ClassIndexStore,
    class_names: List[str],
    *,
    top_k: int = TOP_K,
    n_fallback: int = N_FALLBACK,
    match_threshold: float = MATCH_THRESHOLD,
    device: torch.device | None = None,
) -> Tuple[List[Tuple[str, float]], str]:
    """
    Parameters
    ----------
    img_path : str or Path
        Path to query image.
    model : torch.nn.Module
        Trained model with the ResNet‑50 backbone + classifier head.
    index_store : ClassIndexStore
        Holds FAISS indices per class.
    class_names : list[str]
        Mapping from numeric class index → label string (same order as training).
    top_k : int
        Number of neighbours to return.
    n_fallback : int
        How many top‑k classes (by softmax score) to try as fallbacks.
    match_threshold : float
        Maximum allowed L2 distance for the *best* match in an index to be
        considered “good enough”.  If the first class fails this threshold,
        the next fallback class is tried, and so on.
    device : torch.device or None
        Where to run the model; defaults to CUDA if available.

    Returns
    -------
    results : list[(path, distance)]
    pred_class : str
        The *classifier’s* top‑1 class (for logging).
    """
    img_path = Path(img_path)
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model.eval().to(device)

    # -------------------------- load + preprocess --------------------- #
    transform = _DEFAULT_TF
    img_tensor = transform(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

    # -------------------------- classification ------------------------ #
    logits = model(img_tensor)
    probs = torch.softmax(logits, dim=1)[0]  # (C,)
    topk_vals, topk_idx = probs.topk(k=n_fallback)
    fallback_labels = [class_names[i] for i in topk_idx.tolist()]
    pred_class = fallback_labels[0]

    # -------------------------- embedding ----------------------------- #
    emb = extract_features(model, [str(img_path)], batch_size=1, device=device)[
        0
    ]  # (2048,)

    # -------------------------- search -------------------------------- #
    # index_store.query returns (matched_label, distances, paths)
    _, dists, result_paths = index_store.query(
        labels=fallback_labels,
        vec=emb,
        k=top_k,
        match_threshold=match_threshold,
    )

    # pair paths with distances
    results = list(zip(result_paths, dists.tolist()))
    return results, pred_class
