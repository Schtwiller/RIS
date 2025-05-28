#!/usr/bin/env python3
"""
demo.py  —  end‑to‑end demo for the Reverse‑Image‑Search pipeline.

Steps
-----
1. build processed dataset  (data/processed/)
2. train / fine‑tune ResNet‑50  (checkpoints/resnet50_demo.pt)
3. extract train embeddings      (artifacts/embeddings/train_embeddings.npz)
4. build one FAISS index per class (artifacts/indices/*.index)
5. run a query image, print & **visualise** top‑K matches

Run:
    python -m src.demo --query path/to/image.jpg
"""

from __future__ import annotations
import argparse, subprocess, sys, pickle
from pathlib import Path

import numpy as np
import torch
import faiss
import matplotlib.pyplot as plt
from PIL import Image

from src.indexing.faiss_index import build_class_indices, ClassIndexStore, _IndexWithPaths
from src.models.resnet50 import create_resnet50_model
from src.retrieval.search import retrieve_similar_images

# ------------------------------------------------------------------ #
# Global paths (everything under artifacts/)                         #
# ------------------------------------------------------------------ #
ART_DIR  = Path("artifacts")
CKPT_DIR = ART_DIR / "checkpoints"
EMB_DIR  = ART_DIR / "embeddings"
IDX_DIR  = ART_DIR / "indices"

for d in (CKPT_DIR, EMB_DIR, IDX_DIR):
    d.mkdir(parents=True, exist_ok=True)

DEMO_CKPT = CKPT_DIR / "resnet50_demo.pt"
EMB_FILE  = EMB_DIR / "train_embeddings.npz"

# ------------------------------------------------------------------ #
# Helpers                                                            #
# ------------------------------------------------------------------ #
def sh(cmd: str):
    print(f"[cmd] {cmd}")
    subprocess.run(cmd, shell=True, check=True)

def latest_ckpt(dir_: Path) -> Path | None:
    pts = sorted(dir_.glob("*.pt"), key=lambda p: p.stat().st_mtime, reverse=True)
    return pts[0] if pts else None

def show_gallery(query_path: str, results: list[tuple[str, float]], k: int = 5):
    paths = [query_path] + [p for p, _ in results[:k]]
    titles = ["QUERY"] + [f"{i+1}  d={float(np.asarray(d).flat[0]):.2f}"
                          for i, (_, d) in enumerate(results[:k])]

    plt.figure(figsize=(3*len(paths), 3))
    for i, (img_path, title) in enumerate(zip(paths, titles), 1):
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"[warn] cannot open {img_path} ({e})")
            continue
        plt.subplot(1, len(paths), i)
        plt.imshow(img)
        plt.axis("off")
        plt.title(title, fontsize=8)
    plt.tight_layout()
    plt.show()

# ------------------------------------------------------------------ #
# Main                                                               #
# ------------------------------------------------------------------ #
def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--processed_dir", default="data/processed")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--top_k", type=int, default=5)
    ap.add_argument("--query", required=True)
    args = ap.parse_args(argv)

    processed = Path(args.processed_dir)

    # 1) processed dataset ------------------------------------------------
    if not processed.exists():
        sh("python src/datasets/dataset_builder.py")

    # 2) train / reuse checkpoint -----------------------------------------
    if not DEMO_CKPT.exists():
        sh(
            f"python -m src.scripts.train "
            f"--data_root {processed} "
            f"--epochs {args.epochs} "
            f"--batch_size {args.batch_size}"
        )
        latest = latest_ckpt(Path("checkpoints"))
        if latest is None:
            sys.exit("❌ training produced no checkpoint")
        latest.rename(DEMO_CKPT)
        print(f"[✓] checkpoint copied → {DEMO_CKPT}")
    else:
        print(f"[skip] using existing checkpoint {DEMO_CKPT}")

    # 3) extract embeddings ----------------------------------------------
    if not EMB_FILE.exists():
        sh(
            f"python -m src.scripts.extract_features "
            f"--data_dir {processed/'train'} "
            f"--checkpoint {DEMO_CKPT} "
            f"--output {EMB_FILE}"
        )
    else:
        print(f"[skip] {EMB_FILE} already exists")

    # 4) build FAISS indices ---------------------------------------------
    if not any(IDX_DIR.glob("*.index")):
        data = np.load(EMB_FILE, allow_pickle=True)
        idxs = build_class_indices(data["features"], data["labels"])
        for lab, idx in idxs.items():
            faiss.write_index(idx, str(IDX_DIR / f"{lab}.index"))
            paths = [p for p, l in zip(data["paths"], data["labels"]) if l == lab]
            pickle.dump(paths, open(IDX_DIR / f"{lab}.paths", "wb"))
        print(f"[✓] wrote {len(idxs)} indices → {IDX_DIR}")
    else:
        print(f"[skip] indices already present")

    # 5) live query -------------------------------------------------------
    store = ClassIndexStore(dim=2048)
    for idx_file in IDX_DIR.glob("*.index"):
        lab = idx_file.stem
        wrapper = _IndexWithPaths(2048)  # use the helper
        wrapper.index = faiss.read_index(str(idx_file))
        wrapper.paths = pickle.load(open(idx_file.with_suffix(".paths"), "rb"))
        store.class_indices[lab] = wrapper
    class_names = sorted(store.class_indices.keys())

    model = create_resnet50_model(num_classes=len(class_names))
    ckpt_dict = torch.load(DEMO_CKPT, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt_dict, strict=False)

    results, pred = retrieve_similar_images(
        img_path=args.query,
        model=model,
        index_store=store,
        class_names=class_names,
        top_k=args.top_k,
        device=torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    )

    print(f"\nQuery: {args.query}")
    print(f"Predicted class: {pred}\nTop‑{args.top_k} matches:")
    for rank, (p, d) in enumerate(results, 1):
        dist = float(np.asarray(d).flat[0])
        print(f" {rank:>2}. {p}  (squared‑L2 {dist:.4f})")

    show_gallery(args.query, results, k=args.top_k)


if __name__ == "__main__":
    main()
