#!/usr/bin/env python3
"""
Simple dataset builder: run this script to:
  • read <project_root>/data/raw/labels.csv
  • split 70/15/15 stratified by doc_type
  • copy into <project_root>/data/processed/{train,val,test}/...
  • emit _manifest.json in each split and split.json at root

Usage:
    python src/datasets/dataset_builder.py
"""

from __future__ import annotations
import json
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd
from src.config import RAW_DATA_PATH, PROCESSED_DATA_PATH, TRAIN_VAL_SPLIT
from sklearn.model_selection import train_test_split

from torchvision.datasets import ImageFolder
from torchvision import transforms as tvt

# ─── CONFIG (hard‑coded paths) ──────────────────────────────────────────────────
RAW_DIR = RAW_DATA_PATH
OUT_DIR = PROCESSED_DATA_PATH
TRAIN_RATIO = TRAIN_VAL_SPLIT
VAL_RATIO = (1 - TRAIN_RATIO) / 2
SEED = 1337
LINK_INSTEAD_OF_COPY = False  # set True to hard-link instead of copy
# ────────────────────────────────────────────────────────────────────────────────

def split_dataframe(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    strat_key = df["doc_type"]
    df_train, df_tmp = train_test_split(
        df, train_size=TRAIN_RATIO, stratify=strat_key, random_state=SEED
    )
    test_ratio = 1 - TRAIN_RATIO - VAL_RATIO
    df_val, df_test = train_test_split(
        df_tmp,
        train_size=VAL_RATIO / (VAL_RATIO + test_ratio),
        stratify=strat_key.loc[df_tmp.index],
        random_state=SEED,
    )
    return {"train": df_train, "val": df_val, "test": df_test}

def materialize_split(name: str, subset: pd.DataFrame) -> None:
    manifest: list[dict[str, str]] = []
    for row in subset.itertuples():
        dest = OUT_DIR / name / row.doc_type
        dest.mkdir(parents=True, exist_ok=True)
        src = RAW_DIR / row.filename
        dst = dest / row.filename
        if not dst.exists():
            if LINK_INSTEAD_OF_COPY:
                os.link(src, dst)
            else:
                shutil.copy2(src, dst)
        manifest.append({"path": str(dst), "doc_type": row.doc_type})

    with open(OUT_DIR / name / "_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

def build_datasets(
        data_root: str | Path = PROCESSED_DATA_PATH,
        train_tf=None,
        val_tf=None,
        test_tf=None,
):
    """
    Return (train_ds, val_ds, test_ds) torch‑vision ImageFolder datasets
    that point at the processed splits.
    """
    data_root = Path(data_root)
    train_ds = ImageFolder(data_root / "train", transform=train_tf or tvt.ToTensor())
    val_ds = ImageFolder(data_root / "val", transform=val_tf or tvt.ToTensor())
    test_ds = ImageFolder(data_root / "test", transform=test_tf or tvt.ToTensor())
    return train_ds, val_ds, test_ds

def main() -> None:
    # ensure output dirs
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # load labels
    labels_path = RAW_DIR / "labels.csv"
    if not labels_path.exists():
        raise FileNotFoundError(f"Cannot find labels.csv at {labels_path}")
    df = pd.read_csv(labels_path, dtype=str)

    # check raw files
    missing = [fn for fn in df["filename"] if not (RAW_DIR / fn).exists()]
    if missing:
        raise FileNotFoundError(f"Missing files in {RAW_DIR}: {missing[:5]}...")

    # split and save
    splits = split_dataframe(df)
    for name, subset in splits.items():
        materialize_split(name, subset)

    # save summary
    summary = {k: len(v) for k, v in splits.items()}
    summary["created"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    with open(OUT_DIR / "split.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("✅ Done:", summary)


if __name__ == "__main__":
    main()
