"""
Flatten every sub‑folder in data/raw/ into the raw root and
rebuild data/manifests/labels.csv. Safe to run multiple times.
"""
import csv, shutil, uuid
from pathlib import Path

RAW      = Path("data/raw")
MANIFEST = Path("data/raw/labels.csv")
MANIFEST.parent.mkdir(parents=True, exist_ok=True)

rows = []

for item in RAW.iterdir():
    if item.is_dir():
        label = item.name
        for img in item.rglob("*.*"):
            if img.is_dir():
                continue
            # ensure unique filename in flat space
            new_name = f"{label}_{uuid.uuid4().hex}{img.suffix}"
            dest = RAW / new_name
            shutil.move(str(img), dest)
            rows.append((new_name, label))
        # try to remove the now‑empty folder (ignore errors)
        try:
            item.rmdir()
        except OSError:
            pass
    elif item.is_file():
        # already flat → try to recover its label from prefix, else UNLABELED
        prefix = "_".join(item.name.split("_")[:2])
        rows.append((item.name, prefix if prefix else "UNLABELED"))

# Write (or overwrite) manifest
with MANIFEST.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["filename", "doc_type"])
    writer.writerows(rows)

print(f"✅ Flattened structure; manifest now has {len(rows)} entries.")
