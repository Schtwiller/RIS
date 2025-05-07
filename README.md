# 🔍 Reverse Image Search (Identity‑Docs) &nbsp;—&nbsp; **Working README @ Milestone #1**


> **Status**:  
> ✔ Repo scaffolded  
> ✔ Flat-bucket data scheme  
> ✔ `dataset_builder.py`  
> ✔ Pre-processing & transforms (`transforms.py` + `IDDataset`)  
> ✔ Unit tests for dataset & transforms  
> ✔ Starter training script (`train.py`)  
> **Next up**: Baseline embedding model & metric-learning losses _(see [Roadmap](#roadmap))_

---

## 1 ▪ Why we’re building this

Identity documents (passports, driver’s licenses, national IDs) share rigid visual structure yet vary wildly in wear, capture devices, and lighting.  
We need a **reverse‑image‑search pipeline** that:

* embeds an incoming photo into a compact vector
* finds near‑duplicates / fraud attempts in a corpus of hundreds‑of‑thousands of docs
* scales from a developer laptop to a GPU‑backed cloud service

---

## 2 ▪ What’s working right now

| ✅ Component | Notes |
|--------------|-------|
| **Repo structure** | `src/`, `data/`, `tests/`, etc. (see [Directory Map](#directory-map)). |
| **Flat data bucket** | All images live in `data/raw/` with a single `labels.csv` (filename, doc_type, country…) as the truth table. |
| **Dataset builder** | `python src/datasets/dataset_builder.py` → creates `data/processed/{train,val,test}` with stratified 70/15/15 split + manifest files. |

No environment variables or CLI flags needed—paths and split ratios are hard‑coded for zero‑friction.

---

## 3 ▪ Directory Map

```markdown
reverse_image_search/
├─ data/
│ ├─ raw/ ← flat bucket of originals + labels.csv
│ └─ processed/ ← auto‑generated splits (immutable)
├─ src/
│ ├─ datasets/
│ │ ├─ dataset_builder.py ← implemented
│ │ └─ (transforms.py, init.py) ← next step
│ └─ … (models/, inference/, etc. stubbed)
└─ README.md ← you are here

```

## 4 ▪ Data Workflow (so far)

1. **Drop images**  
   Put your `.jpg/.png` files in `data/raw/` and add rows to `labels.csv`:

   ```csv
   filename,doc_type
   IMG_001.jpg,passport
   ABC123.png,driver_license

2. **Build Splits**
   ```bash
   python src/datasets/dataset_builder.py
   ```
   Creates:
   ```markdown
   data/processed/
   ├─ train/.../_manifest.json
   ├─ val/.../_manifest.json
   ├─ test/.../_manifest.json
   └─ split.json    # {"train":1234,"val":264,"test":266,…}
   
   ```
## 5 ▪ Usage Cheatsheet (so far)
```bash
# clone & install deps (conda / venv)
git clone reverse-image-search-model
cd reverse_image_search
pip install -r requirements.txt   # pandas, scikit‑learn, etc.

# build dataset
python src/datasets/dataset_builder.py

```

✅ Done: {'train': 700, 'val': 150, 'test': 150, 'created': '2025‑05‑07T15:42:00'}

## 6 ▪ Roadmap

| Step                                          | Target                                                      | Target Date | Status      |
|-----------------------------------------------|-------------------------------------------------------------|-------------|-------------| 
| **Prototype Dataset Collection and Cleaning** | 300+ images                                                 | 5/7/2025    | ✅ Done      |
| **Pre-processing pipeline**                   | `transforms.py` + `IDDataset` dataloader                    | 5/14/2025   | In Progress |
| **Baseline embedding model**                  | ResNet-50 + GeM head + Triplet loss                         | 5/21/2025   | next        |
| **Training loop (metric-learning)**           | PyTorch Lightning / bare PyTorch + TensorBoard/W\&B logging | 5/28/2025   | pending     |
| **Offline FAISS index builder**               | IVF-PQ snapshot + search CLI                                | 6/4/2025    | pending     |
| **Extractor micro-service**                   | FastAPI + Torch-scripted embedding API                      | pending     | pending     |
| **Vector-DB (Qdrant) & live API**             | `/search` endpoint wired to online index                    | pending     | pending     |

