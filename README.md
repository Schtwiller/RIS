# Document Reverse Image Search — Prototype README
*(last updated: 2025-05-01)*

## 1 . What is this?
A minimal, end‑to‑end pipeline that:

1. **Ingests** document images through a FastAPI micro‑service (`src/ingestion`).
2. **Cleans & splits** them into `train/ val/ test` using `scripts/make_dataset.py`.
3. **Trains** a CNN‑based classifier / embedder (PyTorch + Lightning).
4. **Builds** Approximate Nearest‑Neighbour (ANN) indexes (FAISS/HNSW) per document type for similarity search.
5. **Retrains & re‑indexes** automatically via a Prefect flow (`src/pipeline`).

Everything is modular—swap backbones or ANN libraries by editing a single YAML config.

---

## 2 . Repo Layout (high‑level)

```text
doc-retrieval/
├── data/
│   ├── raw/          # uploads land here via FastAPI
│   ├── processed/    # train|val|test images
│   └── indexes/      # *.faiss + *.pkl per doc type
├── notebooks/        # quick EDA / experiments
├── scripts/          # one‑off helpers (dataset, search, migrate_official)
├── src/
│   ├── configs/          # Hydra YAMLs
│   ├── ingestion/        # FastAPI app
│   ├── datasets/         # PyTorch Dataset + transforms
│   ├── models/           # backbone + embedder
│   ├── train/            # Lightning training CLI
│   ├── retrieve/         # index builder & search helpers
│   └── pipeline/         # Prefect flow for periodic retrain
├── tests/            # pytest suites
├── environment.yml   # conda env spec
└── README.md         # this file
```

---

## 3 . Quick Start

1. **Create the conda environment**
   ```bash
   conda env create -f environment.yml
   conda activate doc-retrieval
   ```

2. **Start the ingestion API** (new uploads land in `data/raw/`)
   ```bash
   uvicorn src.ingestion.listener:app --host 0.0.0.0 --port 8000
   # POST an image:
   curl -X POST -F "img=@my_id.jpg" http://localhost:8000/upload
   ```

3. **Build prototype dataset & splits**
   ```bash
   python scripts/make_dataset.py --src data/raw --out data/processed --size 512
   ```

4. **Train the model**
   ```bash
   python -m src.train.train_cli train.yaml   # creates checkpoints/cnn_cls.pt
   ```

5. **Build ANN index (per document type)**
   ```bash
   python scripts/build_index.py --ckpt checkpoints/doc_embedder.pt                                      --img-dir data/processed/train/passport                                      --out data/indexes/passport.faiss
   ```

6. **Run a similarity‑search demo**
   ```bash
   python scripts/search_similar.py --image query.jpg --k 5
   ```

7. *(Optional)* **Schedule periodic retraining**
   ```bash
   prefect deployment apply src/pipeline/retrain_flow.py
   ```

---

## 4 . Configuration

- **Hydra** YAMLs live in `src/configs/`; change backbone (`resnet50`, `vit_b_16`, …), embedding dim, batch size, or ANN type here.
- Thresholds for the Prefect flow (`THRESHOLD` new images before retrain) are in `src/pipeline/retrain_flow.py`.

---

## 5 . Testing

```bash
pytest -q
```
- `tests/test_ingestion.py`    FastAPI upload works & file stored  
- `tests/test_dataset.py`      Dataset/augmentations output correct shapes  
- `tests/test_search.py`       Index round‑trip returns self‑match at rank 1

---

## 6 . Troubleshooting

| Symptom | Fix |
|---------|-----|
| _CUDA out of memory_ during training | Reduce `batch_size` in `train.yaml` or switch to CPU (`device=cpu`). |
| _Index build slow_ | Use FAISS IVF+PQ (`metric: ip`, `index_type: ivf_pq`) in config. |
| _Uploads 400 error_ | Ensure `Content‑Type: multipart/form-data` and field name `img`. |

---

## 7 . Next Steps

1. Fine‑tune on real, cleared data once available.  
2. Add OCR/text embeddings for hybrid search.  
3. Deploy container (`Dockerfile`) behind a secure gateway.
