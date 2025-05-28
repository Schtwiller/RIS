# Reverse Image Search for Identity Documents

## Overview

This project implements a **reverse image search pipeline for identity documents** (e.g. passports, driver’s licenses). It finds the most visually similar documents by mapping images to embeddings via a fine-tuned ResNet-50 and performing efficient vector search with FAISS, restricted to the most likely classes.

### Key Features
- **Data Preprocessing:** Automates splitting raw images (flat folder + CSV) into stratified train/val/test directories with JSON manifests.
- **Fine-Tuning:** Uses a pretrained ResNet-50 backbone, with configurable heads, augmentations, and early stopping.
- **Feature Extraction:** Strips the final layer to extract 2048-dim embeddings for any set of images.
- **Per-Class FAISS Indexing:** Builds one `IndexFlatL2` per class (plus leftovers) for fast exact L2 search.
- **Two-Stage Retrieval:** Classify then search within top-N predicted classes with fallback and threshold logic.
- **End-to-End Demo:** `src/demo.py` ties data building, training, indexing, and querying into one command, including a visual gallery of results.

## Repository Structure

```
reverse_image_search/
├── artifacts/                # Generated outputs (checkpoints, embeddings, indices)
├── data/
│   ├── raw/                  # Input images + labels.csv
│   └── processed/            # Stratified train/val/test folders
├── src/
│   ├── datasets/             # Data loading and Albumentations transforms
│   ├── features/             # Feature extraction utilities
│   ├── indexing/             # FAISS index management (ClassIndexStore)
│   ├── models/               # ResNet-50 creation logic
│   ├── retrieval/            # High-level search combining model+indices
│   ├── scripts/              # CLI tools: train.py, extract_features.py, infer.py
│   └── demo.py               # One-command end-to-end pipeline
├── tests/                    # Unit tests for each component
└── README.md                 # This file
```

## Quickstart

1. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```

2. **Prepare Data**  
   - Place all images in `data/raw/`  
   - Create `data/raw/labels.csv` with `filename,doc_type` columns  
   - Run:  
     ```bash
     python src/datasets/dataset_builder.py
     ```

3. **Train Model**  
   ```bash
   python -m src.scripts.train        --data_root data/processed        --epochs 10        --batch_size 32
   ```

4. **Extract Features**  
   ```bash
   python -m src.scripts.extract_features        --data_dir data/processed/train        --checkpoint artifacts/checkpoints/resnet50_demo.pt        --output artifacts/embeddings/train_embeddings.npz
   ```

5. **Build Indices**  
   ```bash
   python - <<'PY'
   import numpy as np, faiss, pickle, pathlib
   from src.indexing.faiss_index import build_class_indices
   data = np.load("artifacts/embeddings/train_embeddings.npz", allow_pickle=True)
   idxs = build_class_indices(data["features"], data["labels"], data["paths"])
   out = pathlib.Path("artifacts/indices"); out.mkdir(exist_ok=True)
   for lab, idx in idxs.items():
       faiss.write_index(idx, str(out/f"{lab}.index"))
       pickle.dump([p for p,l in zip(data["paths"], data["labels"]) if l==lab],
                   open(out/f"{lab}.paths","wb"))
   PY
   ```

6. **Run Demo Query**  
   ```bash
   python -m src.demo --query data/processed/test/passport/img_001.jpg --top_k 5
   ```

## How It Works

At a high level, this project transforms images of identity documents into compact numerical representations (embeddings) and then finds the most similar images by comparing these embeddings. Here’s the step-by-step flow:

1. **Data Preparation**  
   - You start with a flat folder of JPEG/PNG images and a simple CSV (`filename,doc_type`).  
   - A helper script organizes this into `train/`, `val/`, and `test/` directories, each containing subfolders named by document type (e.g. `passport/`, `driver_license/`).  
   - This split ensures your model sees a representative sample of each class during training and evaluation.

2. **Model Training**  
   - We use **ResNet-50**, a proven convolutional neural network, as the backbone.  
   - The network is fine-tuned to classify document types (passports, driver’s licenses, etc.) using your training images.  
   - Early-stopping and validation checks prevent over-fitting: if validation performance doesn’t improve for a few epochs, training stops automatically.

3. **Embedding Extraction**  
   - After training, we remove the final classification layer.  
   - Each image is passed through the network, and the 2048-dimensional output just before the classifier is recorded—this is your **embedding**.  
   - Similar images (even if not identical) tend to have embeddings that are close together in this 2048-dimensional space.

4. **FAISS Indexing**  
   - We build one **exact L2 index** (FAISS IndexFlatL2) per document type.  
   - Each index holds all embeddings for that class. This narrows the search space, making queries faster and more accurate.

5. **Query & Retrieval**  
   - When you query with a new image:
     1. The image is preprocessed (resized, normalized) exactly as during training.
     2. The trained model predicts its document type (e.g. “passport”) and also produces its embedding.
     3. We first search the FAISS index for that predicted class. If the closest match is sufficiently close, we return those top-K images.
     4. If the top match is too far (beyond a threshold), we fall back to search the next most likely class, and so on.  
   - This two-step approach (classification + similarity search) balances speed and accuracy: you only compare against the most likely subset of images.

6. **Visualization**  
   - Finally, a small gallery is displayed: your query on the left and the top-K retrieved images on the right.  
   - You instantly see which documents in your database are most similar, making it easy to spot duplicates or near-duplicates.

This modular pipeline—data split, training, embedding extraction, indexing, and retrieval—lets you plug in your own images and get meaningful similarity results with minimal effort. Feel free to experiment with different backbones, thresholds, or index types as your needs evolve!```

## Configuration

All constants (image size, augmentation settings, indexing thresholds) live in `src/config.py`. Adjust as needed:
```python
IMAGE_SIZE = 512
TRAINING = {"batch_size": 32, "epochs": 10, "lr": 1e-3, "patience": 3}
INDEXING = {"graduating_threshold": 10, "match_threshold": 1.2, "top_k_images": 5}
```

## Testing

Run all unit tests with:
```bash
pytest -q
```

## Recommendations & Roadmap

- **Flexible Backbones:** Refactor to support multiple CNN architectures.
- **Unified Data API:** Consolidate dataset loading into one class.
- **Approximate Indexing:** Add IVF/PQ or HNSW for large-scale deployments.
- **Single-Pass Retrieval:** Modify model to output logits & embeddings in one forward.
- **Logging & CI:** Integrate Python `logging`, TensorBoard/W&B, and GitHub Actions.

---

*This README and its roadmap aim to make the project robust, maintainable, and production-ready.*