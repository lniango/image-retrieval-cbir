# 🔍 Content-Based Image Retrieval (CBIR)

A Python project for image retrieval by content, combining local descriptors (SIFT) and deep CNN features (VGG16/VGG19), with FLANN indexing for efficient nearest-neighbor search.

Built as part of a university project in **Multimedia Analysis & Indexing**.

---

## 📌 Features

- **Local descriptors** — SIFT keypoint detection and description (OpenCV 4)
- **Global descriptors** — Deep features extracted from VGG16 / VGG19 (Keras / TensorFlow)
- **Efficient indexing** — FLANN linear and KD-Tree indexes for fast kNN search
- **Image matching** — Brute-force and FLANN-based descriptor matching with visual output
- **Retrieval pipeline** — Full query search with ranked results and precision-recall evaluation
- **Evaluation metrics** — Average Precision (AP), mean AP (mAP), Precision-Recall curves

---

## 🗂️ Project Structure

```
.
├── feature_description.py   # SIFT feature extraction demo
├── histogram.py             # Grayscale and color histogram computation
├── filtering.py             # Image filtering (blur, Gaussian, median)
├── flann.py                 # FLANN indexing and kNN search demo
├── matcher_ocv3.py          # SIFT descriptor matching between two images
├── draw_matches.py          # Utility to visualize keypoint matches
├── db_indexing_p3.py        # Database indexing with SIFT + FLANN
├── db_indexing_p3FFF.py     # Database indexing with VGG16 global features
├── query_search_p3.py       # Single-query search + PR curve evaluation
├── multi_query_search.py    # Multi-query search + global mAP evaluation
├── vgg16.py                 # VGG16 feature extraction demo
└── vgg19.py                 # VGG19 feature extraction demo
```

---

## ⚙️ Installation

### With Conda (recommended)

```bash
# 1. Clone the repository
git clone https://github.com/<your-username>/cbir-sift-cnn.git
cd cbir-sift-cnn

# 2. Create the environment from the YAML file
conda env create -f environment.yml

# 3. Activate the environment
conda activate cbir-env
```

### Update the environment (if environment.yml has changed)

```bash
conda env update -f environment.yml --prune
```

### Remove the environment

```bash
conda deactivate
conda env remove -n cbir-env
```

> Tested with Python 3.8, OpenCV 4.10, TensorFlow 2.13

---

## 🚀 Usage

### 1. Index a database (SIFT)
```bash
python db_indexing_p3.py -d <database_name>
```

### 2. Query search (single image)
```bash
python query_search_p3.py -d <database_name> -q <image_name> -t LINEAR -r <nb_relevant>
```

### 3. Multi-query evaluation (mAP)
```bash
python multi_query_search.py -d <database_name> -t LINEAR -r <nb_relevant>
```

---

## 📊 Evaluation

The retrieval system is evaluated on standard image databases (COREL, NISTER, Copydays) using:
- **Precision / Recall** per query
- **Average Precision (AP)**
- **Mean Average Precision (mAP)** over all queries

---

## 👥 Authors

Developed by students at **ESIR** (École Supérieure d'Ingénieurs de Rennes) — Semester 9, Multimedia Analysis & Indexing course.
