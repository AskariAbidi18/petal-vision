# petal-vision
# Petal-Vision 🌸
**Unsupervised Flower Image Analysis using Classical Digital Image Processing**

Petal-Vision is a mini-project for the Digital Image Processing course that explores unsupervised analysis of flower images collected around a college campus. The project focuses on classical image processing techniques such as segmentation, feature extraction, and clustering, without relying on labeled data or deep learning models.

---

## 📌 Objectives
- To preprocess and standardize raw flower images
- To segment flower regions from natural backgrounds
- To extract meaningful visual features (color, shape, texture)
- To perform unsupervised clustering based on visual similarity
- To analyze and visualize clustering results

---

## 📂 Project Structure
```
petal-vision/
├── data/
│ ├── raw/ # Original images
│ ├── standardized/ # Format-normalized images
│ ├── resized/ # Preprocessed images
│ └── segmented/ # Segmented flower regions
├── outputs/
│ ├── masks/ # Segmentation masks
│ ├── plots/ # Elbow & PCA plots
│ └── clusters/ # Cluster-wise image folders
├── results/
│ ├── features.csv
│ ├── features_final.csv
│ └── clustered_features.csv
├── src/ # Source code
├── report/ # Markdown report files
└── main.py # Pipeline orchestrator
```

---

## 🧠 Methodology Overview
1. Image standardization (format & naming)
2. Image preprocessing (resize, blur, color conversion)
3. Flower segmentation using GrabCut
4. Feature extraction (HSV color, shape, texture)
5. Feature preprocessing (cleaning & standardization)
6. Unsupervised clustering using K-Means
7. Visualization using PCA

---

## 📊 Evaluation
Since the dataset is unlabeled, internal clustering metrics were used:
- **Silhouette Score**
- **Davies–Bouldin Index**

These metrics indicate moderate but meaningful structure, which is expected for natural image data.

---

## 🚀 How to Run
```bash
python main.py

---

## 🧠 Methodology Overview
1. Image standardization (format & naming)
2. Image preprocessing (resize, blur, color conversion)
3. Flower segmentation using GrabCut
4. Feature extraction (HSV color, shape, texture)
5. Feature preprocessing (cleaning & standardization)
6. Unsupervised clustering using K-Means
7. Visualization using PCA

---

## 📊 Evaluation
Since the dataset is unlabeled, internal clustering metrics were used:
- **Silhouette Score**
- **Davies–Bouldin Index**

These metrics indicate moderate but meaningful structure, which is expected for natural image data.

---

## 🚀 How to Run
```bash
python main.py

The pipeline is incremental and safe to re-run. Only new images are processed.

## 📝 Notes

No supervised learning or deep learning models were used

The focus is on classical Digital Image Processing techniques

Results reflect real-world complexity rather than idealized datasets
