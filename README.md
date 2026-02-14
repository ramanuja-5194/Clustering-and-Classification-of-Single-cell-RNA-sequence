# Clustering-and-Classification-of-Single-cell-RNA-sequence
## 📌 Project Overview
This project performs an end-to-end computational analysis of single-cell RNA sequencing (scRNA-seq) data using the **PBMC 3k dataset** (Peripheral Blood Mononuclear Cells). 

The workflow demonstrates a standard bioinformatics pipeline combining:
1.  **Unsupervised Learning:** To identify distinct cell types without prior labels.
2.  **Supervised Learning:** To train a classifier that can predict cell identity based on gene expression profiles.

## 🚀 Key Features
* **Preprocessing:** Quality control (QC) to filter dead cells, mitochondrial filtering, and normalization.
* **Dimensionality Reduction:** Principal Component Analysis (PCA) and t-SNE for visualization.
* **Clustering:** K-Means clustering to group cells by transcriptional similarity.
* **Marker Gene Discovery:** Statistical identification of genes that define each cluster.
* **Classification:** Support Vector Machine (SVM) classifier to predict cell types with high accuracy (~98%).

## 🛠️ Technologies Used
* **Scanpy:** Efficient single-cell data analysis and preprocessing.
* **Scikit-Learn:** Machine learning algorithms (PCA, K-Means, SVM).
* **Pandas & NumPy:** Data manipulation.
* **Matplotlib & Seaborn:** Visualization of clusters and confusion matrices.
