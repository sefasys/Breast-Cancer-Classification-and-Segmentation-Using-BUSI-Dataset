P.S.: It's for learning semantic segmentation. I learned how transfer learning mechanisms work. It's not the best version. There will be updates...

# Breast Ultrasound Classification & Segmentation

This repository contains a comprehensive deep learning pipeline for analyzing Breast Ultrasound Images (BUSI). It includes complete workflows for both image classification (Benign, Malignant, Normal) and semantic segmentation (Tumor vs. Background).

## 📌 Features

- **Classification Pipeline (EfficientNet-B0)**
  - Transfer learning from ImageNet.
  - 2-Stage training strategy (Classifier Head only $\rightarrow$ Fine-tuning Backbone).
  - Advanced anti-overfitting techniques: Dropout, Label Smoothing, Weight Decay, and Data Augmentation.
- **Segmentation Pipeline (U-Net)**
  - Custom PyTorch U-Net implementation from scratch.
  - Dice + BCE combined loss function for better boundary detection.
  - Generates binary masks for tumor localization.
- **Data Preparation**
  - Scripts to reorganize the raw BUSI dataset into a clean format.
  - Merging multiple overlapping masks using Bitwise OR.
  - Stratified dataset splitting (70% Train, 15% Val, 15% Test).
  - Offline image resizing and class balancing.

## 📂 Project Structure

```
├── src/
│   ├── train_classifier.py        # Classification training script
│   ├── train_segmentation.py      # Segmentation training script
│   └── data_prep/                 # Data restructuring & splitting scripts
├── notebooks/                     # Colab-ready Jupyter Notebooks
├── requirements.txt               # Project dependencies
└── README.md                      # Project documentation
```
*(Note: Datasets and result folders are ignored by git).*

## ⚙️ Installation

1. **Clone the repository:**
```bash
git clone https://github.com/yourusername/Breast-Ultrasound-Analysis.git
cd Breast-Ultrasound-Analysis
```

2. **Install dependencies:**
Make sure you have Python 3.8+ installed. It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

*(Note: If you have a dedicated GPU, make sure you install the PyTorch version that supports your CUDA version from [pytorch.org](https://pytorch.org/get-started/locally/)).*

## 💾 Dataset Preparation

The pipeline expects the **Breast Ultrasound Dataset (BUSI)**. 

1. Download the raw dataset and extract it. 
2. The prepared dataset structure should eventually look like this (created automatically by the `data_prep` scripts):
```
Dataset_BUSI_with_GT_Split/
├── train/
│   ├── benign/
│   │   ├── case_id_1/
│   │   │   ├── image.png
│   │   │   └── mask.png
│   ...
├── val/
└── test/
```

## 🚀 Usage

### 1. Classification
To train the classifier (EfficientNet-B0):
```bash
python src/train_classifier.py
```
This will run the 2-stage training process and save the metrics, plots, and best model weights into `results/classification/`.

### 2. Segmentation
To train the segmentation model (U-Net):
```bash
python src/train_segmentation.py
```
This script trains the model using Dice+BCE loss and evaluates it using Dice Score and IoU. Training curves and mask predictions will be saved to `results/segmentation/`.

## 📊 Results & Visualization

Both training scripts will automatically generate evaluation visualizations:
- **Training History:** Loss and Accuracy/Dice plots over epochs.
- **Confusion Matrix:** For classification performance.
- **Prediction Visualizations:** Side-by-side comparisons of ultrasound images, ground truth masks, and model predictions for segmentation.

---
*Created as part of an iterative learning journey in deep learning applied to medical imaging.*
