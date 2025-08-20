# ✍️ Handwriting Line Segmentation with U-Net

[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)  
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2.0+-red.svg)](https://pytorch.org/)  
[![License](https://img.shields.io/badge/License-Apache_2.0-green.svg)](LICENSE)

A modular deep learning pipeline for **handwritten document line segmentation** using the **U-Net architecture**.  
This repository supports OCR preprocessing, handwriting recognition research, and automated document layout analysis.

---

## 📂 Project Structure

---

```
handwriting-line-segmentation/
├── data_loader.py     # Dataset loading & preprocessing
├── model.py           # U-Net model definition
├── train.py           # Training loop & validation
├── evaluate.py        # Evaluation & metrics
├── utils.py           # Helper functions (plotting, augmentation, etc.)
├── main.py            # Entry point to run training/testing
├── requirements.txt   # Python dependencies
└── README.md          # Project documentation

```
### ✨ Features
* Dataset loader with preprocessing and augmentation (flip, crop, resize).
* U-Net model for precise handwritten line segmentation.
* Training pipeline with checkpointing, early stopping, and configurable hyperparameters.
* Evaluation metrics: Dice, IoU, precision, recall.
* Visualization of predictions overlayed on original images.
* Modular design for easy experimentation.


---

## ⚙️ Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/handwriting-line-segmentation.git](https://github.com/your-username/handwriting-line-segmentation.git)
    cd handwriting-line-segmentation
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```
3.  **(Optional)** Ensure GPU support by installing PyTorch with CUDA: [PyTorch Installation](https://pytorch.org/get-started/locally/)

---

## 🚀 Usage

**Train the model**
```
python main.py --mode train \
               --epochs 50 \
               --batch-size 8 \
               --lr 1e-4 \
               --save-dir checkpoints/
```
---
```
python main.py --mode eval \
               --checkpoint checkpoints/best_model.pth
```
---
Visualize predictions

---

Bash
```
python main.py --mode visualize \
            --checkpoint checkpoints/best_model.pth
```
---

📚 Datasets
You can train this project on:

IAM Handwriting Database

Kaggle Line Counter Dataset

Or your own dataset structured as:
```
data/
├── images/
│   ├── sample1.png
│   ├── sample2.png
└── masks/
    ├── sample1_mask.png
    ├── sample2_mask.png

```
📊 Example Results
Input Image	Ground Truth Mask	Predicted Mask

---

📝 License
This project is licensed under the Apache License 2.0.
