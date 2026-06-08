# CSE 144 Final Project : 100-Class Image Classification

**UCSC CSE 144 Applied Machine Learning | Spring 2026**

![Leaderboard](leaderboard.png)

---

## Overview

This project tackles a 100-class image classification task with only **10 training images per class**. Fine-tuning a neural network from scratch overfits severely in this few-shot regime. Instead, we use **OpenAI CLIP** as a frozen feature extractor and train a lightweight Logistic Regression classifier on top — achieving **88.18% test accuracy** on Kaggle.

---

## Method

**Model:** Ensemble of CLIP ViT-B/16 and CLIP ViT-L/14 (both frozen)

**Pipeline:**
1. Load each training image and pass it through both CLIP models
2. L2-normalize the output embeddings (512-dim from ViT-B/16, 768-dim from ViT-L/14)
3. Concatenate into a single 1280-dim feature vector per image
4. Train a Logistic Regression classifier (C=10, lbfgs, max_iter=3000) on all training features
5. At inference time, repeat feature extraction on test images and predict

**Why CLIP?** CLIP is pretrained on 400 million image-text pairs and produces semantically rich features that transfer extremely well even with very few labeled examples per class.

---

## Results

| Model | Kaggle Score |
|---|---|
| ResNet18 fine-tuned | 16.4% |
| CLIP ViT-B/32 | 69.1% |
| CLIP ViT-B/16 | 80.0% |
| CLIP ViT-L/14 | 87.3% |
| **Ensemble ViT-B/16 + ViT-L/14** | **88.2%** |

---

## Requirements

```bash
pip install openai-clip scikit-learn torch torchvision pillow numpy pandas
```

Python 3.8+ required.

---

## Dataset Setup

Download the dataset from Kaggle and organize it as follows:

```
data/
  train/
    0/
      0.jpg
      1.jpg
      ...
    1/
    ...
    99/
  test/
    0.jpg
    1.jpg
    ...
    1035.jpg
  sample_submission.csv
models/
```

---

## Training

```bash
python train.py
```

This will:
- Download CLIP model weights automatically (first run only, ~1.2GB total)
- Extract features from all 1079 training images using both ViT-B/16 and ViT-L/14
- Train a Logistic Regression classifier on the concatenated features
- Save the trained classifier to `models/clip_classifier.pkl`

Expected output:
```
Classes: 100
Loading ViT-B/16...
Loading ViT-L/14...
Combined features: (1079, 1280)
Training classifier...
Training accuracy: 0.9981
Saved classifier to models/clip_classifier.pkl
```

---

## Inference

```bash
python predict.py
```

This will:
- Load the trained classifier from `models/clip_classifier.pkl`
- Extract CLIP features for all 1036 test images
- Save predictions to `submission.csv`

---

## Pretrained Model Weights

Download the trained classifier (`clip_classifier.pkl`) from Google Drive:

[Google Drive link](https://drive.google.com/drive/folders/1Temn8XREEFuC5Ju6wvAUdkaAC2fFwDLF?usp=sharing)

Place it in the `models/` directory before running `predict.py`.

---

## Project Structure

```
├── train.py              # Feature extraction + classifier training
├── predict.py            # Inference + submission generation
├── models/
│   └── clip_classifier.pkl   # Trained classifier weights
├── data/
│   ├── train/            # Training images (100 classes × ~10 images)
│   └── test/             # Test images (1036 images)
├── submission.csv        # Kaggle submission file
└── README.md
```
