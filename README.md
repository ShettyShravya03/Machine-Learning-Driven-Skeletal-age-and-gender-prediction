# ML-Driven Skeletal Age & Gender Prediction using Cervical Vertebral Maturation (CVM)

> **Best Project of the Year — Commendation Prize · EXPRO 2025-26, NMAM Institute of Technology**

[![Python](https://img.shields.io/badge/Python-3.10-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org)
[![Flask](https://img.shields.io/badge/Flask-000000?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com)
[![React](https://img.shields.io/badge/React-61DAFB?style=flat-square&logo=react&logoColor=black)](https://react.dev)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=flat-square)](LICENSE)

---

## Overview

A full end-to-end computer vision pipeline for **orthodontic skeletal maturity assessment** from lateral cephalometric X-rays. The system automates a clinical workflow that traditionally requires ~15 minutes of manual measurement by an orthodontist, reducing it to **under 2 minutes** with comparable accuracy.

Given a cervical spine X-ray, the pipeline:
1. Segments the C2–C4 vertebrae using an **Attention U-Net**
2. Extracts **79 morphological features** from the segmented masks
3. Uses **SHAP** to select the most predictive features
4. Predicts **skeletal age** (XGBoost regression) and **biological sex** (Random Forest classification)
5. Serves results via a **Flask REST API** to a **React frontend** with live segmentation overlay

---

## Results

| Task | Model | Metric | Score |
|------|-------|--------|-------|
| Vertebrae segmentation | Attention U-Net | Dice coefficient | **0.94** |
| Skeletal age prediction | XGBoost | MAE | **5.87 years** |
| Gender classification | Random Forest | AUC / Accuracy | **0.912 / 84%** |

- Dataset: **1,294 clinical lateral cephalometric X-rays**
- Features extracted: 79 morphological → SHAP-selected 23 (age) and 15 (gender)

---

## Architecture

```
Lateral Cephalometric X-ray
          │
          ▼
  Attention U-Net
  (C2, C3, C4 vertebrae segmentation)
  Dice score: 0.94
          │
          ▼
  79 Morphological Features
  (height ratios, concavity indices,
   vertebral body shapes per C2–C4)
          │
     ┌────┴────┐
     ▼         ▼
  SHAP        SHAP
  23 features 15 features
  (age)       (gender)
     │         │
     ▼         ▼
  XGBoost   Random Forest
  Regression Classification
  MAE: 5.87  AUC: 0.912
  years      84% accuracy
     │         │
     └────┬────┘
          ▼
  Flask REST API (/predict, /health)
          │
          ▼
  React Frontend
  (drag-and-drop upload · live segmentation
   overlay · gender confidence score)
```

---

## Tech Stack

| Layer | Tools |
|-------|-------|
| Segmentation | PyTorch · Attention U-Net |
| Feature selection | SHAP |
| Age prediction | XGBoost |
| Gender classification | scikit-learn (Random Forest) |
| Backend | Flask · Python |
| Frontend | React · Vite · Tailwind CSS |
| Utilities | OpenCV · NumPy · Pandas · Matplotlib |

---

## Installation

### Backend

```bash
git clone https://github.com/ShettyShravya03/Machine-Learning-Driven-Skeletal-age-and-gender-prediction.git
cd Machine-Learning-Driven-Skeletal-age-and-gender-prediction/backend

pip install -r requirements.txt
```

**Key dependencies:** `torch`, `torchvision`, `xgboost`, `scikit-learn`, `shap`, `flask`, `opencv-python`, `numpy`, `pandas`

### Frontend

```bash
cd frontend
npm install
```

---

## Usage

### 1. Start the Flask backend

```bash
cd backend
python app.py
# Server runs at http://localhost:5000
```

### 2. Start the React frontend

```bash
cd frontend
npm run dev
# App runs at http://localhost:5173
```

### 3. Use the API directly

**Predict age and gender from an X-ray:**

```bash
curl -X POST http://localhost:5000/predict \
  -F "image=@/path/to/xray.jpg"
```

**Sample response:**
```json
{
  "skeletal_age": 14.3,
  "gender": "Female",
  "gender_confidence": 0.89,
  "segmentation_overlay": "<base64_image>",
  "shap_features_used": 23
}
```

**Health check:**
```bash
curl http://localhost:5000/health
```

---

## Pipeline Details

### Stage 1 — Segmentation (Attention U-Net)

The Attention U-Net is trained to segment the **C2, C3, and C4 cervical vertebrae** from lateral cephalometric X-rays. Attention gates suppress irrelevant background activations and focus on the vertebral body boundaries.

- **Input:** Grayscale X-ray (resized to 256×256)
- **Output:** Binary segmentation masks per vertebra
- **Dice score:** 0.94 on held-out test set

### Stage 2 — Feature Extraction

79 morphological features are computed from the segmentation masks:
- Vertebral body height and width ratios
- Concavity index of the inferior border of C2
- Ratio of anterior to posterior vertebral height
- Shape descriptors per C3 and C4 (rectangular → square → trapezoidal maturation stages)

### Stage 3 — SHAP Feature Selection

SHAP (SHapley Additive exPlanations) is used to rank feature importance and select the top features for each task, reducing overfitting and improving generalisability:
- **Age model:** Top 23 features selected
- **Gender model:** Top 15 features selected

### Stage 4 — Prediction Models

| Model | Task | Why |
|-------|------|-----|
| XGBoost | Age regression | Handles non-linear feature interactions; robust to outliers in clinical data |
| Random Forest | Gender classification | Strong ensemble method; interpretable feature importances |

---

## File Structure

```
.
├── backend/
│   ├── app.py                  # Flask REST API entry point
│   ├── requirements.txt
│   ├── model/
│   │   ├── attention_unet.py   # U-Net architecture with attention gates
│   │   ├── age_model.py        # XGBoost age regression pipeline
│   │   └── gender_model.py     # Random Forest gender classification pipeline
│   ├── utils/
│   │   ├── feature_extraction.py  # 79 morphological features
│   │   ├── shap_selection.py      # SHAP-based feature selector
│   │   └── preprocessing.py       # Image normalisation, resizing
│   └── checkpoints/
│       └── best_unet.pt        # Trained Attention U-Net weights
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── UploadZone.jsx      # Drag-and-drop X-ray upload
│   │   │   ├── SegmentationView.jsx # Live overlay display
│   │   │   └── ResultCard.jsx       # Age + gender output card
│   │   └── api/
│   │       └── predict.js
│   ├── package.json
│   └── vite.config.js
│
├── notebooks/
│   ├── 01_segmentation_training.ipynb
│   ├── 02_feature_extraction.ipynb
│   ├── 03_shap_selection.ipynb
│   └── 04_model_training_evaluation.ipynb
│
└── README.md
```

---

## Dataset

This project was trained and evaluated on a dataset of **1,294 lateral cephalometric X-rays** with ground-truth skeletal age and gender labels. The dataset was collected in accordance with ethical guidelines for clinical research.

> **Note:** The dataset is not included in this repository due to patient privacy constraints. To reproduce results, a similarly annotated dataset of lateral cephalometric X-rays with CVM stage labels is required.

---

## Awards

**Best Project of the Year — Commendation Prize**
Issued by EXPRO 2025-26, NMAM Institute of Technology · April 2026
*Awarded at the Final Year Students' Project Exhibition & Competition.*

---

## Contributing

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-feature`
3. Commit your changes: `git commit -m "add: your feature description"`
4. Push to the branch: `git push origin feature/your-feature`
5. Open a pull request

---

## License

MIT © 2026 Shravya S Shetty

---

## Topics

`computer-vision` `pytorch` `attention-unet` `xgboost` `shap` `explainable-ai` `flask` `react` `medical-imaging` `skeletal-age-prediction` `python` `deep-learning` `orthodontics` `cervical-vertebral-maturation`
