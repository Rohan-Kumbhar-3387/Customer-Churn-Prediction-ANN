# 🔮 Customer Churn Prediction — ANN

> **Industry-Level Final Year Project** | Deep Learning | Explainable AI | FastAPI | Streamlit

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange)](https://tensorflow.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109-green)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.31-red)](https://streamlit.io)
[![SHAP](https://img.shields.io/badge/SHAP-0.44-purple)](https://shap.readthedocs.io)

---

## 📌 Project Overview

A production-grade customer churn prediction system using a **Deep Artificial Neural Network (ANN)**, featuring:

- 🧠 **ANN Model** — 4-layer deep network with BatchNorm + Dropout
- 🔍 **Explainable AI** — SHAP DeepExplainer for per-prediction transparency
- 🚀 **REST API** — FastAPI with Swagger docs + batch prediction
- 📊 **Business Dashboard** — Streamlit with KPIs, segmentation, what-if analysis
- 💼 **Business Impact** — CLV analysis, retention strategy, revenue at risk

---

## 📊 Model Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | ~82.4% |
| ROC-AUC   | ~86.2% |
| Recall    | ~79.6% |
| Precision | ~73.8% |
| F1-Score  | ~76.6% |

---

## 🏗️ Project Structure

```
churn_prediction/
├── data/
│   ├── raw/           # Original dataset (Telco Churn CSV)
│   ├── processed/     # Train/val/test numpy arrays
│   └── figures/       # EDA & training plots
│
├── models/            # Saved artifacts
│   ├── churn_ann_model.h5
│   ├── scaler.pkl
│   ├── label_encoders.pkl
│   ├── feature_names.pkl
│   └── evaluation_metrics.pkl
│
├── utils/             # Core Python modules
│   ├── data_loader.py    # Step 1 & 2: EDA
│   ├── preprocessor.py   # Step 3 & 4: Preprocessing & split
│   ├── model_trainer.py  # Step 5 & 6: Architecture & training
│   ├── evaluator.py      # Step 7 & 8: Metrics & SHAP
│   └── inference.py      # Step 9: Prediction utility
│
├── api/
│   └── main.py        # Step 10: FastAPI REST API
│
├── dashboard/
│   └── app.py         # Step 11: Streamlit dashboard
│
├── train_pipeline.py  # Master training script
├── requirements.txt
└── README.md
```

---

## ⚡ Quick Start

### 1. Clone & Install
```bash
git clone https://github.com/yourusername/churn-prediction-ann
cd churn-prediction-ann
pip install -r requirements.txt
```

### 2. Train the Model
```bash
python train_pipeline.py
```
This runs the complete pipeline: data download → EDA → preprocessing → ANN training → evaluation → SHAP.

### 3. Start the API
```bash
uvicorn api.main:app --reload --port 8000
```
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

### 4. Launch the Dashboard
```bash
streamlit run dashboard/app.py
```
Opens at http://localhost:8501

---

## 🔌 API Usage

### Single Prediction
```bash
curl -X POST "http://localhost:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gender": "Female",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "tenure": 12,
    "PhoneService": "Yes",
    "MultipleLines": "No",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "No",
    "StreamingMovies": "No",
    "Contract": "Month-to-month",
    "PaperlessBilling": "Yes",
    "PaymentMethod": "Electronic check",
    "MonthlyCharges": 79.85,
    "TotalCharges": 958.2
  }'
```

**Response:**
```json
{
  "churn_probability": 0.7823,
  "churn_percentage": "78.2%",
  "risk_level": "HIGH",
  "explanations": [
    "Month-to-month contract increases churn risk",
    "High monthly charges above average threshold",
    "Low tenure — customer is relatively new"
  ],
  "recommended_action": "🚨 Immediate intervention: Offer 20% discount + personal outreach"
}
```

### What-If Analysis
```bash
curl -X POST "http://localhost:8000/what-if?monthly_charge_reduction=15" \
  -H "Content-Type: application/json" \
  -d '{ ... customer data ... }'
```

### Batch Prediction
```bash
curl -X POST "http://localhost:8000/predict/batch" \
  -F "file=@customers.csv"
```

---

## 🧠 ANN Architecture

```
Input (n_features)
    │
    ├─ Dense(256) → BatchNorm → ReLU → Dropout(0.4)
    │
    ├─ Dense(128) → BatchNorm → ReLU → Dropout(0.3)
    │
    ├─ Dense(64)  → BatchNorm → ReLU → Dropout(0.2)
    │
    ├─ Dense(32)  → ReLU → Dropout(0.1)
    │
    └─ Dense(1)   → Sigmoid → P(Churn)

Loss      : Binary Crossentropy
Optimizer : Adam (lr=0.001)
Callbacks : EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
```

---

## 📊 Dashboard Features

| Feature | Description |
|---------|-------------|
| Executive KPIs | Churn rate, revenue at risk, retention rate |
| Risk Segmentation | HIGH/MEDIUM/LOW with interactive charts |
| Customer Lookup | Search by ID, view profile + gauge chart |
| AI Prediction Form | Real-time single customer prediction |
| SHAP Explanations | Global & local feature importance |
| Batch Analysis | CSV upload + bulk prediction |
| What-If Simulator | Intervention impact waterfall chart |
| CLV Analysis | Revenue-weighted churn risk view |

---

## 📁 Dataset

**IBM Telco Customer Churn** (public domain)
- 7,043 customers, 21 features
- Target: `Churn` (Yes/No)
- Source: [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn) / [IBM GitHub](https://github.com/IBM/telco-customer-churn-on-icp4d)

---

## 🏆 Resume Description

> **Customer Churn Prediction System** | Python, TensorFlow, FastAPI, Streamlit, SHAP  
> Built an end-to-end production ML pipeline predicting telecom customer churn with 82.4% accuracy and 0.862 AUC using a 4-layer deep ANN with batch normalization and dropout regularization. Implemented Explainable AI using SHAP DeepExplainer for business-interpretable predictions. Deployed a RESTful API (FastAPI) with Swagger documentation and an interactive Streamlit business intelligence dashboard featuring KPI monitoring, customer segmentation, CLV analysis, batch prediction, and what-if analysis simulation.

---

## 📄 License

MIT License — Free to use for academic and commercial projects.

---

*Built with ❤️ as an Industry-Level Final Year Project*
