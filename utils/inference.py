"""
Inference Utility
=================
Handles prediction using the saved model + preprocessing pipeline.
Ensures IDENTICAL preprocessing during inference as during training.
"""

import numpy as np
import pandas as pd
import joblib
import shap
from tensorflow.keras.models import load_model
from utils.evaluator import get_risk_level

MODELS_DIR = "models"

# ── Lazy-loaded singletons ──
_model = None
_scaler = None
_label_encoders = None
_feature_names = None
_multi_cat_cols = None
_shap_explainer = None

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']
BINARY_COLS = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']


def load_artifacts():
    """Load all saved model artifacts once."""
    global _model, _scaler, _label_encoders, _feature_names, _multi_cat_cols
    if _model is None:
        _model = load_model(f"{MODELS_DIR}/churn_ann_model.h5")
        _scaler = joblib.load(f"{MODELS_DIR}/scaler.pkl")
        _label_encoders = joblib.load(f"{MODELS_DIR}/label_encoders.pkl")
        _feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")
        _multi_cat_cols = joblib.load(f"{MODELS_DIR}/multi_cat_cols.pkl")
    return _model, _scaler, _label_encoders, _feature_names, _multi_cat_cols


def preprocess_single(customer: dict) -> np.ndarray:
    """
    Preprocess a single customer dict for inference.
    Must mirror training preprocessing exactly.
    """
    model, scaler, label_encoders, feature_names, multi_cat_cols = load_artifacts()

    df = pd.DataFrame([customer])
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df['TotalCharges'].fillna(df['TotalCharges'].median() if not df['TotalCharges'].isnull().all() else 0, inplace=True)

    # Binary encode
    for col in BINARY_COLS:
        if col in df.columns and col in label_encoders:
            le = label_encoders[col]
            df[col] = le.transform(df[col])

    # One-hot encode
    df = pd.get_dummies(df, columns=multi_cat_cols, drop_first=False)

    # Align columns with training feature names
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    df = df[feature_names]

    # Scale numerical
    num_cols_present = [c for c in NUMERICAL_COLS if c in df.columns]
    df[num_cols_present] = scaler.transform(df[num_cols_present])

    return df.values.astype(np.float32)


def predict_customer(customer: dict) -> dict:
    """
    Full prediction pipeline for one customer.
    Returns probability, risk level, explanation, recommended action.
    """
    model, *_ = load_artifacts()
    features = preprocess_single(customer)
    prob = float(model.predict(features, verbose=0).flatten()[0])
    risk = get_risk_level(prob)

    # Simple rule-based explanation (SHAP not called per-request for speed)
    explanations = []
    if customer.get('Contract') == 'Month-to-month':
        explanations.append("Month-to-month contract increases churn risk")
    if float(customer.get('MonthlyCharges', 0)) > 70:
        explanations.append("High monthly charges above average threshold")
    if int(customer.get('tenure', 12)) < 12:
        explanations.append("Low tenure — customer is relatively new")
    if customer.get('InternetService') == 'Fiber optic':
        explanations.append("Fiber optic customers show higher churn rates")
    if not explanations:
        explanations.append("No major churn risk factors detected")

    return {
        'churn_probability': round(prob, 4),
        'churn_percentage': f"{prob * 100:.1f}%",
        'risk_level': risk['level'],
        'risk_color': risk['color'],
        'explanations': explanations,
        'recommended_action': risk['action']
    }


def predict_batch(df: pd.DataFrame) -> pd.DataFrame:
    """Batch prediction for CSV uploads."""
    model, scaler, label_encoders, feature_names, multi_cat_cols = load_artifacts()

    results = []
    for _, row in df.iterrows():
        try:
            result = predict_customer(row.to_dict())
            results.append(result)
        except Exception as e:
            results.append({'churn_probability': None, 'risk_level': 'ERROR',
                            'explanations': [str(e)], 'recommended_action': 'N/A'})

    return pd.DataFrame(results)
