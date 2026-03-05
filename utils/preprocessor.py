"""
Step 3 & 4: Data Preprocessing & Splitting
===========================================
Handles: encoding, scaling, feature selection, imbalance, pipeline saving
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import warnings
warnings.filterwarnings('ignore')

MODELS_DIR = "models"
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)

# ─────────────────────────────────────────────
# COLUMNS CONFIGURATION
# ─────────────────────────────────────────────
DROP_COLS = ['customerID']          # Identifier — no predictive value, leakage risk

TARGET = 'Churn'

BINARY_COLS = [
    'gender', 'Partner', 'Dependents', 'PhoneService',
    'PaperlessBilling', 'Churn'
]

MULTI_CAT_COLS = [
    'MultipleLines', 'InternetService', 'OnlineSecurity',
    'OnlineBackup', 'DeviceProtection', 'TechSupport',
    'StreamingTV', 'StreamingMovies', 'Contract', 'PaymentMethod'
]

NUMERICAL_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']


# ─────────────────────────────────────────────
# STEP 3: PREPROCESSING
# ─────────────────────────────────────────────
def preprocess_data(df: pd.DataFrame, use_smote: bool = True):
    """Full preprocessing pipeline."""
    print("=" * 60)
    print("STEP 3: DATA PREPROCESSING")
    print("=" * 60)

    df = df.copy()

    # ── Fix TotalCharges (hidden non-numeric) ──
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')

    # ── Drop leakage / ID columns ──
    df.drop(columns=DROP_COLS, inplace=True)
    print(f"✅ Dropped columns: {DROP_COLS}")

    # ── Fill missing values ──
    missing_count = df['TotalCharges'].isnull().sum()
    df['TotalCharges'].fillna(df['TotalCharges'].median(), inplace=True)
    print(f"✅ Filled {missing_count} missing TotalCharges with median")

    # ── Encode binary categorical columns (Yes/No → 1/0) ──
    label_encoders = {}
    for col in BINARY_COLS:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col])
            label_encoders[col] = le
    print(f"✅ Label encoded binary columns: {BINARY_COLS}")

    # ── One-Hot Encode multi-category columns ──
    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=False)
    print(f"✅ One-hot encoded: {MULTI_CAT_COLS}")
    print(f"   → Shape after encoding: {df.shape}")

    # ── Separate features and target ──
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    # ── Scale numerical features ──
    scaler = StandardScaler()
    num_cols_present = [c for c in NUMERICAL_COLS if c in X.columns]
    X[num_cols_present] = scaler.fit_transform(X[num_cols_present])
    print(f"✅ StandardScaler applied to: {num_cols_present}")

    # ── Save feature names (critical for inference) ──
    feature_names = list(X.columns)
    joblib.dump(feature_names, f"{MODELS_DIR}/feature_names.pkl")
    joblib.dump(scaler, f"{MODELS_DIR}/scaler.pkl")
    joblib.dump(label_encoders, f"{MODELS_DIR}/label_encoders.pkl")
    joblib.dump(MULTI_CAT_COLS, f"{MODELS_DIR}/multi_cat_cols.pkl")
    print(f"✅ Saved scaler & encoders to {MODELS_DIR}/")

    # ── SMOTE for class imbalance ──
    print(f"\n⚖️  Class distribution before SMOTE:")
    print(f"   Churn=0: {(y==0).sum()} | Churn=1: {(y==1).sum()}")

    # We apply SMOTE AFTER train/test split (next step)
    # Here we just return X, y for splitting

    return X, y, scaler, label_encoders, feature_names


# ─────────────────────────────────────────────
# STEP 4: DATA SPLITTING
# ─────────────────────────────────────────────
def split_data(X: pd.DataFrame, y: pd.Series, use_smote: bool = True):
    """
    Split into train/val/test sets.
    Strategy: 70% train, 15% val, 15% test
    Reasoning:
      - 70% train: enough samples for ANN weight updates
      - 15% val: monitor overfitting during training
      - 15% test: unbiased final evaluation
      - Stratified split: preserves churn ratio in all splits
    """
    print("\n" + "=" * 60)
    print("STEP 4: DATA SPLITTING")
    print("=" * 60)

    # First split: 85% train+val, 15% test
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    # Second split: 70% train, 15% val (from 85% → ~82%/18%)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.176, random_state=42, stratify=y_temp
    )

    print(f"✅ Train   : {X_train.shape[0]} samples ({X_train.shape[0]/len(X)*100:.1f}%)")
    print(f"✅ Val     : {X_val.shape[0]} samples ({X_val.shape[0]/len(X)*100:.1f}%)")
    print(f"✅ Test    : {X_test.shape[0]} samples ({X_test.shape[0]/len(X)*100:.1f}%)")

    # ── Apply SMOTE only to TRAINING data ──
    if use_smote:
        smote = SMOTE(random_state=42)
        X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
        print(f"\n✅ SMOTE applied to training data:")
        print(f"   Before: {(y_train==0).sum()} No, {(y_train==1).sum()} Yes")
        print(f"   After : {(y_train_res==0).sum()} No, {(y_train_res==1).sum()} Yes")
    else:
        X_train_res, y_train_res = X_train, y_train
        print("\n⚠️  SMOTE skipped — using class_weight in model instead")

    # Save processed splits
    np.save("data/processed/X_train.npy", X_train_res.values)
    np.save("data/processed/y_train.npy", y_train_res.values)
    np.save("data/processed/X_val.npy", X_val.values)
    np.save("data/processed/y_val.npy", y_val.values)
    np.save("data/processed/X_test.npy", X_test.values)
    np.save("data/processed/y_test.npy", y_test.values)
    print(f"\n✅ Processed splits saved to data/processed/")

    return X_train_res, X_val, X_test, y_train_res, y_val, y_test


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from utils.data_loader import load_data

    df = load_data()
    X, y, scaler, le, features = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print("\n✅ Preprocessing & Splitting Complete.")
