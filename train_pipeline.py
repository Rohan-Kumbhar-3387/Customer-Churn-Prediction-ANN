"""
Master Pipeline: End-to-End Training
=====================================
Run this script to execute the complete pipeline from data loading to model saving.
Usage: python train_pipeline.py
"""

import os
import sys
import numpy as np

sys.path.append(os.path.dirname(os.path.abspath(__file__)))


def to_f32(arr):
    """Safely convert any array-like to float32 numpy array."""
    if hasattr(arr, 'values'):
        arr = arr.values
    return np.array(arr, dtype=np.float32)


def run_pipeline():
    print("\n" + "🚀 " * 20)
    print("   CUSTOMER CHURN PREDICTION — ANN TRAINING PIPELINE")
    print("🚀 " * 20 + "\n")

    # ── Step 1 & 2: Load & EDA ──
    print("\n📌 [1/6] Data Loading & EDA...")
    from utils.data_loader import load_data, perform_eda
    df = load_data()
    df = perform_eda(df)

    # ── Step 3 & 4: Preprocessing & Split ──
    print("\n📌 [2/6] Preprocessing & Splitting...")
    from utils.preprocessor import preprocess_data, split_data
    X, y, scaler, le, features = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # ── Step 5 & 6: Build & Train ANN ──
    print("\n📌 [3/6] Building & Training ANN...")
    from utils.model_trainer import train_model
    model, history = train_model(
        to_f32(X_train), to_f32(X_val),
        to_f32(y_train), to_f32(y_val)
    )

    # ── Step 7: Evaluate ──
    print("\n📌 [4/6] Evaluating Model...")
    from utils.evaluator import evaluate_model
    X_test_arr = to_f32(X_test)
    y_test_arr = to_f32(y_test)
    metrics, y_pred_prob = evaluate_model(model, X_test_arr, y_test_arr)

    # ── Step 8: SHAP Explainability ──
    print("\n📌 [5/6] Computing SHAP Explanations...")
    import joblib
    feature_names = joblib.load("models/feature_names.pkl")
    try:
        from utils.evaluator import explain_with_shap
        X_train_arr = to_f32(X_train)
        explain_with_shap(model, X_train_arr, X_test_arr, feature_names)
    except Exception as e:
        print(f"⚠️  SHAP explanation skipped: {e}")

    # ── Summary ──
    print("\n" + "✅ " * 20)
    print("\n🎉 PIPELINE COMPLETE! Summary:")
    print(f"  📊 Test Accuracy  : {metrics['Accuracy']:.4f}")
    print(f"  🎯 Test AUC       : {metrics['ROC-AUC']:.4f}")
    print(f"  📈 Recall (Churn) : {metrics['Recall']:.4f}")
    print(f"  💾 Model saved    : models/churn_ann_model.h5")
    print(f"  🔧 Scaler saved   : models/scaler.pkl")
    print(f"\n  🚀 Run API:       uvicorn api.main:app --reload --port 8000")
    print(f"  📊 Run Dashboard:  streamlit run dashboard/app.py\n")
    print("✅ " * 20)


if __name__ == "__main__":
    run_pipeline()