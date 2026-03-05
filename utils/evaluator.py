"""
Steps 7 & 8: Model Evaluation & Explainable AI (SHAP)
=======================================================
Complete evaluation suite + SHAP explanations
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import shap
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    roc_curve
)
from tensorflow.keras.models import load_model

MODELS_DIR = "models"
FIGURES_DIR = "data/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 7: MODEL EVALUATION
# ─────────────────────────────────────────────
def evaluate_model(model, X_test, y_test, threshold: float = 0.5):
    """Comprehensive evaluation with business interpretation."""
    print("=" * 60)
    print("STEP 7: MODEL EVALUATION")
    print("=" * 60)

    y_pred_prob = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_prob >= threshold).astype(int)

    # ── Metrics ──
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_prob)

    metrics = {
        'Accuracy': acc, 'Precision': prec,
        'Recall': rec, 'F1-Score': f1, 'ROC-AUC': auc
    }

    print("\n📊 CLASSIFICATION METRICS:")
    for name, val in metrics.items():
        status = "✅" if val >= 0.75 else "⚠️ "
        print(f"  {status} {name:12s}: {val:.4f}")

    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=['No Churn', 'Churn']))

    print("""
╔══════════════════════════════════════════════════════════╗
║         BUSINESS INTERPRETATION OF METRICS              ║
╠══════════════════════════════════════════════════════════╣
║ RECALL (most critical for churn):                        ║
║   High recall = catch most actual churners               ║
║   Missing a churner = lost revenue (false negative)      ║
║                                                          ║
║ PRECISION:                                               ║
║   High precision = fewer false alarms                    ║
║   False alarm = wasted retention spend on loyal customers║
║                                                          ║
║ ROC-AUC:                                                 ║
║   >0.80 = strong discriminative power                    ║
║   Used by executives as headline model quality metric    ║
╚══════════════════════════════════════════════════════════╝
    """)

    # ── Save metrics ──
    joblib.dump(metrics, f"{MODELS_DIR}/evaluation_metrics.pkl")

    # ── Plots ──
    plot_confusion_matrix(y_test, y_pred)
    plot_roc_curve(y_test, y_pred_prob)

    return metrics, y_pred_prob


def plot_confusion_matrix(y_test, y_pred):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['No Churn', 'Churn'],
                yticklabels=['No Churn', 'Churn'])
    ax.set_title('Confusion Matrix', fontsize=14, fontweight='bold')
    ax.set_ylabel('Actual')
    ax.set_xlabel('Predicted')

    # Annotate TN/FP/FN/TP
    labels = [['True Negative\n(Correctly retained)', 'False Positive\n(Wrongly flagged)'],
              ['False Negative\n(Missed churner!)', 'True Positive\n(Correctly flagged)']]
    for i in range(2):
        for j in range(2):
            ax.text(j + 0.5, i + 0.75, labels[i][j],
                    ha='center', va='center', fontsize=7, color='gray')

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/confusion_matrix.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Confusion matrix saved")


def plot_roc_curve(y_test, y_pred_prob):
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    auc = roc_auc_score(y_test, y_pred_prob)

    plt.figure(figsize=(7, 5))
    plt.plot(fpr, tpr, color='#3498db', lw=2, label=f'AUC = {auc:.4f}')
    plt.plot([0, 1], [0, 1], 'k--', lw=1, label='Random (AUC=0.5)')
    plt.fill_between(fpr, tpr, alpha=0.1, color='#3498db')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate (Recall)')
    plt.title('ROC Curve — Churn ANN', fontsize=13, fontweight='bold')
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/roc_curve.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ ROC curve saved")


# ─────────────────────────────────────────────
# STEP 8: EXPLAINABLE AI (SHAP)
# ─────────────────────────────────────────────
def explain_with_shap(model, X_train, X_test, feature_names, n_samples: int = 200):
    """
    SHAP Explanations:
    1. Global feature importance (what drives churn across all customers)
    2. Summary beeswarm plot
    3. Individual customer explanation
    """
    print("\n" + "=" * 60)
    print("STEP 8: EXPLAINABLE AI — SHAP")
    print("=" * 60)

    print("⏳ Computing SHAP values (this may take a minute)...")

    # Use DeepExplainer for neural networks
    # Use a background sample for efficiency
    background = X_train[np.random.choice(X_train.shape[0], 100, replace=False)]
    explainer = shap.DeepExplainer(model, background)

    # Compute SHAP for test sample
    X_explain = X_test[:n_samples]
    shap_values = explainer.shap_values(X_explain)

    # shap_values from DeepExplainer for binary is list[array]
    if isinstance(shap_values, list):
        sv = shap_values[0]
    else:
        sv = shap_values

    print(f"✅ SHAP values computed for {n_samples} test samples")

    # ── Global Feature Importance ──
    shap_df = pd.DataFrame(np.abs(sv), columns=feature_names)
    global_importance = shap_df.mean().sort_values(ascending=False).head(20)

    plt.figure(figsize=(10, 7))
    global_importance.plot(kind='barh', color='#3498db')
    plt.title('Top 20 Global Feature Importances (SHAP)', fontsize=13, fontweight='bold')
    plt.xlabel('Mean |SHAP Value|')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_global_importance.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ Global SHAP importance plot saved")

    # ── SHAP Summary Plot ──
    plt.figure(figsize=(12, 8))
    shap.summary_plot(
        sv, X_explain,
        feature_names=feature_names,
        max_display=15,
        show=False
    )
    plt.title('SHAP Summary Plot — Churn Drivers', fontsize=13, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/shap_summary.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("✅ SHAP summary plot saved")

    # ── Individual Customer Explanation ──
    print("\n🔍 INDIVIDUAL CUSTOMER EXPLANATION (Customer #0 from test set):")
    explain_single_customer(sv[0], feature_names, X_explain[0])

    # Save explainer and shap values
    joblib.dump({'shap_values': sv, 'feature_names': feature_names,
                 'X_explain': X_explain}, f"{MODELS_DIR}/shap_data.pkl")
    print(f"✅ SHAP data saved to {MODELS_DIR}/shap_data.pkl")

    return sv, explainer


def explain_single_customer(shap_vals, feature_names, customer_features):
    """Print human-readable explanation for one customer."""
    feat_shap = list(zip(feature_names, shap_vals, customer_features))
    feat_shap.sort(key=lambda x: abs(x[1]), reverse=True)

    print("\n┌─────────────────────────────────────────────────────────┐")
    print("│          CHURN RISK EXPLANATION — Customer               │")
    print("├─────────────────────────────────────────────────────────┤")
    print("│ Feature                    | SHAP Value  | Effect        │")
    print("├─────────────────────────────────────────────────────────┤")
    for feat, sv, val in feat_shap[:10]:
        direction = "↑ INCREASES risk" if sv > 0 else "↓ decreases risk"
        print(f"│ {feat[:26]:26s} | {sv:+.4f}    | {direction:15s}│")
    print("└─────────────────────────────────────────────────────────┘")

    top_drivers = [f for f, s, _ in feat_shap[:3] if s > 0]
    print(f"\n📌 Primary Churn Drivers: {', '.join(top_drivers)}")
    print("📌 Recommended Action: Offer contract upgrade + billing review")


def get_risk_level(prob: float) -> dict:
    """Map churn probability to risk level + recommended action."""
    if prob >= 0.7:
        return {
            'level': 'HIGH',
            'color': '#e74c3c',
            'action': '🚨 Immediate intervention: Offer 20% discount + personal outreach',
            'priority': 1
        }
    elif prob >= 0.4:
        return {
            'level': 'MEDIUM',
            'color': '#f39c12',
            'action': '⚠️  Engagement campaign: Send loyalty rewards + service upgrade',
            'priority': 2
        }
    else:
        return {
            'level': 'LOW',
            'color': '#2ecc71',
            'action': '✅ Maintain relationship: Include in loyalty program',
            'priority': 3
        }


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from utils.data_loader import load_data
    from utils.preprocessor import preprocess_data, split_data

    df = load_data()
    X, y, *_ = preprocess_data(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    feature_names = joblib.load(f"{MODELS_DIR}/feature_names.pkl")

    model = load_model(f"{MODELS_DIR}/churn_ann_model.h5")
    metrics, y_pred_prob = evaluate_model(model, X_test.values, y_test.values)
    explain_with_shap(model, X_train.values, X_test.values, feature_names)
    print("\n✅ Evaluation & Explainability Complete.")
