"""
Step 1 & 2: Data Collection, Understanding & EDA
=================================================
Dataset: Telco Customer Churn (IBM/Kaggle)
- Contains demographics, subscription, usage, contract, payment, churn label
- Industry-standard benchmark for churn prediction
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
import os
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────
DATA_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
RAW_DATA_PATH = "data/raw/telco_churn.csv"
FIGURES_DIR = "data/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs("data/raw", exist_ok=True)

# ─────────────────────────────────────────────
# STEP 1: LOAD DATA
# ─────────────────────────────────────────────
def load_data():
    """Download and load the Telco Customer Churn dataset."""
    print("=" * 60)
    print("STEP 1: DATA COLLECTION")
    print("=" * 60)
    print("\n📥 Loading Telco Customer Churn Dataset...")
    print("Source: IBM / Kaggle")
    print("""
WHY THIS DATASET:
  ✅ Real-world telecom industry data
  ✅ 7,043 customers — sufficient for ANN training
  ✅ Rich features: demographics, usage, contract, billing
  ✅ Binary target: Churn (Yes/No)
  ✅ Industry benchmark — widely used in academia & industry
  ✅ Business relevance: telecom churn costs billions annually
    """)

    try:
        df = pd.read_csv(DATA_URL)
        df.to_csv(RAW_DATA_PATH, index=False)
        print(f"✅ Dataset saved to {RAW_DATA_PATH}")
    except Exception:
        print("⚠️  Could not download. Using local file if available.")
        df = pd.read_csv(RAW_DATA_PATH)

    return df


# ─────────────────────────────────────────────
# STEP 2: EDA
# ─────────────────────────────────────────────
def perform_eda(df: pd.DataFrame):
    print("\n" + "=" * 60)
    print("STEP 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 60)

    # ── Basic Info ──
    print(f"\n📊 Dataset Shape: {df.shape}")
    print(f"📌 Columns: {list(df.columns)}")
    print("\n🔍 Data Types:")
    print(df.dtypes.value_counts())

    # ── Target Variable ──
    print("\n🎯 TARGET VARIABLE: Churn")
    print(df['Churn'].value_counts())
    churn_rate = (df['Churn'] == 'Yes').mean() * 100
    print(f"⚠️  Churn Rate: {churn_rate:.1f}%")

    # ── Missing Values ──
    print("\n🔎 Missing Values:")
    # TotalCharges has hidden whitespace
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    missing = df.isnull().sum()
    print(missing[missing > 0])

    # ── Class Imbalance ──
    print(f"\n⚖️  Class Balance:")
    print(f"  No Churn : {(df['Churn']=='No').sum()} ({(df['Churn']=='No').mean()*100:.1f}%)")
    print(f"  Churned  : {(df['Churn']=='Yes').sum()} ({(df['Churn']=='Yes').mean()*100:.1f}%)")
    print("  → Moderate imbalance: will handle with class_weight or SMOTE")

    # ── Numerical Stats ──
    num_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    print(f"\n📈 Numerical Features: {num_cols}")
    print(df[num_cols].describe().round(2))

    generate_visualizations(df)
    return df


def generate_visualizations(df: pd.DataFrame):
    """Generate and save key EDA visualizations."""
    print("\n📊 Generating Visualizations...")

    df_vis = df.copy()
    df_vis['TotalCharges'] = pd.to_numeric(df_vis['TotalCharges'], errors='coerce')
    df_vis['ChurnBinary'] = (df_vis['Churn'] == 'Yes').astype(int)

    fig, axes = plt.subplots(2, 3, figsize=(18, 11))
    fig.suptitle('Telco Customer Churn — EDA Overview', fontsize=16, fontweight='bold')

    # 1. Churn Distribution
    churn_counts = df_vis['Churn'].value_counts()
    axes[0, 0].pie(churn_counts, labels=churn_counts.index, autopct='%1.1f%%',
                   colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[0, 0].set_title('Churn Distribution')

    # 2. Tenure vs Churn
    df_vis.boxplot(column='tenure', by='Churn', ax=axes[0, 1],
                   boxprops=dict(color='steelblue'))
    axes[0, 1].set_title('Tenure vs Churn')
    axes[0, 1].set_xlabel('Churn')
    axes[0, 1].set_ylabel('Tenure (months)')

    # 3. Monthly Charges vs Churn
    df_vis.boxplot(column='MonthlyCharges', by='Churn', ax=axes[0, 2],
                   boxprops=dict(color='coral'))
    axes[0, 2].set_title('Monthly Charges vs Churn')
    axes[0, 2].set_xlabel('Churn')

    # 4. Contract Type vs Churn
    contract_churn = df_vis.groupby('Contract')['ChurnBinary'].mean().reset_index()
    axes[1, 0].bar(contract_churn['Contract'], contract_churn['ChurnBinary'],
                   color=['#3498db', '#2ecc71', '#e74c3c'])
    axes[1, 0].set_title('Churn Rate by Contract Type')
    axes[1, 0].set_ylabel('Churn Rate')
    axes[1, 0].set_xlabel('Contract')
    axes[1, 0].tick_params(axis='x', rotation=15)

    # 5. Internet Service vs Churn
    internet_churn = df_vis.groupby('InternetService')['ChurnBinary'].mean().reset_index()
    axes[1, 1].bar(internet_churn['InternetService'], internet_churn['ChurnBinary'],
                   color=['#9b59b6', '#f39c12', '#1abc9c'])
    axes[1, 1].set_title('Churn Rate by Internet Service')
    axes[1, 1].set_ylabel('Churn Rate')

    # 6. Correlation Heatmap (numerical)
    num_df = df_vis[['tenure', 'MonthlyCharges', 'TotalCharges', 'ChurnBinary']].dropna()
    corr = num_df.corr()
    sns.heatmap(corr, annot=True, fmt='.2f', ax=axes[1, 2], cmap='coolwarm',
                linewidths=0.5)
    axes[1, 2].set_title('Correlation Heatmap')

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/eda_overview.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ EDA plot saved to {FIGURES_DIR}/eda_overview.png")

    # ── Business Insights ──
    print("""
╔══════════════════════════════════════════════════════════╗
║              KEY BUSINESS INSIGHTS (EDA)                 ║
╠══════════════════════════════════════════════════════════╣
║ 1. TENURE: Churned customers have significantly LOWER    ║
║    tenure → new customers need early engagement          ║
║                                                          ║
║ 2. MONTHLY CHARGES: Churned customers pay MORE monthly   ║
║    → High-cost customers feel value mismatch             ║
║                                                          ║
║ 3. CONTRACT TYPE: Month-to-month contracts have ~42%     ║
║    churn vs ~11% (1yr) and ~3% (2yr) → lock-in matters   ║
║                                                          ║
║ 4. INTERNET SERVICE: Fiber optic users churn most        ║
║    → Quality perception or cost issues with fiber        ║
║                                                          ║
║ 5. CLASS IMBALANCE: ~26% churn — needs handling via      ║
║    class weights or SMOTE for fair model training        ║
╚══════════════════════════════════════════════════════════╝
    """)


if __name__ == "__main__":
    df = load_data()
    df = perform_eda(df)
    print("\n✅ Step 1 & 2 Complete.")
