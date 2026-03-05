"""
Steps 5 & 6: ANN Architecture & Training
==========================================
Architecture: Deep ANN for tabular binary classification
"""

import numpy as np
import matplotlib.pyplot as plt
import os
import joblib
import warnings
warnings.filterwarnings('ignore')

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (Dense, Dropout, BatchNormalization, Input)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (EarlyStopping, ReduceLROnPlateau,
                                         ModelCheckpoint)
from tensorflow.keras.regularizers import l2

MODELS_DIR = "models"
FIGURES_DIR = "data/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)


# ─────────────────────────────────────────────
# STEP 5: ANN ARCHITECTURE
# ─────────────────────────────────────────────
def build_ann(input_dim: int) -> tf.keras.Model:
    """
    ANN Architecture for Tabular Churn Prediction
    ──────────────────────────────────────────────
    Input Layer  : input_dim features
    Hidden 1     : 256 units, ReLU, BatchNorm, Dropout(0.4)
    Hidden 2     : 128 units, ReLU, BatchNorm, Dropout(0.3)
    Hidden 3     : 64  units, ReLU, BatchNorm, Dropout(0.2)
    Hidden 4     : 32  units, ReLU, Dropout(0.1)
    Output       : 1 unit, Sigmoid → P(Churn)

    Design Decisions:
    - ReLU: avoids vanishing gradient, computationally efficient
    - BatchNormalization: stabilizes training, allows higher LR
    - Dropout: regularization, prevents overfitting on tabular data
    - L2 regularization: weight decay to penalize complexity
    - Sigmoid output: binary probability (0–1)
    - Binary crossentropy: standard for binary classification
    - Adam optimizer: adaptive LR, converges faster than SGD
    """
    model = Sequential([
        Input(shape=(input_dim,)),

        # Hidden Layer 1
        Dense(256, activation='relu', kernel_regularizer=l2(0.001),
              name='hidden_1'),
        BatchNormalization(),
        Dropout(0.4),

        # Hidden Layer 2
        Dense(128, activation='relu', kernel_regularizer=l2(0.001),
              name='hidden_2'),
        BatchNormalization(),
        Dropout(0.3),

        # Hidden Layer 3
        Dense(64, activation='relu', kernel_regularizer=l2(0.001),
              name='hidden_3'),
        BatchNormalization(),
        Dropout(0.2),

        # Hidden Layer 4
        Dense(32, activation='relu', name='hidden_4'),
        Dropout(0.1),

        # Output Layer
        Dense(1, activation='sigmoid', name='output')
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.AUC(name='auc'),
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall')]
    )

    print("\n" + "=" * 60)
    print("STEP 5: ANN ARCHITECTURE")
    print("=" * 60)
    model.summary()
    print(f"""
Architecture Explanation:
  Input Dim   : {input_dim} features
  Hidden 1    : 256 neurons (wide — capture complex patterns)
  Hidden 2    : 128 neurons (compress representations)
  Hidden 3    : 64  neurons (extract abstract features)
  Hidden 4    : 32  neurons (final feature distillation)
  Output      : 1 sigmoid → churn probability

  Loss        : Binary Crossentropy (standard for binary classification)
  Optimizer   : Adam lr=0.001 (adaptive, fast convergence)
  Batch Size  : 32 (good gradient estimate, memory efficient)
  Max Epochs  : 100 (EarlyStopping will prevent overfitting)
    """)

    return model


# ─────────────────────────────────────────────
# STEP 6: TRAINING PIPELINE
# ─────────────────────────────────────────────
def train_model(X_train, X_val, y_train, y_val):
    """Complete training pipeline with callbacks."""
    print("\n" + "=" * 60)
    print("STEP 6: MODEL TRAINING")
    print("=" * 60)

    input_dim = X_train.shape[1]
    model = build_ann(input_dim)

    # ── Callbacks ──
    callbacks = [
        EarlyStopping(
            monitor='val_auc',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=7,
            min_lr=1e-6,
            verbose=1
        ),
        ModelCheckpoint(
            filepath=f"{MODELS_DIR}/best_ann_checkpoint.h5",
            monitor='val_auc',
            save_best_only=True,
            mode='max',
            verbose=1
        )
    ]

    # ── Compute class weights (handle imbalance alternative to SMOTE) ──
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()
    class_weight = {0: 1.0, 1: neg / pos}
    print(f"\n📊 Class Weights: {{0: 1.0, 1: {neg/pos:.2f}}}")

    print("\n🚀 Training started...")
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # ── Save final model ──
    model.save(f"{MODELS_DIR}/churn_ann_model.h5")
    print(f"\n✅ Model saved to {MODELS_DIR}/churn_ann_model.h5")

    # ── Plot training curves ──
    plot_training_history(history)

    return model, history


def plot_training_history(history):
    """Plot training vs validation loss and AUC curves."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('ANN Training History', fontsize=14, fontweight='bold')

    # Loss
    axes[0].plot(history.history['loss'], label='Train Loss', color='#3498db')
    axes[0].plot(history.history['val_loss'], label='Val Loss', color='#e74c3c')
    axes[0].set_title('Training vs Validation Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Binary Crossentropy Loss')
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    # AUC
    axes[1].plot(history.history['auc'], label='Train AUC', color='#2ecc71')
    axes[1].plot(history.history['val_auc'], label='Val AUC', color='#e67e22')
    axes[1].set_title('Training vs Validation AUC')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('ROC-AUC')
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{FIGURES_DIR}/training_history.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ Training curves saved to {FIGURES_DIR}/training_history.png")


if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from utils.data_loader import load_data
    from utils.preprocessor import preprocess_data, split_data

    _df = load_data()
    _X, _y, *_ = preprocess_data(_df)
    _splits = split_data(_X, _y)
    _X_train, _X_val, _X_test, _y_train, _y_val, _y_test = _splits

    _model, _history = train_model(
        _X_train.values if hasattr(_X_train, 'values') else _X_train,
        _X_val.values if hasattr(_X_val, 'values') else _X_val,
        _y_train.values if hasattr(_y_train, 'values') else _y_train,
        _y_val.values if hasattr(_y_val, 'values') else _y_val,
    )
    print("\n✅ Training Complete.")