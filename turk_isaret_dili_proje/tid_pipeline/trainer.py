"""
=============================================================
TRAINING & EVALUATION PIPELINE — 2 EL (156-dim)
=============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score
)
import tensorflow as tf
from tensorflow import keras

from config import cfg
from model import build_mlp, compile_model, get_callbacks


def train_mode(data_tuple, label_encoder,
               mode_name: str, model_save_path: str,
               epochs: int = None):
    if data_tuple is None or label_encoder is None:
        print(f"  [{mode_name}] Veri bulunamadı, atlanıyor.")
        return None, None

    if epochs is None:
        epochs = cfg.EPOCHS

    X_train, X_val, X_test, y_train, y_val, y_test = data_tuple
    num_classes = len(label_encoder.classes_)
    input_dim = X_train.shape[1]  # 156 (2 el) otomatik algılanır

    print(f"\n{'=' * 55}")
    print(f"  [{mode_name}] Model Eğitimi Başlıyor (2 EL)")
    print(f"{'=' * 55}")
    print(f"  Input dim    : {input_dim} (2 el × 78)")
    print(f"  Sınıf sayısı : {num_classes}")
    print(f"  Train        : {len(X_train)}")
    print(f"  Val          : {len(X_val)}")
    print(f"  Test         : {len(X_test)}")

    class_weights = _compute_class_weights(y_train, num_classes)

    model = build_mlp(input_dim, num_classes)
    model = compile_model(model)
    model.summary(print_fn=lambda s: print(f"    {s}"))

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=cfg.BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(model_save_path),
        verbose=1
    )

    print(f"\n  [{mode_name}] Test değerlendirmesi:")
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"    Test Accuracy : {test_acc:.4f}  ({test_acc*100:.2f}%)")
    print(f"    Test Loss     : {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
    print(f"    Weighted F1   : {f1:.4f}")

    return model, history


def evaluate_model(model, data_tuple, label_encoder,
                   mode_name: str, output_dir: str = None):
    if model is None or data_tuple is None:
        return {}

    if output_dir is None:
        output_dir = cfg.OUTPUT_PATH

    X_train, X_val, X_test, y_train, y_val, y_test = data_tuple
    class_names = list(label_encoder.classes_)

    y_pred_proba = model.predict(X_test, verbose=0)
    y_pred = np.argmax(y_pred_proba, axis=1)

    print(f"\n{'=' * 55}")
    print(f"  [{mode_name}] Detaylı Değerlendirme (2 EL)")
    print(f"{'=' * 55}")

    report = classification_report(
        y_test, y_pred,
        target_names=class_names,
        zero_division=0,
        output_dict=False
    )
    print(report)

    cm = confusion_matrix(y_test, y_pred)
    _plot_confusion_matrix(cm, class_names, mode_name, output_dir)

    csv_path = f"{output_dir}/{mode_name.lower()}_model_2hands_history.csv"
    if os.path.exists(csv_path):
        _plot_training_history(csv_path, mode_name, output_dir)

    per_class_acc = cm.diagonal() / (cm.sum(axis=1) + 1e-9)
    worst_idx = np.argsort(per_class_acc)[:5]
    print(f"\n  En düşük doğruluklu 5 sınıf:")
    for i in worst_idx:
        print(f"    {class_names[i]:25s} → {per_class_acc[i]:.2%}")

    acc = np.mean(y_pred == y_test)
    f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)

    return {
        'mode': mode_name,
        'test_accuracy': float(acc),
        'weighted_f1': float(f1),
        'per_class_accuracy': dict(zip(class_names, per_class_acc.tolist())),
    }


def _compute_class_weights(y_train: np.ndarray, num_classes: int) -> dict:
    n = len(y_train)
    weights = {}
    for cls in range(num_classes):
        n_cls = np.sum(y_train == cls)
        if n_cls > 0:
            weights[cls] = n / (num_classes * n_cls)
        else:
            weights[cls] = 1.0
    return weights


def _plot_confusion_matrix(cm, class_names, mode_name, output_dir):
    n = len(class_names)
    fig_size = max(10, n * 0.6)
    plt.figure(figsize=(fig_size, fig_size * 0.8))

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    sns.heatmap(
        cm_norm,
        annot=(n <= 30),
        fmt='.0%' if n <= 30 else '',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        vmin=0, vmax=1
    )
    plt.title(f'{mode_name} — Confusion Matrix (2 EL)', fontsize=14)
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin Edilen Sınıf')
    plt.xticks(rotation=45, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()

    save_path = f"{output_dir}/{mode_name.lower()}_confusion_matrix.png"
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\n  Confusion matrix kaydedildi: {save_path}")


def _plot_training_history(csv_path, mode_name, output_dir):
    hist = pd.read_csv(csv_path)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4))

    ax1.plot(hist['accuracy'], label='Train', color='steelblue')
    ax1.plot(hist['val_accuracy'], label='Validation', color='coral')
    ax1.set_title(f'{mode_name} — Accuracy (2 EL)')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(hist['loss'], label='Train', color='steelblue')
    ax2.plot(hist['val_loss'], label='Validation', color='coral')
    ax2.set_title(f'{mode_name} — Loss (2 EL)')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    ax2.grid(alpha=0.3)

    plt.tight_layout()
    save_path = f"{output_dir}/{mode_name.lower()}_training_history.png"
    plt.savefig(save_path, dpi=150)
    plt.close()
    print(f"  Eğitim grafiği kaydedildi: {save_path}")