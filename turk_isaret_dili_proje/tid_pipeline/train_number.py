"""
=============================================================
SAYI MODELİ EĞİTİMİ
=============================================================
Kullanım:
    python train_number.py

Veri seti yapısı:
    digit-dataset/
    ├── train/
    │   ├── A0/  ← 0 rakamı
    │   ├── A1/  ← 1 rakamı
    │   └── ...
    ├── test/
    └── validation/

Çıktılar:
    output/number_model.h5
    output/number_model.tflite
    output/number_classes.json
=============================================================
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from feature_extraction import HandFeatureExtractor

# ─────────────────────────────────────────────────────────────
# AYARLAR — Sadece burası değiştirilir
# ─────────────────────────────────────────────────────────────

DATASET_PATH  = r"C:\Users\YunusEmre\Desktop\Bitirme Projesi\turk_isaret_dili_proje\tid_pipeline\data\dataset3"   # veri seti ana klasörü
OUTPUT_PATH   = "output"
MODEL_PATH    = "output/number_model.h5"
TFLITE_PATH   = "output/number_model.tflite"
CLASSES_PATH  = "output/number_classes.json"
CACHE_PATH    = "data/processed/numbers.pkl"

BATCH_SIZE    = 32
EPOCHS        = 120
DROPOUT       = 0.35
LR            = 0.001
PATIENCE      = 15
TEST_SIZE     = 0.10
VAL_SIZE      = 0.15
CONFIDENCE_THRESHOLD = 0.65

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
ARA_KLASORLER  = {'train', 'test', 'val', 'valid', 'validation', 'testing', 'training'}

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# VERİ YÜKLEME
# ─────────────────────────────────────────────────────────────

def klasor_adi_to_label(name: str) -> str:
    """
    A0 → "0", A1 → "1", ... A9 → "9"
    Eğer klasör adı zaten sayıysa olduğu gibi kullan.
    """
    cleaned = name.strip().upper()
    # A0, A1 ... A9 formatı
    if cleaned.startswith("A") and len(cleaned) == 2 and cleaned[1].isdigit():
        return cleaned[1]
    # 0, 1 ... 9 formatı
    if cleaned.isdigit():
        return cleaned
    # Diğer formatlar olduğu gibi
    return cleaned


def collect_classes(dataset_path: Path) -> dict:
    """Tüm split'lerden sınıfları topla."""
    label_to_images = {}

    for subdir in sorted(dataset_path.iterdir()):
        if not subdir.is_dir():
            continue

        if subdir.name.lower() in ARA_KLASORLER:
            # train/test/val klasörü içine gir
            for cls_dir in sorted(subdir.iterdir()):
                if not cls_dir.is_dir():
                    continue
                label = klasor_adi_to_label(cls_dir.name)
                imgs  = [f for f in cls_dir.iterdir()
                         if f.suffix.lower() in SUPPORTED_EXTS]
                if imgs:
                    label_to_images.setdefault(label, []).extend(imgs)
        else:
            # Doğrudan sınıf klasörü
            label = klasor_adi_to_label(subdir.name)
            imgs  = [f for f in subdir.iterdir()
                     if f.suffix.lower() in SUPPORTED_EXTS]
            if imgs:
                label_to_images[label] = imgs

    return label_to_images


def load_number_dataset(dataset_path: str) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  [Sayılar] Veri seti yükleniyor...")
    print("=" * 55)

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"  HATA: {dataset_path} bulunamadı!")
        return pd.DataFrame()

    label_to_images = collect_classes(dataset_path)
    if not label_to_images:
        print("  HATA: Hiç sınıf klasörü bulunamadı!")
        return pd.DataFrame()

    print(f"  Bulunan sınıflar : {sorted(label_to_images.keys())}")
    total = sum(len(v) for v in label_to_images.values())
    print(f"  Toplam görüntü   : {total}\n")

    extractor = HandFeatureExtractor()
    records   = []
    errors    = 0

    for label, image_files in sorted(label_to_images.items()):
        class_ok = 0
        for img_path in tqdm(image_files, desc=f"  Sayı {label:>2}", ncols=70):
            img = cv2.imread(str(img_path))
            if img is None:
                errors += 1
                continue
            features = extractor.extract_features(img, bbox=None)
            if features is not None:
                records.append({
                    'features': features,
                    'label': label,
                    'source': 'digit_dataset'
                })
                class_ok += 1
            else:
                errors += 1

        rate = class_ok / max(len(image_files), 1) * 100
        print(f"  Sayı {label:>2}: {class_ok}/{len(image_files)} ({rate:.0f}% el tespit)")

    if not records:
        print("  HATA: Hiç feature çıkarılamadı!")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"\n  Başarılı  : {len(df)}/{total}")
    print(f"  Sınıf dağılımı:\n{df['label'].value_counts().sort_index().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────────────────────

def build_model(input_dim: int, num_classes: int) -> keras.Model:
    inputs = keras.Input(shape=(input_dim,), name='landmarks')

    x = layers.Dense(256, use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(DROPOUT)(x)

    x = layers.Dense(128, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(DROPOUT)(x)

    x = layers.Dense(64, use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Dropout(DROPOUT * 0.5)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model   = keras.Model(inputs, outputs, name='TID_Number_MLP')
    return model


def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_PATH, monitor='val_accuracy',
            save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=PATIENCE // 2, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(
            MODEL_PATH.replace('.h5', '_history.csv'))
    ]


def compute_class_weights(y, num_classes):
    n = len(y)
    return {c: n / (num_classes * max(np.sum(y == c), 1))
            for c in range(num_classes)}


# ─────────────────────────────────────────────────────────────
# DEĞERLENDİRME
# ─────────────────────────────────────────────────────────────

def evaluate(model, X_test, y_test, class_names):
    print("\n" + "=" * 55)
    print("  [Sayılar] Değerlendirme")
    print("=" * 55)

    y_pred  = np.argmax(model.predict(X_test, verbose=0), axis=1)
    test_acc, test_loss = model.evaluate(X_test, y_test, verbose=0)[1], \
                         model.evaluate(X_test, y_test, verbose=0)[0]

    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test Loss     : {test_loss:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=class_names, zero_division=0))

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.title('Sayılar — Confusion Matrix')
    plt.ylabel('Gerçek')
    plt.xlabel('Tahmin')
    plt.tight_layout()
    plt.savefig(f"{OUTPUT_PATH}/sayilar_confusion_matrix.png", dpi=150)
    plt.close()
    print(f"  Confusion matrix: {OUTPUT_PATH}/sayilar_confusion_matrix.png")

    return test_acc


# ─────────────────────────────────────────────────────────────
# TFLite EXPORT
# ─────────────────────────────────────────────────────────────

def export_tflite():
    print("\n" + "=" * 55)
    print("  TFLite Dönüşümü")
    print("=" * 55)

    model     = keras.models.load_model(MODEL_PATH)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations       = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    tflite_model = converter.convert()
    with open(TFLITE_PATH, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(TFLITE_PATH) / 1024
    print(f"  Kaydedildi : {TFLITE_PATH}")
    print(f"  Boyut      : {size_kb:.1f} KB")

    # Doğrula
    interp = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interp.allocate_tensors()
    in_det  = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]
    print(f"  Input      : {in_det['shape']}")
    print(f"  Output     : {out_det['shape']}")
    print(f"  Dönüşüm BAŞARILI ✓")


# ─────────────────────────────────────────────────────────────
# ANA AKIŞ
# ─────────────────────────────────────────────────────────────

def main():
    print("╔" + "═" * 53 + "╗")
    print("║   SAYI MODELİ EĞİTİMİ                              ║")
    print("║   MediaPipe Landmarks + MLP                         ║")
    print("╚" + "═" * 53 + "╝")

    # ── 1. Veri yükle ──────────────────────────────────────
    cache = Path(CACHE_PATH)
    if cache.exists():
        print(f"\n[Cache] Okunuyor: {CACHE_PATH}")
        df = pd.read_pickle(CACHE_PATH)
        print(f"  {len(df)} kayıt yüklendi.")
    else:
        df = load_number_dataset(DATASET_PATH)
        if len(df) == 0:
            print("Veri yüklenemedi, çıkılıyor.")
            return
        df.to_pickle(CACHE_PATH)
        print(f"  Cache kaydedildi: {CACHE_PATH}")

    # ── 2. Split ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  [Adım 2] Veri Split")
    print("=" * 55)

    X  = np.stack(df['features'].values).astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df['label'].values)

    class_names = list(le.classes_)
    num_classes = len(class_names)
    input_dim   = X.shape[1]

    # Test ayır
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y)

    # Val ayır
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp)

    print(f"  Sınıflar    : {class_names}")
    print(f"  Train       : {len(X_train)}")
    print(f"  Val         : {len(X_val)}")
    print(f"  Test        : {len(X_test)}")

    # Classes JSON kaydet
    with open(CLASSES_PATH, 'w', encoding='utf-8') as f:
        json.dump({'classes': class_names, 'num_classes': num_classes},
                  f, ensure_ascii=False, indent=2)
    print(f"  Classes kaydedildi: {CLASSES_PATH}")

    # ── 3. Eğitim ──────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  [Adım 3] Model Eğitimi")
    print("=" * 55)
    print(f"  Input dim   : {input_dim}")
    print(f"  Sınıf sayısı: {num_classes}")

    model = build_model(input_dim, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    model.summary()

    class_weights = compute_class_weights(y_train, num_classes)

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weights,
        callbacks=get_callbacks(),
        verbose=1
    )

    # ── 4. Değerlendir ─────────────────────────────────────
    test_acc = evaluate(model, X_test, y_test, class_names)

    # ── 5. TFLite ──────────────────────────────────────────
    export_tflite()

    # ── Özet ───────────────────────────────────────────────
    print("\n╔" + "═" * 53 + "╗")
    print("║   TAMAMLANDI                                        ║")
    print(f"║   Test Accuracy : {test_acc*100:.2f}%{' ' * 35}║")
    print("╠" + "═" * 53 + "╣")
    print("║   Oluşturulan dosyalar:                             ║")
    for f in ['number_model.h5', 'number_model.tflite', 'number_classes.json']:
        p = f"{OUTPUT_PATH}/{f}"
        if os.path.exists(p):
            kb = os.path.getsize(p) / 1024
            print(f"║   {f:<30s} {kb:>7.1f} KB  ║")
    print("╚" + "═" * 53 + "╝")


if __name__ == '__main__':
    main()
