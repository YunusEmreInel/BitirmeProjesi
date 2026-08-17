"""
=============================================================
TÜRKÇE HARFLER (Ç, Ğ, İ, Ö, Ş, Ü DAHİL) + MIRROR AUGMENTATION
=============================================================
Tek komutla tüm pipeline:
  1. Dataset1'den tüm harfleri yükle (Ç, Ğ, İ, Ö, Ş, Ü dahil)
  2. MediaPipe ile feature extraction
  3. Mirror augmentation (veriyi 2x yap)
  4. Model eğitimi
  5. TFLite export

Kullanım:
    python train_turkish_letters_mirror.py

Gerekli:
    - dataset1/ klasöründe A, B, C, Ç, D, E, F, G, Ğ, H, I, İ, 
      J, K, L, M, N, O, Ö, P, R, S, Ş, T, U, Ü, V, Y, Z, 
      del, space, nothing klasörleri

Çıktı:
    output/letter_model.tflite
    output/letter_classes.json
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
# AYARLAR
# ─────────────────────────────────────────────────────────────

DATASET_PATH  = r"C:\Users\YunusEmre\Desktop\Bitirme Projesi\turk_isaret_dili_proje\tid_pipeline\data\dataset1"
OUTPUT_PATH   = "output"
MODEL_H5      = "output/letter_model.h5"
MODEL_TFLITE  = "output/letter_model.tflite"
CLASSES_JSON  = "output/letter_classes.json"
CACHE_PATH    = "data/processed/letters_turkish.pkl"
CM_PNG        = "output/harfler_turkish_confusion_matrix.png"

BATCH_SIZE    = 32
EPOCHS        = 120
DROPOUT       = 0.35
LR            = 0.001
PATIENCE      = 15
TEST_SIZE     = 0.10
VAL_SIZE      = 0.15

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}

os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs("data/processed", exist_ok=True)


# ─────────────────────────────────────────────────────────────
# VERİ YÜKLEME (Türkçe Harfler Dahil)
# ─────────────────────────────────────────────────────────────

def load_letters_dataset(dataset_path: str, use_cache: bool = True) -> pd.DataFrame:
    """
    Dataset1'den tüm harfleri yükle.
    Yapı: dataset1/train/A/, dataset1/train/B/, dataset1/test/A/, ...
    Ç, Ğ, İ, Ö, Ş, Ü klasörleri varsa otomatik dahil edilir.
    """
    cache = Path(CACHE_PATH)
    
    if use_cache and cache.exists():
        print(f"\n[Cache] Okunuyor: {CACHE_PATH}")
        df = pd.read_pickle(CACHE_PATH)
        print(f"  {len(df)} kayıt yüklendi.")
        print(f"  Sınıflar: {sorted(df['label'].unique())}")
        return df
    
    print("\n" + "=" * 55)
    print("  [Adım 1] Harfler Veri Seti Yükleniyor...")
    print("=" * 55)
    
    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"  HATA: {dataset_path} bulunamadı!")
        return pd.DataFrame()
    
    # train/ ve test/ klasörlerini kontrol et
    split_folders = ['train', 'test', 'validation', 'val']
    
    extractor = HandFeatureExtractor()
    records   = []
    errors    = 0
    
    # Sınıf klasörlerini topla
    label_to_images = {}
    
    for split_name in split_folders:
        split_path = dataset_path / split_name
        if not split_path.exists():
            continue
        
        print(f"  Taranıyor: {split_name}/")
        
        for class_dir in sorted(split_path.iterdir()):
            if not class_dir.is_dir():
                continue
            
            label = class_dir.name  # A, B, Ç, Ğ, İ, ...
            images = [f for f in class_dir.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]
            
            if label not in label_to_images:
                label_to_images[label] = []
            label_to_images[label].extend(images)
    
    if not label_to_images:
        print("  HATA: Hiç sınıf klasörü bulunamadı!")
        return pd.DataFrame()
    
    print(f"\n  Bulunan sınıflar: {len(label_to_images)}")
    print(f"  {sorted(label_to_images.keys())}\n")
    
    # Her sınıf için feature extraction
    for label, images in sorted(label_to_images.items()):
        class_ok = 0
        for img_path in tqdm(images, desc=f"  {label:>10s}", ncols=70):
            # OpenCV encoding fix: read as bytes first
            try:
                with open(img_path, 'rb') as f:
                    file_bytes = np.frombuffer(f.read(), dtype=np.uint8)
                img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
            except Exception as e:
                errors += 1
                continue
            
            if img is None:
                errors += 1
                continue
            
            features = extractor.extract_features(img, bbox=None)
            if features is not None:
                records.append({
                    'features': features,
                    'label': label,
                    'source': 'dataset1'
                })
                class_ok += 1
            else:
                errors += 1
        
        rate = class_ok / max(len(images), 1) * 100
        print(f"  {label:>10s}: {class_ok}/{len(images)} (%{rate:.0f} el tespit)")
    
    if not records:
        print("  HATA: Hiç feature çıkarılamadı!")
        return pd.DataFrame()
    
    df = pd.DataFrame(records)
    
    # Cache'e kaydet
    df.to_pickle(CACHE_PATH)
    print(f"\n  Cache kaydedildi: {CACHE_PATH}")
    print(f"  Başarılı: {len(df)} kayıt")
    print(f"  Sınıf dağılımı:\n{df['label'].value_counts().sort_index().to_string()}")
    
    return df


# ─────────────────────────────────────────────────────────────
# MIRROR AUGMENTATION
# ─────────────────────────────────────────────────────────────

def mirror_features(features: np.ndarray) -> np.ndarray:
    """78 boyutlu vektörün x koordinatlarını ters çevir."""
    mirrored = features.copy().astype(np.float32)
    mirrored[0:63:3] = -mirrored[0:63:3]  # x'leri ters çevir
    return mirrored


def apply_mirror_augmentation(df: pd.DataFrame) -> pd.DataFrame:
    """Veri setini aynalı kopyalarla 2x yap."""
    print("\n" + "=" * 55)
    print("  [Adım 2] Mirror Augmentation")
    print("=" * 55)
    print(f"  Orijinal: {len(df)} örnek")
    
    mirrored_records = []
    for _, row in df.iterrows():
        mirrored_records.append({
            'features': mirror_features(row['features']),
            'label': row['label'],
            'source': row.get('source', '') + '_mirror'
        })
    
    df_mirror = pd.DataFrame(mirrored_records)
    df_augmented = pd.concat([df, df_mirror], ignore_index=True)
    df_augmented = df_augmented.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    print(f"  Augmented: {len(df_augmented)} örnek (2x)")
    return df_augmented


# ─────────────────────────────────────────────────────────────
# MODEL MİMARİSİ
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
    return keras.Model(inputs, outputs, name='TID_Turkish_Letters_MLP')


def get_callbacks():
    return [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy', patience=PATIENCE,
            restore_best_weights=True, verbose=1),
        keras.callbacks.ModelCheckpoint(
            filepath=MODEL_H5, monitor='val_accuracy',
            save_best_only=True, verbose=1),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5,
            patience=PATIENCE // 2, min_lr=1e-6, verbose=1),
        keras.callbacks.CSVLogger(
            MODEL_H5.replace('.h5', '_history.csv'))
    ]


def compute_class_weights(y, num_classes):
    n = len(y)
    return {c: n / (num_classes * max(np.sum(y == c), 1))
            for c in range(num_classes)}


# ─────────────────────────────────────────────────────────────
# DEĞERLENDİRME
# ─────────────────────────────────────────────────────────────

def evaluate_model(model, X_test, y_test, class_names):
    print("\n" + "=" * 55)
    print("  [Adım 4] Değerlendirme")
    print("=" * 55)
    
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    eval_result = model.evaluate(X_test, y_test, verbose=0)
    test_loss, test_acc = eval_result[0], eval_result[1]
    
    print(f"  Test Accuracy : {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"  Test Loss     : {test_loss:.4f}")
    print("\n  Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=class_names, zero_division=0))
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, normalize='true')
    fig_size = max(10, int(len(class_names) * 0.4))
    plt.figure(figsize=(fig_size, fig_size))
    sns.heatmap(cm, annot=True, fmt='.0%', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names,
                cbar=True, annot_kws={'size': 7})
    plt.title('Türkçe Harfler — Confusion Matrix (Mirror)')
    plt.ylabel('Gerçek Sınıf')
    plt.xlabel('Tahmin')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(CM_PNG, dpi=150)
    plt.close()
    print(f"  Confusion matrix: {CM_PNG}")
    
    return test_acc


# ─────────────────────────────────────────────────────────────
# TFLite EXPORT
# ─────────────────────────────────────────────────────────────

def export_tflite():
    print("\n" + "=" * 55)
    print("  [Adım 5] TFLite Dönüşümü")
    print("=" * 55)
    
    model = keras.models.load_model(MODEL_H5)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]
    
    tflite_model = converter.convert()
    with open(MODEL_TFLITE, 'wb') as f:
        f.write(tflite_model)
    
    size_kb = os.path.getsize(MODEL_TFLITE) / 1024
    print(f"  Kaydedildi : {MODEL_TFLITE}")
    print(f"  Boyut      : {size_kb:.1f} KB")
    
    # Doğrula
    interp = tf.lite.Interpreter(model_path=MODEL_TFLITE)
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
    print("\n╔" + "═" * 55 + "╗")
    print("║   TÜRKÇE HARFLER + MIRROR AUGMENTATION EĞİTİMİ      ║")
    print("║   Ç, Ğ, İ, Ö, Ş, Ü dahil - Sol/Sağ el desteği      ║")
    print("╚" + "═" * 55 + "╝")
    
    # ── 1. Veri yükle ──────────────────────────────────────
    df = load_letters_dataset(DATASET_PATH, use_cache=False)
    if len(df) == 0:
        print("Veri yüklenemedi, çıkılıyor.")
        return
    
    # ── 2. Mirror augmentation ─────────────────────────────
    df_augmented = apply_mirror_augmentation(df)
    
    # ── 3. Split ───────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  [Adım 3] Veri Split")
    print("=" * 55)
    
    X  = np.stack(df_augmented['features'].values).astype(np.float32)
    le = LabelEncoder()
    y  = le.fit_transform(df_augmented['label'].values)
    
    class_names = list(le.classes_)
    num_classes = len(class_names)
    input_dim   = X.shape[1]
    
    # Test ayır
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=42, stratify=y)
    
    # Validation ayır
    val_ratio = VAL_SIZE / (1.0 - TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp)
    
    print(f"  Sınıflar    : {class_names}")
    print(f"  Sınıf sayısı: {num_classes}")
    print(f"  Train       : {len(X_train)}")
    print(f"  Val         : {len(X_val)}")
    print(f"  Test        : {len(X_test)}")
    
    # Classes JSON kaydet
    with open(CLASSES_JSON, 'w', encoding='utf-8') as f:
        json.dump({'classes': class_names, 'num_classes': num_classes},
                  f, ensure_ascii=False, indent=2)
    print(f"  Classes JSON: {CLASSES_JSON}")
    
    # ── 4. Model eğitimi ───────────────────────────────────
    print("\n" + "=" * 55)
    print("  [Adım 4] Model Eğitimi")
    print("=" * 55)
    
    model = build_model(input_dim, num_classes)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=LR, clipnorm=1.0),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
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
    
    # ── 5. Değerlendir ─────────────────────────────────────
    test_acc = evaluate_model(model, X_test, y_test, class_names)
    
    # ── 6. TFLite ──────────────────────────────────────────
    export_tflite()
    
    # ── Özet ───────────────────────────────────────────────
    print("\n╔" + "═" * 55 + "╗")
    print("║   TAMAMLANDI                                        ║")
    print(f"║   Test Accuracy : {test_acc*100:.2f}%{' ' * (43 - len(f'{test_acc*100:.2f}'))}║")
    print("╠" + "═" * 55 + "╣")
    print("║   Oluşturulan Dosyalar:                             ║")
    for fname in ['letter_model.h5', 'letter_model.tflite', 'letter_classes.json']:
        fpath = os.path.join(OUTPUT_PATH, fname)
        if os.path.exists(fpath):
            kb = os.path.getsize(fpath) / 1024
            print(f"║   {fname:<30s} {kb:>7.1f} KB{'  ' * 2}║")
    print("╚" + "═" * 55 + "╝")
    
    print("\n📱 Android'e Yüklemek İçin:")
    print("   1. output/letter_model.tflite")
    print("   2. output/letter_classes.json")
    print("   3. app/src/main/assets/ klasörüne kopyala")
    print("   4. Android Studio'da Run ▶\n")


if __name__ == '__main__':
    main()
