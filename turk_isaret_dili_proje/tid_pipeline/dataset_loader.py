"""
=============================================================
DATASET LOADER
=============================================================
Dataset 1 (Kaggle — Harfler):
  Yapı: dataset1/<HARF_ADI>/<görüntü>.jpg
  Annotation: YOK — MediaPipe tam görüntüye uygulanır

Dataset 2 (Roboflow — Kelimeler):
  Yapı: dataset2/images/**/<görüntü>.jpg
         dataset2/labels/**/<görüntü>.txt
  Annotation: YOLO formatı → "class_id cx cy w h"
  Örnek: "0 0.4859375 0.51875 0.66484375 0.43671875"
=============================================================
"""

import os
import json
import numpy as np
import pandas as pd
import cv2
from pathlib import Path
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from config import cfg
from feature_extraction import HandFeatureExtractor


# ─────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ─────────────────────────────────────────────────────────────

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def save_classes(classes: list, path: str):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump({'classes': list(classes), 'num_classes': len(classes)}, f,
                  ensure_ascii=False, indent=2)


def load_classes(path: str) -> list:
    with open(path, encoding='utf-8') as f:
        return json.load(f)['classes']


# ─────────────────────────────────────────────────────────────
# DATASET 1 LOADER — Kaggle Harfler
# ─────────────────────────────────────────────────────────────

def _get_images(folder: Path) -> list:
    """Klasördeki tüm desteklenen görüntü dosyalarını döndür."""
    return [f for f in folder.iterdir()
            if f.suffix.lower() in SUPPORTED_EXTS]


def _collect_letter_classes(dataset_path: Path) -> dict:
    """
    Hem düz hem iç içe yapıyı destekler:
      Düz  : dataset1/A/*.jpg
      İç içe: dataset1/train/A/*.jpg + dataset1/test/A/*.jpg
    """
    label_to_images = {}
    ARA_KLASORLER = {'train', 'test', 'val', 'valid', 'validation',
                     'training', 'testing'}

    for subdir in sorted(dataset_path.iterdir()):
        if not subdir.is_dir():
            continue

        if subdir.name.lower() in ARA_KLASORLER:
            # İçine gir, harf klasörlerini al
            for letter_dir in sorted(subdir.iterdir()):
                if not letter_dir.is_dir():
                    continue
                label = letter_dir.name.strip()
                images = _get_images(letter_dir)
                if images:
                    label_to_images.setdefault(label, []).extend(images)
        else:
            # Direkt harf klasörü
            label = subdir.name.strip()
            images = _get_images(subdir)
            if images:
                label_to_images[label] = images

    return label_to_images


def _debug_structure(path: Path, depth: int = 0, max_depth: int = 2):
    """Klasör yapısını debug için yazdır."""
    if depth > max_depth:
        return
    prefix = "    " * (depth + 1)
    for item in sorted(path.iterdir())[:8]:
        icon = "D" if item.is_dir() else "F"
        print(f"{prefix}[{icon}] {item.name}")
        if item.is_dir() and depth < max_depth:
            _debug_structure(item, depth + 1, max_depth)


def load_dataset1_letters(dataset_path: str) -> pd.DataFrame:
    """
    Dataset 1'i yükle. Düz ve iç içe (train/test) yapıyı otomatik algılar.
    """
    print("\n" + "=" * 55)
    print("  [Dataset 1] Harf veri seti yükleniyor...")
    print("=" * 55)

    extractor = HandFeatureExtractor()
    records = []
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"  HATA: {dataset_path} bulunamadı!")
        return pd.DataFrame()

    # Klasör yapısını otomatik algıla
    label_to_images = _collect_letter_classes(dataset_path)

    if not label_to_images:
        print("  HATA: Hiç harf klasörü bulunamadı!")
        print("  Mevcut klasör yapısı:")
        _debug_structure(dataset_path)
        return pd.DataFrame()

    print(f"  Bulunan sınıf sayısı : {len(label_to_images)}")
    print(f"  Sınıflar             : {sorted(label_to_images.keys())}")
    total_images = sum(len(v) for v in label_to_images.values())
    print(f"  Toplam görüntü       : {total_images}")

    total_success = 0

    for label, image_files in sorted(label_to_images.items()):
        class_success = 0
        for img_path in tqdm(image_files, desc=f"  {label:>4}", leave=False,
                              ncols=70):
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            features = extractor.extract_features(img, bbox=None)
            if features is not None:
                records.append({
                    'features': features,
                    'label': label,
                    'source': 'dataset1_letter'
                })
                class_success += 1

        total_success += class_success
        rate = class_success / max(len(image_files), 1) * 100
        print(f"  {label:>4}: {class_success}/{len(image_files)} "
              f"({rate:.0f}% el tespit)")

    if not records:
        print("\n  HATA: Hiç feature çıkarılamadı!")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"\n  Toplam: {total_success}/{total_images} görüntü başarılı")
    print(f"  El tespit oranı: {total_success/max(total_images,1)*100:.1f}%")
    print(f"  Sınıf dağılımı:\n{df['label'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────
# DATASET 2 LOADER — Roboflow Kelimeler
# ─────────────────────────────────────────────────────────────

def load_dataset2_words(dataset_path: str, yaml_path: str = None) -> pd.DataFrame:
    """
    Dataset 2'yi (Roboflow YOLO format) yükle.
    Label dosyasından bbox alınır → crop → MediaPipe.

    Returns:
        DataFrame ile sütunlar: features, label, source
    """
    print("\n" + "=" * 55)
    print("  [Dataset 2] Kelime veri seti yükleniyor...")
    print("=" * 55)

    extractor = HandFeatureExtractor()
    records = []
    dataset_path = Path(dataset_path)

    if not dataset_path.exists():
        print(f"  HATA: {dataset_path} bulunamadı!")
        return pd.DataFrame()

    # 1. Sınıf isimlerini yükle
    class_names = _load_class_names(yaml_path, dataset_path)
    print(f"  Sınıflar ({len(class_names)}): {class_names}")

    # 2. Tüm görüntü dosyalarını bul (train/val/test split'lerinden)
    all_images = _collect_all_images(dataset_path)
    print(f"  Bulunan toplam görüntü: {len(all_images)}")

    no_label = 0
    no_hand = 0
    success = 0

    for img_path in tqdm(all_images, desc="  İşleniyor", ncols=70):
        label_path = _find_label_file(img_path, dataset_path)
        if label_path is None:
            no_label += 1
            continue

        annotations = _parse_yolo_label(label_path)
        if not annotations:
            no_label += 1
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        for class_id, cx, cy, bw, bh in annotations:
            if class_id >= len(class_names):
                continue

            label = class_names[class_id]
            features = extractor.extract_features(img, bbox=(cx, cy, bw, bh))

            if features is not None:
                records.append({
                    'features': features,
                    'label': label,
                    'source': 'dataset2_word'
                })
                success += 1
            else:
                no_hand += 1

    df = pd.DataFrame(records)
    print(f"\n  Başarılı: {success} | Label yok: {no_label} | "
          f"El tespit edilemedi: {no_hand}")
    if len(df) > 0:
        print(f"  Sınıf dağılımı:\n{df['label'].value_counts().to_string()}")
    return df


# ─────────────────────────────────────────────────────────────
# VERİ BİRLEŞTİRME VE SPLIT
# ─────────────────────────────────────────────────────────────

def prepare_splits(df: pd.DataFrame, mode_name: str):
    """
    DataFrame'i X/y train/val/test split'lerine böl.
    Stratified split → sınıf dengesini korur.

    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test), LabelEncoder
    """
    if len(df) == 0:
        print(f"  [{mode_name}] Veri yok, atlanıyor.")
        return None, None

    X = np.stack(df['features'].values).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    # İlk split: test ayır
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y,
        test_size=cfg.TEST_SIZE,
        random_state=42,
        stratify=y
    )

    # İkinci split: val ayır
    val_ratio = cfg.VAL_SIZE / (1.0 - cfg.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_ratio,
        random_state=42,
        stratify=y_temp
    )

    print(f"\n  [{mode_name}] Split sonucu:")
    print(f"    Train : {len(X_train)} örnek")
    print(f"    Val   : {len(X_val)} örnek")
    print(f"    Test  : {len(X_test)} örnek")
    print(f"    Sınıflar ({len(le.classes_)}): {list(le.classes_)}")

    return (X_train, X_val, X_test, y_train, y_val, y_test), le


# ─────────────────────────────────────────────────────────────
# YARDIMCI PRIVATE FONKSİYONLAR
# ─────────────────────────────────────────────────────────────

def _load_class_names(yaml_path, dataset_path: Path) -> list:
    """data.yaml / classes.txt'den sınıf isimlerini oku."""
    candidates = []
    if yaml_path:
        candidates.append(Path(yaml_path))
    candidates += [
        dataset_path / 'data.yaml',
        dataset_path / 'dataset.yaml',
        dataset_path / 'classes.yaml',
    ]

    for p in candidates:
        if p and p.exists():
            try:
                import yaml
                with open(p, encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if 'names' in data:
                    names = data['names']
                    # YAML bazen dict {0: 'Anne', 1: 'Baba'} verir
                    if isinstance(names, dict):
                        names = [names[k] for k in sorted(names)]
                    return names
            except Exception as e:
                print(f"  UYARI: {p} okunamadı: {e}")

    for txt in ['classes.txt', 'obj.names', '_darknet.labels']:
        p = dataset_path / txt
        if p.exists():
            with open(p, encoding='utf-8') as f:
                return [ln.strip() for ln in f if ln.strip()]

    print("  UYARI: Sınıf dosyası bulunamadı → varsayılan TURKISH_WORDS kullanılıyor")
    return cfg.TURKISH_WORDS


def _collect_all_images(dataset_path: Path) -> list:
    """Dataset altındaki tüm görüntü dosyalarını topla (train/val/test)."""
    images = []
    for ext in SUPPORTED_EXTS:
        images.extend(dataset_path.rglob(f'*{ext}'))
        images.extend(dataset_path.rglob(f'*{ext.upper()}'))
    # labels/ klasöründeki görüntüleri hariç tut
    images = [p for p in images if 'label' not in str(p).lower()]
    return list(set(images))


def _find_label_file(img_path: Path, dataset_path: Path):
    """Görüntüye karşılık gelen YOLO label dosyasını bul."""
    stem = img_path.stem

    # Yol içinde images/ → labels/ değiştir
    path_str = str(img_path)
    for sep in ['/', '\\']:
        replaced = path_str.replace(
            f'{sep}images{sep}', f'{sep}labels{sep}'
        )
        candidate = Path(replaced).with_suffix('.txt')
        if candidate.exists():
            return candidate

    # Aynı klasörde .txt ara
    direct = img_path.parent / (stem + '.txt')
    if direct.exists():
        return direct

    # Dataset genelinde ara
    for lbl in dataset_path.rglob(f'{stem}.txt'):
        return lbl

    return None


def _parse_yolo_label(label_path: Path) -> list:
    """
    YOLO label dosyasını parse et.
    Her satır: class_id cx cy w h (hepsi float/int)
    Returns: list of (class_id, cx, cy, bw, bh)
    """
    annotations = []
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    class_id = int(parts[0])
                    cx, cy, bw, bh = map(float, parts[1:])
                    annotations.append((class_id, cx, cy, bw, bh))
                except ValueError:
                    continue
    return annotations