"""
DATASET LOADER — 2 EL DESTEKLİ
(Python 3.11 + MediaPipe + Windows Türkçe karakter uyumlu)
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

SUPPORTED_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _safe_imread(path):
    """
    Windows'ta Türkçe karakter (İ,Ş,Ğ,Ö,Ü,Ç) içeren
    dosya yollarını güvenli okur.
    cv2.imread() Unicode path'lerde hata verebilir.
    """
    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except Exception:
        return None


def save_classes(classes: list, path: str):
    """Sınıf isimlerini JSON olarak kaydet (Android uyumlu düz liste)."""
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(list(classes), f, ensure_ascii=False, indent=2)


def load_classes(path: str) -> list:
    """JSON'dan sınıf listesi yükle."""
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
        # Hem düz liste hem dict formatını destekle
        if isinstance(data, list):
            return data
        elif isinstance(data, dict) and 'classes' in data:
            return data['classes']
        return data


def _get_images(folder: Path) -> list:
    return [f for f in folder.iterdir() if f.suffix.lower() in SUPPORTED_EXTS]


def _collect_letter_classes(dataset_path: Path) -> dict:
    label_to_images = {}
    ARA = {'train', 'test', 'val', 'valid', 'validation', 'training', 'testing'}
    for subdir in sorted(dataset_path.iterdir()):
        if not subdir.is_dir():
            continue
        if subdir.name.lower() in ARA:
            for letter_dir in sorted(subdir.iterdir()):
                if not letter_dir.is_dir():
                    continue
                label = letter_dir.name.strip()
                imgs = _get_images(letter_dir)
                if imgs:
                    label_to_images.setdefault(label, []).extend(imgs)
        else:
            label = subdir.name.strip()
            imgs = _get_images(subdir)
            if imgs:
                label_to_images[label] = imgs
    return label_to_images


def _debug_structure(path: Path, depth=0, max_depth=2):
    if depth > max_depth:
        return
    prefix = "    " * (depth + 1)
    for item in sorted(path.iterdir())[:8]:
        print(f"{prefix}[{'D' if item.is_dir() else 'F'}] {item.name}")
        if item.is_dir() and depth < max_depth:
            _debug_structure(item, depth + 1, max_depth)


def load_dataset1_letters(dataset_path: str) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  [Dataset 1] Harf veri seti yükleniyor (2 EL)...")
    print("=" * 55)

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"  HATA: {dataset_path} bulunamadı!")
        return pd.DataFrame()

    label_to_images = _collect_letter_classes(dataset_path)
    if not label_to_images:
        print("  HATA: Hiç harf klasörü bulunamadı!")
        _debug_structure(dataset_path)
        return pd.DataFrame()

    print(f"  Bulunan sınıf sayısı : {len(label_to_images)}")
    print(f"  Sınıflar             : {sorted(label_to_images.keys())}")

    tasks = [(img_path, label)
             for label, imgs in sorted(label_to_images.items())
             for img_path in imgs]
    total = len(tasks)
    print(f"  Toplam görüntü       : {total}")
    print(f"  Feature boyutu       : {cfg.TOTAL_FEATURES} (2 el × 78)\n")

    extractor = HandFeatureExtractor()
    records = []
    errors = 0

    for img_path, label in tqdm(tasks, desc="  Feature çıkarılıyor", ncols=70):
        img = _safe_imread(img_path)  # ✅ Türkçe karakter güvenli
        if img is None:
            errors += 1
            continue
        features = extractor.extract_features(img, bbox=None)
        if features is not None:
            records.append({'features': features, 'label': label,
                            'source': 'dataset1_letter'})
        else:
            errors += 1

    if not records:
        print("  HATA: Hiç feature çıkarılamadı!")
        for img_path, label in tasks[:3]:
            img = _safe_imread(img_path)
            print(f"  DEBUG: {img_path.name} → imread {'OK' if img is not None else 'FAIL'}")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"\n  Başarılı  : {len(df)}/{total} ({len(df) / total * 100:.1f}%)")
    print(f"  Başarısız : {errors}")
    print(f"  Sınıf dağılımı:\n{df['label'].value_counts().to_string()}")
    return df


def load_dataset2_words(dataset_path: str, yaml_path: str = None) -> pd.DataFrame:
    print("\n" + "=" * 55)
    print("  [Dataset 2] Kelime veri seti yükleniyor (2 EL)...")
    print("=" * 55)

    dataset_path = Path(dataset_path)
    if not dataset_path.exists():
        print(f"  HATA: {dataset_path} bulunamadı!")
        return pd.DataFrame()

    class_names = _load_class_names(yaml_path, dataset_path)
    print(f"  Sınıflar ({len(class_names)}): {class_names}")

    all_images = _collect_all_images(dataset_path)
    print(f"  Bulunan toplam görüntü: {len(all_images)}")

    tasks = []
    no_label = 0
    for img_path in all_images:
        lp = _find_label_file(img_path, dataset_path)
        if lp is None:
            no_label += 1
            continue
        for class_id, cx, cy, bw, bh in _parse_yolo_label(lp):
            if class_id < len(class_names):
                tasks.append((img_path, class_names[class_id], (cx, cy, bw, bh)))

    print(f"  Annotation bulunamayan : {no_label}")
    print(f"  İşlenecek annotation   : {len(tasks)}\n")

    extractor = HandFeatureExtractor()
    records = []

    for img_path, label, bbox in tqdm(tasks, desc="  Feature çıkarılıyor", ncols=70):
        img = _safe_imread(img_path)  # ✅ Türkçe karakter güvenli
        if img is None:
            continue
        features = extractor.extract_features(img, bbox=bbox)
        if features is not None:
            records.append({'features': features, 'label': label,
                            'source': 'dataset2_word'})

    if not records:
        print("  HATA: Hiç feature çıkarılamadı!")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    print(f"\n  Başarılı: {len(df)}/{len(tasks)}")
    print(f"  Sınıf dağılımı:\n{df['label'].value_counts().to_string()}")
    return df


def prepare_splits(df: pd.DataFrame, mode_name: str):
    if len(df) == 0:
        print(f"  [{mode_name}] Veri yok, atlanıyor.")
        return None, None

    X = np.stack(df['features'].values).astype(np.float32)
    le = LabelEncoder()
    y = le.fit_transform(df['label'].values)

    print(f"\n  [{mode_name}] Feature boyutu: {X.shape[1]} (2 el × 78)")

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=cfg.TEST_SIZE, random_state=42, stratify=y)
    val_ratio = cfg.VAL_SIZE / (1.0 - cfg.TEST_SIZE)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp)

    print(f"  [{mode_name}] Split sonucu:")
    print(f"    Train : {len(X_train)} | Val : {len(X_val)} | Test : {len(X_test)}")
    print(f"    Sınıflar ({len(le.classes_)}): {list(le.classes_)}")
    return (X_train, X_val, X_test, y_train, y_val, y_test), le


def _load_class_names(yaml_path, dataset_path: Path) -> list:
    candidates = []
    if yaml_path:
        candidates.append(Path(yaml_path))
    candidates += [dataset_path / 'data.yaml', dataset_path / 'dataset.yaml']
    for p in candidates:
        if p and p.exists():
            try:
                import yaml
                with open(p, encoding='utf-8') as f:
                    data = yaml.safe_load(f)
                if 'names' in data:
                    names = data['names']
                    if isinstance(names, dict):
                        names = [names[k] for k in sorted(names)]
                    return names
            except Exception as e:
                print(f"  UYARI: {p} okunamadı: {e}")
    for txt in ['classes.txt', 'obj.names']:
        p = dataset_path / txt
        if p.exists():
            with open(p, encoding='utf-8') as f:
                return [ln.strip() for ln in f if ln.strip()]
    return cfg.TURKISH_WORDS


def _collect_all_images(dataset_path: Path) -> list:
    images = []
    for ext in SUPPORTED_EXTS:
        images.extend(dataset_path.rglob(f'*{ext}'))
    images = [p for p in images if 'label' not in str(p).lower()]
    return list(set(images))


def _find_label_file(img_path: Path, dataset_path: Path):
    stem = img_path.stem
    for sep in ['/', '\\']:
        replaced = str(img_path).replace(f'{sep}images{sep}', f'{sep}labels{sep}')
        c = Path(replaced).with_suffix('.txt')
        if c.exists():
            return c
    direct = img_path.parent / (stem + '.txt')
    if direct.exists():
        return direct
    for lbl in dataset_path.rglob(f'{stem}.txt'):
        return lbl
    return None


def _parse_yolo_label(label_path: Path) -> list:
    annotations = []
    with open(label_path, encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                try:
                    annotations.append((int(parts[0]), *map(float, parts[1:])))
                except ValueError:
                    continue
    return annotations