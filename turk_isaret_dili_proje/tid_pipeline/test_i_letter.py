import cv2
import numpy as np
from pathlib import Path
from feature_extraction import HandFeatureExtractor

data_path = Path("data/dataset1/train/İ")

if not data_path.exists():
    print("❌ İ klasörü bulunamadı!")
    exit()

images = [f for f in data_path.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]
print(f"📂 İ klasöründe {len(images)} fotoğraf var\n")

# İlk 20 fotoğrafı test et
extractor = HandFeatureExtractor()
okunan = 0
el_bulunan = 0
hata = 0

for img_path in images[:20]:
    data = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    
    if img is None:
        print(f"  ❌ OKUNAMADI: {img_path.name}")
        hata += 1
        continue
    
    okunan += 1
    features = extractor.extract_features(img)
    
    if features is not None:
        el_bulunan += 1
        print(f"  ✅ EL BULUNDU: {img_path.name}")
    else:
        print(f"  ⚠️ EL BULUNAMADI: {img_path.name}")

print(f"\n📊 SONUÇ (20 örnek):")
print(f"  Okunan:      {okunan}/20")
print(f"  El bulunan:  {el_bulunan}/20")
print(f"  Okuma hatası:{hata}/20")