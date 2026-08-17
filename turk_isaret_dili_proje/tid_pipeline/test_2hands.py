import cv2
import numpy as np
from feature_extraction import HandFeatureExtractor

extractor = HandFeatureExtractor()

# Dataset1'den rastgele 50 resim test et
import os, random
from pathlib import Path

data_path = Path("data/dataset1/train")
all_images = []
for letter_dir in data_path.iterdir():
    if letter_dir.is_dir():
        for img in letter_dir.iterdir():
            if img.suffix.lower() in {'.jpg', '.jpeg', '.png'}:
                all_images.append(img)

sample = random.sample(all_images, min(50, len(all_images)))

tek_el = 0
iki_el = 0
sifir = 0

for img_path in sample:
    data = np.fromfile(str(img_path), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if img is None:
        continue
    
    features = extractor.extract_features(img)
    if features is None:
        sifir += 1
        continue
    
    # İlk 78 = sol el, son 78 = sağ el
    left = features[:78]
    right = features[78:]
    
    left_nonzero = np.any(left != 0)
    right_nonzero = np.any(right != 0)
    
    if left_nonzero and right_nonzero:
        iki_el += 1
    elif left_nonzero or right_nonzero:
        tek_el += 1
    else:
        sifir += 1

print(f"\n📊 SONUÇ (50 örnek):")
print(f"  2 el algılanan : {iki_el}")
print(f"  1 el algılanan : {tek_el}")
print(f"  El bulunamayan : {sifir}")
print(f"\n  2 el oranı: {iki_el}/{iki_el+tek_el+sifir} = {iki_el/(iki_el+tek_el+sifir)*100:.0f}%")