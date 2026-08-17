# Türk İşaret Dili Tanıma — Tam ML Pipeline

## Mimari Karar: Neden MediaPipe + MLP?

| Özellik | MediaPipe + MLP ✓ | CNN Alternatifi |
|---|---|---|
| Model boyutu | ~200 KB | ~4–14 MB |
| CPU inference | < 5 ms | 30–80 ms |
| Işık bağımsızlığı | ✓ Evet | ✗ Hayır |
| Arka plan bağımsız | ✓ Evet | ✗ Hayır |
| Veri ihtiyacı | Az (~500+/sınıf) | Çok (2000+) |
| TFLite uyumu | Mükemmel | Orta |

MediaPipe el işkeletini tespit eder (21 landmark), biz koordinatları normalize eder + 15 eklem açısı hesaplarız → 78 boyutlu vektör → MLP ile sınıflandırma.

---

## İki Mod Sistemi — Doğruluğu Artırır mı?

**Evet, anlamlı şekilde artırır.** Nedenleri:

1. **Küçük output uzayı**: 29 harfli model "A" ile "B"yi karıştırma riski taşır ama "Merhaba" ile karıştırmaz. İki ayrı model → confusion azalır.
2. **Kullanıcı niyeti**: Kullanıcı mod seçerek sisteme sinyal verir. Model doğru sınıf uzayında arama yapar.
3. **Bağımsız optimizasyon**: Her mod ayrı eğitilebilir, güncellenebilir.
4. **Ortalama doğruluk artışı**: Literatürde benzer sistemlerde %3–7 iyileşme rapor edilmiştir.

---

## Proje Yapısı

```
tid_pipeline/
├── requirements.txt        # Bağımlılıklar
├── config.py               # Tüm sabitler ve ayarlar
├── feature_extraction.py   # MediaPipe + normalizasyon
├── dataset_loader.py       # Dataset 1 ve 2 yükleyici
├── model.py                # MLP mimarisi
├── trainer.py              # Eğitim + değerlendirme
├── inference.py            # Tahmin sınıfları
├── export_tflite.py        # TFLite dönüşümü
├── inference_example.py    # Kullanım örnekleri
├── main.py                 # Ana pipeline çalıştırıcı
└── data/
    ├── dataset1/           # Kaggle → harf klasörleri
    │   ├── A/
    │   ├── B/
    │   └── ...
    └── dataset2/           # Roboflow → images/ + labels/
        ├── data.yaml
        ├── images/
        │   ├── train/
        │   └── val/
        └── labels/
            ├── train/
            └── val/
```

---

## Kurulum

```bash
# Sanal ortam (önerilen)
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# Bağımlılıkları kur
pip install -r requirements.txt
```

---

## Kullanım

### Tam Pipeline (ilk kez çalıştırma)
```bash
python main.py
```

### Sadece harf modeli
```bash
python main.py --mode letter --epochs 80
```

### Sadece kelime modeli
```bash
python main.py --mode word
```

### Feature çıkarımını atla (pkl cache var)
```bash
python main.py --skip-extract
```

### Tek görüntü tahmini
```bash
python inference_example.py --image goruntu.jpg --mode letter --show
python inference_example.py --image goruntu.jpg --mode word --tflite
```

### Webcam demo
```bash
python main.py --demo
# veya eğitimden sonra direkt:
python -c "from main import step6_demo; step6_demo()"
```

---

## Veri Seti Hazırlığı

### Dataset 1 (Kaggle Harfler)
```
data/dataset1/
├── A/          ← 4800 görüntü
├── B/          ← 4800 görüntü
├── C/
├── Ç/          ← Türkçe karakter desteklenir
...
└── Z/
```
> Annotation yok. MediaPipe tam görüntüde çalışır.

### Dataset 2 (Roboflow Kelimeler)
```
data/dataset2/
├── data.yaml   ← ZORUNLU: sınıf isimleri
├── images/
│   ├── train/  ← .jpg dosyaları
│   └── val/
└── labels/
    ├── train/  ← .txt dosyaları (YOLO format)
    └── val/
```
> Label format: `0 0.4859 0.5187 0.6648 0.4367`
> (class_id cx cy w h — normalize 0-1)

---

## Model Eğitim Detayları

### MLP Mimarisi
```
Input(78)
  → Dense(256) + BatchNorm + ReLU + Dropout(0.35)
  → Dense(128) + BatchNorm + ReLU + Dropout(0.35)
  → Dense(64)  + BatchNorm + ReLU + Dropout(0.17)
  → Dense(N)   + Softmax
```

### Hiperparametreler
- Optimizer: Adam (lr=0.001, clipnorm=1.0)
- Loss: sparse_categorical_crossentropy
- Batch: 32
- Max epoch: 120 (EarlyStopping ile genellikle 40-60'ta durur)
- EarlyStopping patience: 15

### Beklenen Performans
| Dataset | Sınıf | Beklenen Accuracy |
|---|---|---|
| Harfler (Dataset 1) | 29 | %85–95 |
| Kelimeler (Dataset 2) | 20 | %80–92 |

> Not: El tespiti oranına bağlı. MediaPipe bazı karmaşık pozlarda el tespit edemeyebilir.

---

## TFLite Dönüşümü

```bash
# Eğitimden sonra otomatik çalışır, ya da:
python export_tflite.py
```

### Beklenen Dosyalar
| Dosya | Boyut | Format |
|---|---|---|
| letter_model.tflite | ~150-250 KB | FP16 quant |
| word_model.tflite   | ~100-200 KB | FP16 quant |
| letter_classes.json | < 1 KB | Sınıf listesi |
| word_classes.json   | < 1 KB | Sınıf listesi |

---

## Gerçek Zamanlı Kamera Pseudocode (Mobil)

```
BAŞLAT:
  mediapipe_hands = Hands(max_hands=1, static_mode=false)
  letter_model = TFLite("letter_model.tflite")
  word_model   = TFLite("word_model.tflite")
  letter_classes = JSON("letter_classes.json")
  word_classes   = JSON("word_classes.json")
  mod = "harf"          # kullanıcı seçer
  son_tahmin_zamanı = 0
  ARALIK = 1.5 saniye   # önerilen

DÖNGÜ (kamera açık iken):
  frame = kamera.yeni_frame()

  # Her frame'de görselleştirme (akıcı görünüm)
  landmarks_vis = mediapipe_hands.tespit_et(frame)
  ekrana_çiz(frame, landmarks_vis)

  # 1.5 saniyede bir tahmin
  EĞER (şu_an - son_tahmin_zamanı >= ARALIK):
    
    # Feature çıkar
    landmarks = mediapipe_hands.koordinatlar(frame)
    EĞER landmarks == null:
      ekrana_yaz("El bulunamadı")
      DEVAM_ET

    # Normalize
    features = normalize(landmarks)  # (78,) float32
    angles   = eklem_açıları(landmarks)
    input_vector = birleştir(features, angles)  # [1, 78]

    # Mod seç ve tahmin yap
    EĞER mod == "harf":
      proba = letter_model.inference(input_vector)
      sınıflar = letter_classes
    YOKSA:
      proba = word_model.inference(input_vector)
      sınıflar = word_classes

    # Karar
    en_yüksek_idx  = argmax(proba)
    güven_skoru    = proba[en_yüksek_idx]
    EĞER güven_skoru >= 0.65:
      etiket = sınıflar[en_yüksek_idx]
    YOKSA:
      etiket = "Belirsiz"

    ekrana_yaz(etiket, güven_skoru)
    son_tahmin_zamanı = şu_an

ÇIKIŞ:
  kamera.kapat()
```

---

## Python API Kullanımı

```python
from inference_example import TIDPredictor
import cv2

# Başlat
predictor = TIDPredictor(use_tflite=False)  # veya True

# Dosyadan
label, confidence = predictor.predict("goruntu.jpg", mode="letter")
print(f"Tahmin: {label} (%{confidence*100:.1f})")

# Kamera frame'inden
cap = cv2.VideoCapture(0)
ret, frame = cap.read()
label, confidence = predictor.predict_frame(frame, mode="word")

# Tam sonuç (top3, is_confident, vb.)
result = predictor.predict_full(frame, mode="letter")
print(result)
# {
#   'label': 'A',
#   'confidence': 0.87,
#   'is_confident': True,
#   'top3': [('A', 0.87), ('B', 0.08), ('H', 0.02)],
#   'hand_detected': True,
#   'mode': 'letter'
# }
```

---

## Confidence Threshold Ayarı

`config.py` içindeki `CONFIDENCE_THRESHOLD = 0.65` değerini ayarlayın:

| Threshold | Etki |
|---|---|
| 0.50 | Daha fazla tahmin, daha fazla yanlış pozitif |
| 0.65 | Önerilen denge noktası ✓ |
| 0.80 | Çok güvenli, az tahmin ("Belirsiz" daha sık çıkar) |

---

## Yaygın Sorunlar

**El tespit oranı düşük (<60%):**
- Dataset 1 görüntüleri bulanık veya çok küçük olabilir
- MediaPipe `min_detection_confidence` değerini 0.3'e düşür
- Görüntüleri 640x480'e yeniden boyutlandır

**Accuracy düşük:**
- Sınıf başına örnek sayısını kontrol et (min 200 önerilir)
- Epoch sayısını artır: `--epochs 150`
- Dropout'u azalt: `config.py` → `DROPOUT_RATE = 0.25`

**TFLite dönüşümü başarısız:**
- TF sürümünü kontrol et: `tensorflow==2.15.0`
- Model dosyasının mevcut olduğunu kontrol et
