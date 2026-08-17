"""
=============================================================
INFERENCE ÖRNEĞİ — Tek Görüntüden Tahmin
=============================================================
Pipeline eğitildikten sonra bu dosyayı çalıştırın:

    python inference_example.py --image /yol/goruntu.jpg --mode letter

Veya TFLite ile:

    python inference_example.py --image /yol/goruntu.jpg --tflite
=============================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import cv2

from config import cfg
from dataset_loader import load_classes
from inference import SignLanguageInference, TFLiteInference


def parse_args():
    p = argparse.ArgumentParser(description='TID — Tek Görüntü Tahmin')
    p.add_argument('--image', type=str, required=True,
                   help='Tahmin yapılacak görüntü yolu')
    p.add_argument('--mode', choices=['letter', 'word'], default='letter',
                   help='Harf modu mu, kelime modu mu (varsayılan: letter)')
    p.add_argument('--tflite', action='store_true',
                   help='Keras yerine TFLite modeli kullan')
    p.add_argument('--threshold', type=float, default=cfg.CONFIDENCE_THRESHOLD,
                   help=f'Confidence eşiği (varsayılan: {cfg.CONFIDENCE_THRESHOLD})')
    p.add_argument('--show', action='store_true',
                   help='Görüntüyü landmark ile göster')
    return p.parse_args()


def load_class_lists():
    """Sınıf listelerini yükle."""
    letter_classes, word_classes = [], []

    if os.path.exists(cfg.LETTER_CLASSES_JSON):
        letter_classes = load_classes(cfg.LETTER_CLASSES_JSON)
        print(f"Harf sınıfları yüklendi: {len(letter_classes)} sınıf")
    else:
        print(f"UYARI: {cfg.LETTER_CLASSES_JSON} bulunamadı!")

    if os.path.exists(cfg.WORD_CLASSES_JSON):
        word_classes = load_classes(cfg.WORD_CLASSES_JSON)
        print(f"Kelime sınıfları yüklendi: {len(word_classes)} sınıf")
    else:
        print(f"UYARI: {cfg.WORD_CLASSES_JSON} bulunamadı!")

    return letter_classes, word_classes


def print_result(result: dict):
    """Tahmin sonucunu güzel biçimde yazdır."""
    print("\n" + "─" * 45)
    print("  TAHMİN SONUCU")
    print("─" * 45)

    hand_icon = "✓" if result['hand_detected'] else "✗"
    print(f"  El tespit    : {hand_icon} {'Evet' if result['hand_detected'] else 'Hayır'}")

    if result['hand_detected']:
        conf_icon = "✓" if result['is_confident'] else "⚠"
        print(f"  Tahmin       : {result['label']}")
        print(f"  Güven skoru  : {result['confidence']:.1%} {conf_icon}")
        print(f"  Mod          : {result['mode']}")

        if result.get('top3'):
            print(f"\n  Top-3 Tahminler:")
            for rank, (lbl, score) in enumerate(result['top3'], 1):
                bar = '█' * int(score * 20)
                print(f"    #{rank} {lbl:25s} {score:.1%} {bar}")

        if not result['is_confident']:
            print(f"\n  ⚠ Güven skoru eşiğin altında ({cfg.CONFIDENCE_THRESHOLD:.0%})")
            print(f"    → 'Belirsiz' olarak işaretlendi")
            print(f"    → El kameraya daha yakın tutulabilir veya")
            print(f"       threshold düşürülebilir (--threshold 0.5)")
    print("─" * 45)


def visualize_with_landmarks(image: np.ndarray, result: dict, mode: str):
    """Görüntüyü landmark ve tahmin ile göster."""
    from feature_extraction import HandFeatureExtractor
    extractor = HandFeatureExtractor()
    vis = extractor.visualize_landmarks(image.copy())

    h, w = vis.shape[:2]
    label = result.get('label', '?')
    conf  = result.get('confidence', 0.0)
    color = (0, 200, 50) if result.get('is_confident') else (0, 140, 255)

    # Arka plan kutusu
    cv2.rectangle(vis, (0, h-80), (w, h), (15, 15, 15), -1)
    cv2.putText(vis, f"{label}", (10, h-45),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, color, 3)
    cv2.putText(vis, f"Guven: {conf:.1%} | Mod: {mode}",
                (10, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (200, 200, 200), 1)

    cv2.imshow("TID Tahmin", vis)
    print("\n  Görüntü gösteriliyor. Kapatmak için herhangi bir tuşa basın.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main():
    args = parse_args()

    # Görüntüyü oku
    if not os.path.exists(args.image):
        print(f"HATA: Görüntü bulunamadı: {args.image}")
        sys.exit(1)

    image = cv2.imread(args.image)
    if image is None:
        print(f"HATA: Görüntü okunamadı: {args.image}")
        sys.exit(1)

    print(f"Görüntü: {args.image}  ({image.shape[1]}x{image.shape[0]})")

    # Sınıf listelerini yükle
    letter_classes, word_classes = load_class_lists()

    # Predictor oluştur
    if args.tflite:
        print("\nTFLite inference kullanılıyor...")
        predictor = TFLiteInference(
            letter_tflite_path=cfg.LETTER_TFLITE_PATH,
            word_tflite_path=cfg.WORD_TFLITE_PATH,
            letter_classes=letter_classes,
            word_classes=word_classes,
            confidence_threshold=args.threshold
        )
    else:
        print("\nKeras inference kullanılıyor...")
        predictor = SignLanguageInference(
            letter_model_path=cfg.LETTER_MODEL_PATH,
            word_model_path=cfg.WORD_MODEL_PATH,
            letter_classes=letter_classes,
            word_classes=word_classes,
            confidence_threshold=args.threshold
        )

    # Tahmin yap
    result = predictor.predict(image, mode=args.mode)

    # Sonucu yazdır
    print_result(result)

    # Görselleştir (opsiyonel)
    if args.show:
        visualize_with_landmarks(image, result, args.mode)


# ─────────────────────────────────────────────────────────────
# PYTHON API OLARAK KULLANIM ÖRNEĞİ
# ─────────────────────────────────────────────────────────────

def example_api_usage():
    """
    Bu fonksiyon Python kodunuzda direkt import edip
    kullanabileceğiniz örnek bir wrapper'dır.

    Örnek:
        from inference_example import TIDPredictor
        predictor = TIDPredictor()
        label, conf = predictor.predict_letter('/yol/goruntu.jpg')
    """
    pass


class TIDPredictor:
    """
    Kolay kullanım için sarmalayıcı sınıf.

    Kullanım:
        p = TIDPredictor()
        label, confidence = p.predict('/yol/goruntu.jpg', mode='letter')
        # veya kameradan gelen frame ile:
        label, confidence = p.predict_frame(frame, mode='word')
    """

    def __init__(self, use_tflite: bool = False,
                 confidence_threshold: float = None):
        letter_classes = load_classes(cfg.LETTER_CLASSES_JSON) \
            if os.path.exists(cfg.LETTER_CLASSES_JSON) else []
        word_classes = load_classes(cfg.WORD_CLASSES_JSON) \
            if os.path.exists(cfg.WORD_CLASSES_JSON) else []

        threshold = confidence_threshold or cfg.CONFIDENCE_THRESHOLD

        if use_tflite:
            self._predictor = TFLiteInference(
                letter_tflite_path=cfg.LETTER_TFLITE_PATH,
                word_tflite_path=cfg.WORD_TFLITE_PATH,
                letter_classes=letter_classes,
                word_classes=word_classes,
                confidence_threshold=threshold
            )
        else:
            self._predictor = SignLanguageInference(
                letter_model_path=cfg.LETTER_MODEL_PATH,
                word_model_path=cfg.WORD_MODEL_PATH,
                letter_classes=letter_classes,
                word_classes=word_classes,
                confidence_threshold=threshold
            )

    def predict(self, image_path: str, mode: str = 'letter'):
        """Dosya yolundan tahmin yap."""
        img = cv2.imread(image_path)
        result = self._predictor.predict(img, mode=mode)
        return result['label'], result['confidence']

    def predict_frame(self, frame: np.ndarray, mode: str = 'letter'):
        """NumPy array (kamera frame) ile tahmin yap."""
        result = self._predictor.predict(frame, mode=mode)
        return result['label'], result['confidence']

    def predict_full(self, frame: np.ndarray, mode: str = 'letter') -> dict:
        """Tam sonuç dict'ini döndür (top3 ve tüm detaylar)."""
        return self._predictor.predict(frame, mode=mode)


if __name__ == '__main__':
    main()
