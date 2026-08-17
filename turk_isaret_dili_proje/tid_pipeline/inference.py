"""
inference.py — Kamera demo + tek görüntü tahmin
MediaPipe 0.10.33 Tasks API uyumlu
"""

import os
import json
import numpy as np
import cv2
import time

from config import cfg
from feature_extraction import HandFeatureExtractor


def load_classes(path: str) -> list:
    if not os.path.exists(path):
        return []
    with open(path, encoding='utf-8') as f:
        return json.load(f)['classes']


# ─────────────────────────────────────────────────────────────
# KERAS INFERENCE
# ─────────────────────────────────────────────────────────────

class SignLanguageInference:

    def __init__(self, letter_model_path=None, word_model_path=None,
                 letter_classes=None, word_classes=None,
                 confidence_threshold=None):

        self.extractor = HandFeatureExtractor()
        self.threshold = confidence_threshold or cfg.CONFIDENCE_THRESHOLD
        self.letter_classes = letter_classes or []
        self.word_classes   = word_classes or []
        self._letter_model  = None
        self._word_model    = None

        if letter_model_path and os.path.exists(letter_model_path):
            import tensorflow as tf
            self._letter_model = tf.keras.models.load_model(letter_model_path)
            print(f"✓ Harf modeli yüklendi ({len(self.letter_classes)} sınıf)")

        if word_model_path and os.path.exists(word_model_path):
            import tensorflow as tf
            self._word_model = tf.keras.models.load_model(word_model_path)
            print(f"✓ Kelime modeli yüklendi ({len(self.word_classes)} sınıf)")

    def predict(self, image: np.ndarray, mode: str = 'letter',
                bbox: tuple = None) -> dict:
        features = self.extractor.extract_features(image, bbox=bbox)
        if features is None:
            return self._no_hand(mode)

        model, classes = (self._letter_model, self.letter_classes) \
            if mode == 'letter' else (self._word_model, self.word_classes)

        if model is None or not classes:
            return self._no_model(mode)

        proba = model.predict(features.reshape(1, -1), verbose=0)[0]
        return self._build(proba, classes, mode)

    def _build(self, proba, classes, mode):
        idx  = int(np.argmax(proba))
        conf = float(proba[idx])
        ok   = conf >= self.threshold
        top3 = [(classes[i], float(proba[i]))
                for i in np.argsort(proba)[::-1][:3] if i < len(classes)]
        return {'label': classes[idx] if ok else 'Belirsiz',
                'confidence': conf, 'is_confident': ok,
                'top3': top3, 'hand_detected': True, 'mode': mode}

    @staticmethod
    def _no_hand(mode):
        return {'label': 'El bulunamadı', 'confidence': 0.0,
                'is_confident': False, 'top3': [], 'hand_detected': False, 'mode': mode}

    @staticmethod
    def _no_model(mode):
        return {'label': 'Model yüklenmedi', 'confidence': 0.0,
                'is_confident': False, 'top3': [], 'hand_detected': True, 'mode': mode}


# ─────────────────────────────────────────────────────────────
# GERÇEK ZAMANLI KAMERA DEMO
# ─────────────────────────────────────────────────────────────

class RealTimeRecognizer:

    def __init__(self, predictor, frame_interval=None):
        self.predictor = predictor
        self.interval  = frame_interval or cfg.FRAME_INTERVAL_SEC
        self.mode      = 'letter'

    def run(self, camera_index: int = 0):
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("HATA: Kamera açılamadı! Kamera indeksini değiştir (0→1)")
            return

        print("\n[Kamera Demo Başladı]")
        print("  'l' → Harf modu")
        print("  'w' → Kelime modu")
        print("  'q' veya ESC → Çıkış\n")

        last_time   = 0.0
        last_result = None

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Kamera görüntüsü alınamadı!")
                break

            frame = cv2.flip(frame, 1)   # ayna görüntü

            # Belirli aralıklarla tahmin
            now = time.time()
            if now - last_time >= self.interval:
                last_result = self.predictor.predict(frame, mode=self.mode)
                last_time   = now
                if last_result['hand_detected']:
                    label = last_result['label']
                    conf  = last_result['confidence']
                    print(f"  [{self.mode.upper()}] {label} ({conf:.1%})")

            self._draw(frame, last_result)
            cv2.imshow('TID Demo  |  l=Harf  w=Kelime  q=Cikis', frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord('q'), 27):
                break
            elif key == ord('l'):
                self.mode = 'letter'
                print("  → Harf moduna geçildi")
            elif key == ord('w'):
                self.mode = 'word'
                print("  → Kelime moduna geçildi")

        cap.release()
        cv2.destroyAllWindows()

    def _draw(self, frame, result):
        if result is None:
            return
        h, w = frame.shape[:2]

        # Alt şerit arka planı
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 140), (w, h), (15, 15, 15), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Mod etiketi
        mod_str = "[ HARF MODU ]" if self.mode == 'letter' else "[ KELIME MODU ]"
        cv2.putText(frame, mod_str, (12, h - 110),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 220, 255), 1)

        # Ana tahmin
        label = result.get('label', '?')
        conf  = result.get('confidence', 0.0)
        color = (0, 230, 60) if result.get('is_confident') else (0, 130, 255)
        cv2.putText(frame, label, (12, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.6, color, 3)
        cv2.putText(frame, f"Guven: {conf:.1%}", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        # Top-3 (sağ taraf)
        for i, (cls, score) in enumerate(result.get('top3', [])[:3]):
            cv2.putText(frame, f"#{i+1} {cls}  {score:.0%}",
                        (w - 210, h - 110 + i * 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        # El bulunamadı uyarısı
        if not result.get('hand_detected'):
            cv2.putText(frame, "El kameraya goster...",
                        (w//2 - 160, h//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 130, 255), 2)
