"""
=============================================================
INFERENCE — Tek Görüntü Tahmin Modülü
=============================================================
Bu modül production'da kullanılacak asıl tahmin sınıfını içerir.
Hem .h5 Keras modeli hem de .tflite modeli desteklenir.

Mobil uygulamada kullanım:
  - Python tabanlı backend: SignLanguageInference sınıfı
  - Android/iOS native: TFLite interpreter (Java/Swift)
=============================================================
"""

import os
import numpy as np
import cv2
from pathlib import Path

from config import cfg
from feature_extraction import HandFeatureExtractor


# ─────────────────────────────────────────────────────────────
# KERAS INFERENCE
# ─────────────────────────────────────────────────────────────

class SignLanguageInference:
    """
    İşaret dili tanıma — tek görüntü inference.

    Kullanım:
        predictor = SignLanguageInference(
            letter_model_path='output/letter_model.h5',
            word_model_path='output/word_model.h5',
            letter_classes=['A','B','C',...],
            word_classes=['Anne','Baba',...]
        )
        result = predictor.predict(frame, mode='letter')
        print(result['label'], result['confidence'])
    """

    def __init__(self,
                 letter_model_path: str = None,
                 word_model_path: str = None,
                 letter_classes: list = None,
                 word_classes: list = None,
                 confidence_threshold: float = None):

        self.extractor = HandFeatureExtractor()
        self.threshold = confidence_threshold or cfg.CONFIDENCE_THRESHOLD

        self._letter_model = None
        self._word_model = None
        self.letter_classes = letter_classes or []
        self.word_classes = word_classes or []

        # Model yükleme
        if letter_model_path and os.path.exists(letter_model_path):
            import tensorflow as tf
            self._letter_model = tf.keras.models.load_model(letter_model_path)
            print(f"✓ Harf modeli yüklendi: {letter_model_path}")
        else:
            print(f"⚠ Harf modeli bulunamadı: {letter_model_path}")

        if word_model_path and os.path.exists(word_model_path):
            import tensorflow as tf
            self._word_model = tf.keras.models.load_model(word_model_path)
            print(f"✓ Kelime modeli yüklendi: {word_model_path}")
        else:
            print(f"⚠ Kelime modeli bulunamadı: {word_model_path}")

    # ── ANA TAHMİN FONKSİYONU ────────────────────────────────

    def predict(self, image: np.ndarray, mode: str = 'letter',
                bbox: tuple = None) -> dict:
        """
        Tek görüntüden tahmin yap.

        Args:
            image : BGR numpy array (kameradan gelen frame)
            mode  : 'letter' veya 'word'
            bbox  : (cx, cy, w, h) normalize YOLO bbox — opsiyonel

        Returns:
            {
              'label'        : str   — tahmin edilen etiket
              'confidence'   : float — en yüksek sınıf olasılığı
              'is_confident' : bool  — threshold'u geçti mi
              'top3'         : list  — [(label, score), ...]
              'hand_detected': bool  — el tespit edildi mi
              'mode'         : str   — hangi mod kullanıldı
            }
        """
        # ── 1. Feature çıkar ────────────────────────────────
        features = self.extractor.extract_features(image, bbox=bbox)

        if features is None:
            return self._no_hand_result(mode)

        # ── 2. Model seç ────────────────────────────────────
        model, classes = self._get_model_and_classes(mode)
        if model is None:
            return self._model_not_loaded_result(mode)

        # ── 3. Tahmin ───────────────────────────────────────
        proba = model.predict(features.reshape(1, -1), verbose=0)[0]
        return self._build_result(proba, classes, mode)

    def predict_from_path(self, image_path: str,
                           mode: str = 'letter') -> dict:
        """Dosya yolundan tahmin yap."""
        img = cv2.imread(str(image_path))
        if img is None:
            raise FileNotFoundError(f"Görüntü okunamadı: {image_path}")
        return self.predict(img, mode)

    # ── YARDIMCI METOTLAR ────────────────────────────────────

    def _get_model_and_classes(self, mode: str):
        if mode == 'letter':
            return self._letter_model, self.letter_classes
        elif mode == 'word':
            return self._word_model, self.word_classes
        else:
            raise ValueError(f"Geçersiz mod: '{mode}'. 'letter' veya 'word' olmalı.")

    def _build_result(self, proba: np.ndarray, classes: list,
                       mode: str) -> dict:
        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])

        # Threshold kontrolü
        is_confident = confidence >= self.threshold
        label = classes[pred_idx] if is_confident else 'Belirsiz'

        # Top-3
        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [(classes[i], float(proba[i])) for i in top3_idx
                if i < len(classes)]

        return {
            'label': label,
            'confidence': confidence,
            'is_confident': is_confident,
            'top3': top3,
            'hand_detected': True,
            'mode': mode
        }

    @staticmethod
    def _no_hand_result(mode: str) -> dict:
        return {
            'label': 'El tespit edilemedi',
            'confidence': 0.0,
            'is_confident': False,
            'top3': [],
            'hand_detected': False,
            'mode': mode
        }

    @staticmethod
    def _model_not_loaded_result(mode: str) -> dict:
        return {
            'label': 'Model yüklenmedi',
            'confidence': 0.0,
            'is_confident': False,
            'top3': [],
            'hand_detected': True,
            'mode': mode
        }


# ─────────────────────────────────────────────────────────────
# TFLite INFERENCE — Mobil Deployment Pattern
# ─────────────────────────────────────────────────────────────

class TFLiteInference:
    """
    TFLite modeli ile inference.
    Mobil uygulamada (Python backend) veya test amacıyla kullanılır.

    Android/iOS native TFLite pattern'ı için alt kısımdaki
    pseudocode bölümüne bakın.
    """

    def __init__(self,
                 letter_tflite_path: str = None,
                 word_tflite_path: str = None,
                 letter_classes: list = None,
                 word_classes: list = None,
                 confidence_threshold: float = None):

        import tensorflow as tf
        self.tf = tf
        self.extractor = HandFeatureExtractor()
        self.threshold = confidence_threshold or cfg.CONFIDENCE_THRESHOLD

        self._letter_interp = self._load_tflite(letter_tflite_path)
        self._word_interp = self._load_tflite(word_tflite_path)
        self.letter_classes = letter_classes or []
        self.word_classes = word_classes or []

    def _load_tflite(self, path: str):
        if path and os.path.exists(path):
            interp = self.tf.lite.Interpreter(model_path=path)
            interp.allocate_tensors()
            print(f"✓ TFLite yüklendi: {path}")
            return interp
        return None

    def predict(self, image: np.ndarray, mode: str = 'letter') -> dict:
        """TFLite ile tahmin yap."""
        features = self.extractor.extract_features(image)
        if features is None:
            return {'label': 'El tespit edilemedi', 'confidence': 0.0,
                    'hand_detected': False, 'top3': [], 'mode': mode}

        interp = self._letter_interp if mode == 'letter' else self._word_interp
        classes = self.letter_classes if mode == 'letter' else self.word_classes

        if interp is None:
            return {'label': 'TFLite yüklenmedi', 'confidence': 0.0,
                    'hand_detected': True, 'top3': [], 'mode': mode}

        # Input/output details
        in_det = interp.get_input_details()[0]
        out_det = interp.get_output_details()[0]

        # Float16 quantization varsa dönüştür
        input_data = features.reshape(1, -1)
        if in_det['dtype'] == np.float16:
            input_data = input_data.astype(np.float16)
        else:
            input_data = input_data.astype(np.float32)

        interp.set_tensor(in_det['index'], input_data)
        interp.invoke()
        proba = interp.get_tensor(out_det['index'])[0].astype(np.float32)

        pred_idx = int(np.argmax(proba))
        confidence = float(proba[pred_idx])
        is_confident = confidence >= self.threshold
        label = classes[pred_idx] if is_confident else 'Belirsiz'

        top3_idx = np.argsort(proba)[::-1][:3]
        top3 = [(classes[i], float(proba[i])) for i in top3_idx
                if i < len(classes)]

        return {
            'label': label,
            'confidence': confidence,
            'is_confident': is_confident,
            'top3': top3,
            'hand_detected': True,
            'mode': mode
        }


# ─────────────────────────────────────────────────────────────
# GERÇEK ZAMANLI KAMERA DÖNGÜSÜ (PC Demo)
# ─────────────────────────────────────────────────────────────

class RealTimeRecognizer:
    """
    Webcam tabanlı demo.
    Mobil uygulamada kamera frame'i doğrudan predict()'e verilecek.
    """

    def __init__(self, predictor, frame_interval: float = None):
        self.predictor = predictor   # SignLanguageInference veya TFLiteInference
        self.interval = frame_interval or cfg.FRAME_INTERVAL_SEC
        self.mode = 'letter'

        import mediapipe as mp
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils

    def run(self, camera_index: int = 0):
        """
        Webcam'den sürekli okuyup tahmin yap.
        Tuşlar: 'l' = harf modu, 'w' = kelime modu, 'q' = çıkış
        """
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            print("HATA: Kamera açılamadı!")
            return

        import time
        last_pred_time = 0.0
        last_result = None

        print("\n[Gerçek Zamanlı Demo Başladı]")
        print("  'l' → Harf modu | 'w' → Kelime modu | 'q' → Çıkış\n")

        with self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.5
        ) as hands:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = cv2.flip(frame, 1)

                # Her frame'de landmark görselleştir
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                res = hands.process(rgb)
                if res.multi_hand_landmarks:
                    for lm in res.multi_hand_landmarks:
                        self.mp_drawing.draw_landmarks(
                            frame, lm, self.mp_hands.HAND_CONNECTIONS
                        )

                # Belirli aralıklarla tahmin yap
                now = time.time()
                if now - last_pred_time >= self.interval:
                    last_result = self.predictor.predict(frame, mode=self.mode)
                    last_pred_time = now
                    if last_result['hand_detected']:
                        print(f"  [{self.mode}] {last_result['label']} "
                              f"({last_result['confidence']:.1%})")

                self._draw_overlay(frame, last_result)
                cv2.imshow('TID Tanıma — ESC/q: çıkış', frame)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord('q'), 27):
                    break
                elif key == ord('l'):
                    self.mode = 'letter'
                    print("  → Mod: Harfler")
                elif key == ord('w'):
                    self.mode = 'word'
                    print("  → Mod: Kelimeler")

        cap.release()
        cv2.destroyAllWindows()

    def _draw_overlay(self, frame: np.ndarray, result: dict):
        if result is None:
            return
        h, w = frame.shape[:2]

        # Yarı saydam arka plan
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, h - 130), (w, h), (20, 20, 20), -1)
        cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

        mode_label = "HARF MODU" if self.mode == 'letter' else "KELİME MODU"
        cv2.putText(frame, mode_label, (12, h - 98),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

        label = result.get('label', '?')
        conf = result.get('confidence', 0.0)
        color = (0, 230, 60) if result.get('is_confident') else (0, 140, 255)

        cv2.putText(frame, label, (12, h - 55),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
        cv2.putText(frame, f"Guven: {conf:.1%}", (12, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Top-3 (sağ taraf)
        for rank, (cls, score) in enumerate(result.get('top3', [])[:3]):
            y_pos = h - 98 + rank * 25
            cv2.putText(frame, f"#{rank+1} {cls} {score:.0%}",
                        (w - 200, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (200, 200, 200), 1)
