"""
=============================================================
feature_extraction.py — 2 EL DESTEKLİ (156-dim)
=============================================================
MediaPipe ile 2 el algılar, her el için 78-dim feature çıkarır.
Toplam: 156-dim = Sol(78) + Sağ(78)

Tek el algılandığında diğer elin feature'ları sıfır olur (padding).
"""

import os
import urllib.request
import numpy as np
import cv2

import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision

from config import cfg

# ── Model dosyası ──────────────────────────────────────────
MODEL_FILENAME = "hand_landmarker.task"
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)


def _ensure_model():
    if not os.path.exists(MODEL_FILENAME):
        print(f"  MediaPipe model indiriliyor: {MODEL_FILENAME} ...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        print(f"  Model indirildi ({os.path.getsize(MODEL_FILENAME) // 1024} KB)")
    return MODEL_FILENAME


class HandFeatureExtractor:
    """
    156 boyutlu feature vektörü çıkarır (2 el).

    [0:78]   → Sol el:  21 landmark × (x,y,z) + 15 açı
    [78:156] → Sağ el:  21 landmark × (x,y,z) + 15 açı

    Tek el algılandığında diğer elin feature'ları sıfır olur.
    """

    def __init__(self):
        model_path = _ensure_model()
        base_opts = mp_python.BaseOptions(model_asset_path=model_path)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_opts,
            num_hands=cfg.NUM_HANDS,          # 2 EL
            min_hand_detection_confidence=0.4,
            min_hand_presence_confidence=0.4,
            min_tracking_confidence=0.4,
            running_mode=mp_vision.RunningMode.IMAGE
        )
        self._detector = mp_vision.HandLandmarker.create_from_options(options)

    def extract_features(self, image: np.ndarray, bbox: tuple = None):
        """
        Görüntüden 156-dim feature çıkar.

        Returns:
            np.ndarray(156,) veya None (el bulunamazsa)
        """
        crop = self._crop_with_padding(image, bbox) if bbox is not None else image

        # BGR → RGB → MediaPipe Image
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        # Kırpılmış görüntüde el bulunamazsa tam görüntüyü dene
        if not result.hand_landmarks:
            if bbox is not None:
                rgb_full = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                mp_full = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_full)
                result = self._detector.detect(mp_full)
            if not result.hand_landmarks:
                return None

        # ── Sol ve sağ eli ayır ──────────────────────────
        left_landmarks = None
        right_landmarks = None

        for idx, landmarks in enumerate(result.hand_landmarks):
            if idx < len(result.handedness):
                handedness = result.handedness[idx][0].category_name
                if handedness == "Left":
                    left_landmarks = landmarks
                elif handedness == "Right":
                    right_landmarks = landmarks

        # Eğer handedness bilgisi yoksa ilk eli sol olarak ata
        if left_landmarks is None and right_landmarks is None:
            if len(result.hand_landmarks) >= 1:
                left_landmarks = result.hand_landmarks[0]
            if len(result.hand_landmarks) >= 2:
                right_landmarks = result.hand_landmarks[1]

        # ── Her el için 78-dim feature çıkar ─────────────
        left_features = self._extract_single_hand(left_landmarks) \
            if left_landmarks is not None else np.zeros(cfg.SINGLE_HAND_FEATURES, dtype=np.float32)

        right_features = self._extract_single_hand(right_landmarks) \
            if right_landmarks is not None else np.zeros(cfg.SINGLE_HAND_FEATURES, dtype=np.float32)

        # ── 156-dim: [Sol 78] + [Sağ 78] ────────────────
        return np.concatenate([left_features, right_features]).astype(np.float32)

    def _extract_single_hand(self, landmarks) -> np.ndarray:
        """Tek el için 78-dim feature çıkar."""
        normalized = self._normalize_landmarks(landmarks)
        coord_features = normalized.flatten()              # 63
        angle_features = self._compute_angles(normalized)  # 15
        return np.concatenate([coord_features, angle_features])  # 78

    def visualize_landmarks(self, image: np.ndarray) -> np.ndarray:
        """Landmark'ları görüntü üzerine çiz (debug)."""
        vis = image.copy()
        rgb = cv2.cvtColor(vis, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._detector.detect(mp_image)

        if result.hand_landmarks:
            h, w = vis.shape[:2]
            colors = [(0, 255, 0), (0, 0, 255)]  # Yeşil: sol, Kırmızı: sağ
            for hand_idx, hand_lms in enumerate(result.hand_landmarks):
                color = colors[hand_idx % 2]
                for lm in hand_lms:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(vis, (cx, cy), 4, color, -1)
        return vis

    def _crop_with_padding(self, image, bbox, padding=0.08):
        h, w = image.shape[:2]
        cx, cy, bw, bh = bbox
        x1 = max(0, int((cx - bw / 2 - padding) * w))
        y1 = max(0, int((cy - bh / 2 - padding) * h))
        x2 = min(w, int((cx + bw / 2 + padding) * w))
        y2 = min(h, int((cy + bh / 2 + padding) * h))
        if x2 <= x1 or y2 <= y1:
            return image
        return image[y1:y2, x1:x2]

    def _normalize_landmarks(self, landmarks) -> np.ndarray:
        coords = np.array([[lm.x, lm.y, lm.z] for lm in landmarks])
        coords -= coords[0].copy()   # bilek origin
        scale = np.linalg.norm(coords[9])
        if scale > 1e-6:
            coords /= scale
        return coords

    def _compute_angles(self, coords: np.ndarray) -> np.ndarray:
        angles = []
        for joint in cfg.FINGER_JOINTS:
            a = self._angle_between(coords[joint[0]], coords[joint[1]], coords[joint[2]])
            angles.append(a / 180.0)
        return np.array(angles, dtype=np.float32)

    @staticmethod
    def _angle_between(p1, p2, p3) -> float:
        v1, v2 = p1 - p2, p3 - p2
        n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
        if n1 < 1e-6 or n2 < 1e-6:
            return 0.0
        return float(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / (n1 * n2), -1, 1))))