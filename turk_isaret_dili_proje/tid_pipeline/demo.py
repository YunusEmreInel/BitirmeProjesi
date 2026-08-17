"""
demo.py — Kamera demo
Kullanım: python demo.py
          python demo.py --mode word
          python demo.py --camera 1
"""
import argparse
import os
from config import cfg
from inference import SignLanguageInference, RealTimeRecognizer, load_classes

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--mode',   choices=['letter','word'], default='letter')
    p.add_argument('--camera', type=int, default=0)
    args = p.parse_args()

    # Sınıfları yükle
    letter_classes = load_classes(cfg.LETTER_CLASSES_JSON)
    word_classes   = load_classes(cfg.WORD_CLASSES_JSON)

    print(f"Harf sınıfları  : {letter_classes}")
    print(f"Kelime sınıfları: {word_classes}")

    # Model yükle
    predictor = SignLanguageInference(
        letter_model_path=cfg.LETTER_MODEL_PATH,
        word_model_path=cfg.WORD_MODEL_PATH,
        letter_classes=letter_classes,
        word_classes=word_classes,
        confidence_threshold=cfg.CONFIDENCE_THRESHOLD
    )

    # Demo başlat
    demo = RealTimeRecognizer(predictor, frame_interval=cfg.FRAME_INTERVAL_SEC)
    demo.mode = args.mode
    demo.run(camera_index=args.camera)

if __name__ == '__main__':
    main()
