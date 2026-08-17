"""
=============================================================
TÜRK İŞARET DİLİ TANIMA — ANA PIPELINE (2 EL)
=============================================================
Kullanım:
    python main.py                    # Tam pipeline
    python main.py --skip-extract     # Feature çıkarımı atla (pkl var)
    python main.py --mode letter      # Sadece harf modeli
    python main.py --mode word        # Sadece kelime modeli
    python main.py --demo             # Webcam demo
=============================================================
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd

from config import cfg
from dataset_loader_parallel import (
    load_dataset1_letters,
    load_dataset2_words,
    prepare_splits,
    save_classes,
    load_classes,
)
from trainer import train_mode, evaluate_model
from export_tflite import export_all


def parse_args():
    parser = argparse.ArgumentParser(
        description='TID Pipeline - Türk İşaret Dili Tanıma (2 EL)'
    )
    parser.add_argument(
        '--skip-extract', action='store_true',
        help='Feature çıkarımını atla (pkl dosyaları mevcut olmalı)'
    )
    parser.add_argument(
        '--mode', choices=['letter', 'word', 'both'], default='both',
        help='Hangi modeli eğit (varsayılan: both)'
    )
    parser.add_argument(
        '--no-export', action='store_true',
        help='TFLite export adımını atla'
    )
    parser.add_argument(
        '--demo', action='store_true',
        help='Webcam demosunu çalıştır (eğitimden sonra)'
    )
    parser.add_argument(
        '--epochs', type=int, default=cfg.EPOCHS,
        help=f'Epoch sayısı (varsayılan: {cfg.EPOCHS})'
    )
    return parser.parse_args()


def step1_load_data(skip_extract: bool, mode: str):
    """Veri setlerini yükle veya cache'den oku."""
    # ✅ 2 el cache dosyaları (eski 78-dim cache ile karışmasın)
    pkl_letters = f"{cfg.PROCESSED_DATA_PATH}/letters_2hands.pkl"
    pkl_words   = f"{cfg.PROCESSED_DATA_PATH}/words_2hands.pkl"

    df_letters = pd.DataFrame()
    df_words   = pd.DataFrame()

    if skip_extract and os.path.exists(pkl_letters):
        print(f"\n[Cache] Harf verisi okunuyor: {pkl_letters}")
        df_letters = pd.read_pickle(pkl_letters)
        print(f"  {len(df_letters)} kayıt yüklendi.")
    elif mode in ('letter', 'both'):
        df_letters = load_dataset1_letters(cfg.DATASET1_PATH)
        df_letters.to_pickle(pkl_letters)
        print(f"  Cache'e kaydedildi: {pkl_letters}")

    if skip_extract and os.path.exists(pkl_words):
        print(f"\n[Cache] Kelime verisi okunuyor: {pkl_words}")
        df_words = pd.read_pickle(pkl_words)
        print(f"  {len(df_words)} kayıt yüklendi.")
    elif mode in ('word', 'both'):
        df_words = load_dataset2_words(cfg.DATASET2_PATH, cfg.DATASET2_YAML)
        df_words.to_pickle(pkl_words)
        print(f"  Cache'e kaydedildi: {pkl_words}")

    return df_letters, df_words


def step2_split(df_letters, df_words, mode):
    print("\n" + "=" * 55)
    print("  [Adım 2] Veri Split (2 EL)")
    print("=" * 55)

    letter_data, le_letter = None, None
    word_data, le_word = None, None

    if mode in ('letter', 'both') and len(df_letters) > 0:
        letter_data, le_letter = prepare_splits(df_letters, 'Harfler')
        if le_letter:
            save_classes(le_letter.classes_, cfg.LETTER_CLASSES_JSON)

    if mode in ('word', 'both') and len(df_words) > 0:
        word_data, le_word = prepare_splits(df_words, 'Kelimeler')
        if le_word:
            save_classes(le_word.classes_, cfg.WORD_CLASSES_JSON)

    return letter_data, le_letter, word_data, le_word


def step3_train(letter_data, le_letter, word_data, le_word,
                mode, epochs):
    print("\n" + "=" * 55)
    print("  [Adım 3] Model Eğitimi (2 EL - 156 dim)")
    print("=" * 55)

    letter_model, word_model = None, None

    if mode in ('letter', 'both') and letter_data is not None:
        letter_model, _ = train_mode(
            letter_data, le_letter,
            'Harfler', cfg.LETTER_MODEL_PATH,
            epochs=epochs
        )

    if mode in ('word', 'both') and word_data is not None:
        word_model, _ = train_mode(
            word_data, le_word,
            'Kelimeler', cfg.WORD_MODEL_PATH,
            epochs=epochs
        )

    return letter_model, word_model


def step4_evaluate(letter_model, letter_data, le_letter,
                   word_model, word_data, le_word):
    print("\n" + "=" * 55)
    print("  [Adım 4] Değerlendirme (2 EL)")
    print("=" * 55)

    results = {}

    if letter_model and letter_data:
        results['letter'] = evaluate_model(
            letter_model, letter_data, le_letter,
            'Harfler', cfg.OUTPUT_PATH
        )

    if word_model and word_data:
        results['word'] = evaluate_model(
            word_model, word_data, le_word,
            'Kelimeler', cfg.OUTPUT_PATH
        )

    summary_path = f"{cfg.OUTPUT_PATH}/evaluation_summary.json"
    with open(summary_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\n  Değerlendirme özeti: {summary_path}")

    return results


def step5_export():
    print("\n" + "=" * 55)
    print("  [Adım 5] TFLite Export (2 EL)")
    print("=" * 55)
    return export_all(quantization='fp16')


def step6_demo():
    from inference import SignLanguageInference, RealTimeRecognizer

    letter_classes, word_classes = [], []
    if os.path.exists(cfg.LETTER_CLASSES_JSON):
        letter_classes = load_classes(cfg.LETTER_CLASSES_JSON)
    if os.path.exists(cfg.WORD_CLASSES_JSON):
        word_classes = load_classes(cfg.WORD_CLASSES_JSON)

    predictor = SignLanguageInference(
        letter_model_path=cfg.LETTER_MODEL_PATH,
        word_model_path=cfg.WORD_MODEL_PATH,
        letter_classes=letter_classes,
        word_classes=word_classes
    )
    demo = RealTimeRecognizer(predictor)
    demo.run()


def print_final_summary(eval_results, tflite_results):
    print("\n")
    print("╔" + "═" * 53 + "╗")
    print("║    PIPELINE TAMAMLANDI — ÖZET (2 EL)               ║")
    print("╠" + "═" * 53 + "╣")

    for mode_key in ('letter', 'word'):
        mode_name = 'Harfler' if mode_key == 'letter' else 'Kelimeler'
        if mode_key in eval_results:
            acc = eval_results[mode_key].get('test_accuracy', 0)
            f1  = eval_results[mode_key].get('weighted_f1', 0)
            print(f"║  {mode_name:12s} → Accuracy: {acc:.2%}  F1: {f1:.2%}       ║")
        if mode_key in tflite_results:
            kb = tflite_results[mode_key].get('size_kb', 0)
            q  = tflite_results[mode_key].get('quantization', '')
            print(f"║  {mode_name:12s} → TFLite ({q}): {kb:.0f} KB               ║")
        print("║" + "─" * 53 + "║")

    print("║  Çıktı dosyaları:                                   ║")
    for f in sorted(os.listdir(cfg.OUTPUT_PATH)):
        size = os.path.getsize(f"{cfg.OUTPUT_PATH}/{f}") / 1024
        print(f"║    {f:<30s} {size:>7.0f} KB  ║")
    print("╚" + "═" * 53 + "╝")


def main():
    args = parse_args()

    print("╔" + "═" * 53 + "╗")
    print("║   TÜRK İŞARET DİLİ TANIMA — 2 EL PIPELINE          ║")
    print("║   Yaklaşım: MediaPipe 2 El + MLP (156-dim)          ║")
    print("╚" + "═" * 53 + "╝")

    df_letters, df_words = step1_load_data(args.skip_extract, args.mode)

    letter_data, le_letter, word_data, le_word = step2_split(
        df_letters, df_words, args.mode
    )

    letter_model, word_model = step3_train(
        letter_data, le_letter, word_data, le_word,
        args.mode, args.epochs
    )

    eval_results = step4_evaluate(
        letter_model, letter_data, le_letter,
        word_model, word_data, le_word
    )

    tflite_results = {}
    if not args.no_export:
        tflite_results = step5_export()

    print_final_summary(eval_results, tflite_results)

    if args.demo:
        step6_demo()


if __name__ == '__main__':
    main()