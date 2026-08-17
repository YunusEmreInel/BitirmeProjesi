"""
=============================================================
TÜRK İŞARET DİLİ TANIMA - KONFİGÜRASYON (2 EL)
=============================================================
"""

import os


class Config:
    # ── VERİ SETİ YOLLARI ──────────────────────────────────
    DATASET1_PATH = r"C:\Users\YunusEmre\Desktop\modelim\Bitirme Projesi\turk_isaret_dili_proje\tid_pipeline\data\dataset1"
    DATASET2_PATH = r"C:\Users\YunusEmre\Desktop\modelim\Bitirme Projesi\turk_isaret_dili_proje\tid_pipeline\data\dataset2"
    DATASET2_YAML = r"C:\Users\YunusEmre\Desktop\modelim\Bitirme Projesi\turk_isaret_dili_proje\tid_pipeline\data\dataset2\data.yaml"

    PROCESSED_DATA_PATH = "data/processed"
    OUTPUT_PATH = "output"

    # ── TÜRKÇE HARFLER (TİD EL ALFABESİ) ──────────────────
    TURKISH_LETTERS = [
        'A', 'B', 'C', 'CH', 'D', 'E', 'F', 'G', 'GH',
        'H', 'I', 'IH', 'J', 'K', 'L', 'M', 'N', 'O',
        'OH', 'P', 'R', 'S', 'SH', 'T', 'U', 'UH', 'V', 'Y', 'Z'
    ]
    TURKISH_LETTERS_NATIVE = [
        'A', 'B', 'C', 'Ç', 'D', 'E', 'F', 'G', 'Ğ', 'H',
        'I', 'İ', 'J', 'K', 'L', 'M', 'N', 'O', 'Ö', 'P',
        'R', 'S', 'Ş', 'T', 'U', 'Ü', 'V', 'Y', 'Z'
    ]

    # ── TÜRKÇE KELİMELER ───────────────────────────────────
    TURKISH_WORDS = [
        "Anne", "Arkadas", "Baba", "Dur", "Ev", "Evet", "Hayir",
        "icmek", "iyi", "Kardes", "kotu", "Merhaba", "Nasil",
        "Nerede", "Ozur-Dilemek", "Tamam", "Telefon",
        "Tesekkurler", "Tuvalet", "Yemek"
    ]

    # ── MEDIAPIPE (2 EL) ──────────────────────────────────
    NUM_HANDS = 2
    NUM_LANDMARKS = 21
    COORDS_PER_LANDMARK = 3
    NUM_COORD_FEATURES = 63
    NUM_ANGLE_FEATURES = 15
    SINGLE_HAND_FEATURES = 78
    TOTAL_FEATURES = 156

    # Parmak eklem üçlüleri (açı hesabı için)
    FINGER_JOINTS = [
        [1, 2, 3], [2, 3, 4],
        [5, 6, 7], [6, 7, 8],
        [9, 10, 11], [10, 11, 12],
        [13, 14, 15], [14, 15, 16],
        [17, 18, 19], [18, 19, 20],
        [0, 5, 9], [0, 9, 13], [0, 13, 17],
        [5, 0, 17], [0, 5, 17],
    ]

    # ── MODEL HİPERPARAMETRELERİ ───────────────────────────
    BATCH_SIZE = 32
    EPOCHS = 120
    LEARNING_RATE = 0.001
    EARLY_STOPPING_PATIENCE = 15
    DROPOUT_RATE = 0.35
    VAL_SIZE = 0.15
    TEST_SIZE = 0.10

    # ── INFERENCE ──────────────────────────────────────────
    CONFIDENCE_THRESHOLD = 0.65
    FRAME_INTERVAL_SEC = 1.5

    # ── ÇIKTI DOSYALARI (2 EL) ─────────────────────────────
    LETTER_MODEL_PATH   = "output/letter_model_2hands.h5"
    WORD_MODEL_PATH     = "output/word_model_2hands.h5"
    LETTER_TFLITE_PATH  = "output/letter_model_2hands.tflite"
    WORD_TFLITE_PATH    = "output/word_model_2hands.tflite"
    LETTER_CLASSES_JSON = "output/letter_classes.json"
    WORD_CLASSES_JSON   = "output/word_classes.json"


cfg = Config()

os.makedirs(cfg.OUTPUT_PATH, exist_ok=True)
os.makedirs(cfg.PROCESSED_DATA_PATH, exist_ok=True)