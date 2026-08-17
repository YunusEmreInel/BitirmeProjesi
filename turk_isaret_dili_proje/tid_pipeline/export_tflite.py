"""
=============================================================
TFLite EXPORT — Mobil Deployment (2 EL - 156-dim)
=============================================================
Keras .h5 modellerini TFLite formatına dönüştürür.

Mobil Input/Output Formatı:
  Input  : float32[1, 156]  (1 örnek × 156 feature = 2 el × 78)
  Output : float32[1, N]    (N = sınıf sayısı, softmax çıkışı)
=============================================================
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras

from config import cfg


def convert_to_tflite(model_path: str,
                       tflite_path: str,
                       quantization: str = 'fp16') -> dict:
    if not os.path.exists(model_path):
        print(f"  HATA: Model bulunamadı: {model_path}")
        return {}

    print(f"\n{'=' * 55}")
    print(f"  TFLite Dönüşümü (2 EL)")
    print(f"{'=' * 55}")
    print(f"  Kaynak : {model_path}")
    print(f"  Hedef  : {tflite_path}")
    print(f"  Quant  : {quantization}")

    model = keras.models.load_model(model_path)

    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    if quantization == 'fp16':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_types = [tf.float16]
    elif quantization == 'int8':
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        print("  UYARI: INT8 için kalibrasyon verisi önerilir.")
    elif quantization == 'none':
        pass
    else:
        raise ValueError(f"Geçersiz quantization: {quantization}")

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(tflite_path) or '.', exist_ok=True)
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)

    size_kb = os.path.getsize(tflite_path) / 1024
    print(f"  Model boyutu: {size_kb:.1f} KB")

    info = _inspect_tflite(tflite_path)
    info['size_kb'] = size_kb
    info['quantization'] = quantization

    print(f"  Input  shape : {info['input_shape']}")
    print(f"  Output shape : {info['output_shape']}")
    print(f"  Dönüşüm BAŞARILI ✓")

    return info


def _inspect_tflite(tflite_path: str) -> dict:
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    return {
        'input_shape': list(in_det['shape']),
        'input_dtype': str(in_det['dtype']),
        'output_shape': list(out_det['shape']),
        'output_dtype': str(out_det['dtype']),
    }


def verify_tflite_with_random_input(tflite_path: str,
                                      input_dim: int = 156) -> np.ndarray:
    """156-dim rastgele girişle TFLite modelini test et."""
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()

    in_det = interp.get_input_details()[0]
    out_det = interp.get_output_details()[0]

    dummy_input = np.random.rand(1, input_dim).astype(np.float32)

    if in_det['dtype'] == np.float16:
        dummy_input = dummy_input.astype(np.float16)

    interp.set_tensor(in_det['index'], dummy_input)
    interp.invoke()
    output = interp.get_tensor(out_det['index'])

    print(f"  Test inference çıktısı shape: {output.shape}")
    print(f"  Olasılık toplamı: {output[0].sum():.4f} (≈1.0 olmalı)")
    return output[0]


def export_all(output_dir: str = None, quantization: str = 'fp16'):
    if output_dir is None:
        output_dir = cfg.OUTPUT_PATH

    results = {}

    # Harf modeli
    if os.path.exists(cfg.LETTER_MODEL_PATH):
        info = convert_to_tflite(
            cfg.LETTER_MODEL_PATH,
            cfg.LETTER_TFLITE_PATH,
            quantization=quantization
        )
        verify_tflite_with_random_input(cfg.LETTER_TFLITE_PATH, cfg.TOTAL_FEATURES)
        results['letter'] = info
    else:
        print(f"  Harf modeli bulunamadı: {cfg.LETTER_MODEL_PATH}")

    # Kelime modeli
    if os.path.exists(cfg.WORD_MODEL_PATH):
        info = convert_to_tflite(
            cfg.WORD_MODEL_PATH,
            cfg.WORD_TFLITE_PATH,
            quantization=quantization
        )
        verify_tflite_with_random_input(cfg.WORD_TFLITE_PATH, cfg.TOTAL_FEATURES)
        results['word'] = info
    else:
        print(f"  Kelime modeli bulunamadı: {cfg.WORD_MODEL_PATH}")

    # Özet JSON kaydet
    summary_path = f"{output_dir}/tflite_export_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Export özeti: {summary_path}")

    return results


if __name__ == '__main__':
    export_all(quantization='fp16')