"""
=============================================================
MODEL MİMARİSİ — MLP (Çok Katmanlı Algılayıcı) — 2 EL
=============================================================

Mimari:
  Input(156) → Dense(256)+BN+ReLU+Dropout
             → Dense(128)+BN+ReLU+Dropout
             → Dense(64)+BN+ReLU+Dropout
             → Dense(num_classes)+Softmax

156-dim input: Sol el (78) + Sağ el (78)
=============================================================
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from config import cfg


def build_mlp(input_dim: int, num_classes: int,
              dropout_rate: float = None) -> keras.Model:
    """
    İşaret dili MLP modeli oluştur.

    Args:
        input_dim   : Feature vektör boyutu (156 = 2 el × 78)
        num_classes : Çıkış sınıf sayısı
        dropout_rate: Dropout oranı (None → cfg'den alır)

    Returns:
        Compile edilmemiş Keras modeli
    """
    if dropout_rate is None:
        dropout_rate = cfg.DROPOUT_RATE

    inputs = keras.Input(shape=(input_dim,), name='landmark_features')

    x = layers.Dense(256, use_bias=False, name='dense_256')(inputs)
    x = layers.BatchNormalization(name='bn_256')(x)
    x = layers.ReLU(name='relu_256')(x)
    x = layers.Dropout(dropout_rate, name='drop_256')(x)

    x = layers.Dense(128, use_bias=False, name='dense_128')(x)
    x = layers.BatchNormalization(name='bn_128')(x)
    x = layers.ReLU(name='relu_128')(x)
    x = layers.Dropout(dropout_rate, name='drop_128')(x)

    x = layers.Dense(64, use_bias=False, name='dense_64')(x)
    x = layers.BatchNormalization(name='bn_64')(x)
    x = layers.ReLU(name='relu_64')(x)
    x = layers.Dropout(dropout_rate * 0.5, name='drop_64')(x)

    outputs = layers.Dense(num_classes, activation='softmax',
                            name='class_probs')(x)

    model = keras.Model(inputs, outputs, name='TID_MLP_2HANDS')
    return model


def compile_model(model: keras.Model,
                  learning_rate: float = None) -> keras.Model:
    if learning_rate is None:
        learning_rate = cfg.LEARNING_RATE

    model.compile(
        optimizer=keras.optimizers.Adam(
            learning_rate=learning_rate,
            clipnorm=1.0
        ),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model


def get_callbacks(model_save_path: str,
                  patience: int = None) -> list:
    if patience is None:
        patience = cfg.EARLY_STOPPING_PATIENCE

    csv_path = model_save_path.replace('.h5', '_history.csv')

    return [
        keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        ),
        keras.callbacks.ModelCheckpoint(
            filepath=model_save_path,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=patience // 2,
            min_lr=1e-6,
            verbose=1
        ),
        keras.callbacks.CSVLogger(csv_path, append=False),
    ]