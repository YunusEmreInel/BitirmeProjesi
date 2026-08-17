import tensorflow as tf
import numpy as np

print(f"TF: {tf.__version__}")

model = tf.keras.models.load_model("output/letter_model_2hands.h5")
print(f"Model: {model.input_shape} → {model.output_shape}")

# Concrete function ile çevir
run_model = tf.function(lambda x: model(x, training=False))
concrete_func = run_model.get_concrete_function(
    tf.TensorSpec([1, 156], tf.float32)
)

converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]

tflite_model = converter.convert()

with open("output/letter_model_2hands.tflite", "wb") as f:
    f.write(tflite_model)

size_kb = len(tflite_model) / 1024
print(f"✅ Kaydedildi: {size_kb:.1f} KB")

# Test
interp = tf.lite.Interpreter(model_content=tflite_model)
interp.allocate_tensors()
inp = interp.get_input_details()[0]
out = interp.get_output_details()[0]
print(f"Input:  {inp['shape']}")
print(f"Output: {out['shape']}")

dummy = np.random.rand(1, 156).astype(np.float32)
interp.set_tensor(inp['index'], dummy)
interp.invoke()
result = interp.get_tensor(out['index'])
print(f"Test: toplam={result[0].sum():.4f}")
print("✅ Android'e kopyalanabilir!")