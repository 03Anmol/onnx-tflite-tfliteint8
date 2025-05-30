import tensorflow as tf
import numpy as np
import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""  
def representative_dataset():
    for _ in range(100):
        yield [np.random.rand(1, 224, 224, 3).astype(np.float32)]

converter = tf.lite.TFLiteConverter.from_saved_model("/home/anmol/Documents/change/tf_model/resnet18")
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8, tf.lite.OpsSet.TFLITE_BUILTINS]
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
converter.experimental_new_converter = True
converter.allow_custom_ops = True

os.makedirs("/home/anmol/Documents/change/tflite_int8_model", exist_ok=True)
try:
    tflite_model = converter.convert()
    with open("/home/anmol/Documents/change/tflite_int8_model/resnet18_int8.tflite", "wb") as f:
        f.write(tflite_model)
except Exception as e:
    print(f"Full int8 quantization failed: {e}")
    print("Falling back to hybrid quantization...")
    converter = tf.lite.TFLiteConverter.from_saved_model("/home/anmol/Documents/change/tf_model/resnet18")
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
    converter.experimental_new_converter = True
    converter.allow_custom_ops = True
    tflite_model = converter.convert()
    with open("/home/anmol/Documents/change/tflite_int8_model/resnet18_hybrid.tflite", "wb") as f:
        f.write(tflite_model)
    