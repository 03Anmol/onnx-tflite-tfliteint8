import onnx2tf
import os

os.makedirs("/home/anmol/Documents/change/test/tf_model", exist_ok=True)
onnx2tf.convert(
    input_onnx_file_path="/home/anmol/Documents/change/test/onnx_model/resnet18.onnx",
    output_folder_path="/home/anmol/Documents/change/test/tf_model/resnet18",
    output_signaturedefs=True,
    copy_onnx_input_output_names_to_tflite=True,
    non_verbose=True
)