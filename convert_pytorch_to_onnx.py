import torch
import torch.onnx
import torchvision.models as models
import os

model = models.resnet18()
num_ftrs = model.fc.in_features
model.fc = torch.nn.Linear(num_ftrs, len(torch.load("/home/anmol/Documents/change/test/resnet18_best.pth")['fc.bias']))
model.load_state_dict(torch.load("/home/anmol/Documents/change/test/resnet18_best.pth"))
model.eval()

os.makedirs("/home/anmol/Documents/change/test/onnx_model", exist_ok=True)

dummy_input = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy_input, "/home/anmol/Documents/change/test/onnx_model/resnet18.onnx", 
                 export_params=True, opset_version=11, do_constant_folding=True, 
                 input_names=['input'], output_names=['output'])



