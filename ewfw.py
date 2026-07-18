import torch

print(torch.__version__)
print(torch.cuda.is_available())
print(torch.version.cuda)
print(torch.version.hip)
print(torch.cuda.get_device_name(0))


from backend.onnx_model import ONNX_Model

model = ONNX_Model("./models/vec-768-layer-12.onnx")
model.print_info()

model = ONNX_Model("./models/rmvpe.onnx")
model.print_info()