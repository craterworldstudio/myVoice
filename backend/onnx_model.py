import numpy as np
import onnxruntime as ort
import torch
from torch.nn import functional as F

class ONNX_Model:
    def __init__(self, path):
        providers = []

        available = ort.get_available_providers()

        if "ROCMExecutionProvider" in available:
            providers.append("ROCMExecutionProvider")

        providers.append("CPUExecutionProvider")

        self.session = ort.InferenceSession(
            path,
            providers=providers
        )

        self.inputs = self.session.get_inputs()
        self.outputs = self.session.get_outputs()

        self.input_names = [i.name for i in self.inputs]
        self.output_names = [o.name for o in self.outputs]

    def print_info(self):

        print("Inputs")
    
        for inp in self.inputs:
            print(inp.name, inp.shape, inp.type)
    
        print()
    
        print("Outputs")
    
        for out in self.outputs:
            print(out.name, out.shape, out.type)

    def _infer(self, mel):

        n_frames = mel.shape[-1]

        pad = (32 - (n_frames % 32)) % 32

        if pad > 0:
            mel = F.pad(mel, (0, pad))

        mel = mel.cpu().numpy().astype(np.float32)

        outputs = self.session.run(
            self.output_names,
            {
                self.input_names[0]: mel
            }
        )

        hidden = outputs[0]

        return hidden[:, :n_frames]