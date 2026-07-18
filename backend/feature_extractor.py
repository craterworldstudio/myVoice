import numpy as np
import torch
from backend.onnx_model import ONNX_Model


class FeatureExtractor(ONNX_Model):

    def __init__(self, model_path, device):
        super().__init__(model_path)
        self.device = device
        
        
    #@torch.no_grad()
    def extract(self, audio):
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        audio = audio[None, None, :]
        print("ContentVec Input:", audio.shape)
        output = self.session.run(
            self.output_names,
            {
                self.input_names[0]: audio
            }
        )[0]

        return output