import numpy as np
import torch
from backend.onnx_model import ONNX_Model


class FeatureExtractor(ONNX_Model):

    def __init__(self, model_path, device):
        super().__init__(model_path)
        self.device = device
        
        
    def extract(self, audio):
        audio = np.asarray(audio, dtype=np.float32)

        if audio.ndim == 2 and audio.shape[1] == 1:
            audio = audio.squeeze(1)

        # Pad symmetrically by 160 samples on both ends so ContentVec convolutional
        # receptive field outputs exact time-aligned frames (100 Hz / 10ms per frame)
        audio = np.pad(audio, (160, 160), mode="reflect")

        audio = audio[None, None, :]
        output = self.session.run(
            self.output_names,
            {
                self.input_names[0]: audio
            }
        )[0]
        output = np.repeat(output, 2, axis=1)

        return output