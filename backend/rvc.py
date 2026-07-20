import sys

import numpy as np
import torch

from core.stage import Stage
from backend.loader import RVCLoader
from backend.infer import RVCInfer
from backend.feature_extractor import FeatureExtractor
from backend.rmvpe import RMVPE

import soundfile as sf

from utils.config import config

class RVC(Stage):
    def __init__(self
                ):
        super().__init__("rvc")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) # Same for CUDA and ROCm

        self.history = np.empty((0, 1), dtype=np.float32)
        self.WINDOW = 16000

        self.path_pth = config.rvc.model
        self.index_pth = config.rvc.index
        self.loader = RVCLoader(self.path_pth, self.device)
        loaded = self.loader.load()

        self.model = loaded["net_g"]
        self.sr = loaded["sr"]
        self.version = loaded["version"]
        self.f0 = loaded["f0"]

        self.feature_extractor = FeatureExtractor(config.rvc.vec, self.device)
        #self.feature_extractor.print_info()
        if config.rvc.use_onnx:
            self.rmvpe_path = config.rvc.rmvpe_onnx
        else:
            self.rmvpe_path = config.rvc.rmvpe_pt

        self.rmvpe = RMVPE(self.rmvpe_path)
        #self.rmvpe.print_info()

        self.inference = RVCInfer(self.model, self.device)

        

        
        

    def process(self, audio):
        #mel = self.rmvpe.mel(audio)
        #print("Mel Shape",mel.shape)
        #hidden = self.rmvpe._infer(mel)
        #print("Hidden Shape",hidden.shape)
        
        #print(audio.min(), audio.max())
        audio = np.asarray(audio, dtype=np.float32).reshape(-1, 1)
        #print(audio.min(), audio.max())
        self.history = np.concatenate(
            (self.history, audio),
            axis=0
        )

        if len(self.history) > self.WINDOW:
            self.history = self.history[-self.WINDOW:]

        if len(self.history) < self.WINDOW:
            return np.zeros_like(audio)
        print("Inferencing....")
        sf.write(
            "history.wav",
            self.history.squeeze(),
            16000
        )

        phone = self.feature_extractor.extract(self.history)
        #print("Phone SHape", phone.shape, phone.shape[1])
        #print(self.history.shape)
        #print(len(self.history) / 16000)
        

        #print(
        #    "Feature:",
        #    phone.min().item(),
        #    phone.max().item(),
        #    phone.mean().item(),
        #    phone.std().item(),
        #)
        

        #print("VERSION: ",self.version)
        #print("SAMPLERATE: ",self.sr)
        #print("F0: ",self.f0)
        pitch, pitchf = self.rmvpe.extract(self.history)
        #print("Pitch Min Max Shape", pitch.min(), pitch.max())
        #
        print("PHONE :", phone.shape, phone.min(), phone.max(), phone.mean())
        print("PITCH :", pitch.shape, pitch.min(), pitch.max(), pitch.mean())
        print("PITCHF:", pitchf.shape, pitchf.min(), pitchf.max(), pitchf.mean())
    
        audiop = self.inference.infer(
            phone,
            pitch,
            pitchf,
            sid=0
        )

        #print("Audio Shape",audiop.shape)

        sf.write(
            "output.wav",
            audiop,
            self.sr
        )
        #print(audiop.shape)
        #print(audiop.min(), audiop.max(), audiop.mean())
        audio = np.pad(
            audio,
            ((0, self.sr - len(audio)), (0, 0)),
            mode="constant"
        )
        return audio
        #return audiop[-len(audio):].reshape(-1, 1)