import torch

from core.stage import Stage
from backend.loader import RVCLoader
from backend.infer import RVCInfer
from backend.feature_extractor import FeatureExtractor
from backend.rmvpe import RMVPE

from utils.config import config

class RVC(Stage):
    def __init__(self
                ):
        super().__init__("rvc")

        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        ) # Same for CUDA and ROCm
        
        self.path_pth = config.rvc.model
        self.index_pth = config.rvc.index
        self.loader = RVCLoader(self.path_pth, self.device)
        loaded = self.loader.load()

        self.model = loaded["net_g"]
        self.sr = loaded["sr"]
        self.version = loaded["version"]

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
        mel = self.rmvpe.mel(audio)
        #print("Mel Shape",mel.shape)
        hidden = self.rmvpe._infer(mel)
        #print("Hidden Shape",hidden.shape)

        phone = self.feature_extractor.extract(audio)
        #print("Phone SHape", phone.shape, phone.shape[1])
        pitch, pitchf = self.rmvpe.extract(audio)
        #print("Pitch Shape", pitch.shape)
        #print("Pitch Min Max Shape", pitch.min(), pitch.max())
        
        
        audio = self.inference.infer(
            phone,
            pitch,
            pitchf,
            sid=0
        )

        print("Audio Shape",audio.shape)
        

        return audio