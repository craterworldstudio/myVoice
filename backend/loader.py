import torch
from models.synthesizer import SynthesizerTrn

class RVCLoader:

    def __init__(self, path_pth, device):
        self.path_pth = path_pth
        self.model = None
        self.device = device

    def load(self):        

        self.model = torch.load(self.path_pth, map_location=self.device, weights_only=False)

        
        self.config  = self.model["config"]
        self.weights = self.model["weight"]
        self.sr      = self.model["sr"]
        self.f0      = self.model["f0"]
        self.version = self.model["version"]
        self.vocoder = self.model["vocoder"]

        #print(self.model.keys(), self.config, self.weights.keys(), self.version, self.f0, self.sr)
        #print(self.vocoder)
        #print("Checkpoint Keys.")
        #for i in self.model.keys():
        #    print(i)

        #print(self.weights.keys())
        print(self.model["speakers_id"])
#
        #print(self.config)
        self.net_g = SynthesizerTrn(*self.config)
        missing, unexpected = self.net_g.load_state_dict(self.weights, strict=True)
        print("Missing:", missing)
        print("Unexpected:", unexpected)
        self.net_g.eval()

        self.net_g.to(self.device)

        #print("config =", self.config)
        #print("sr =", self.sr)
        #print("f0 =", self.f0)
        #print("version =", self.version)
        #print("vocoder =", self.vocoder)


        


        return {
            "net_g": self.net_g,
            "sr": self.sr,
            "f0": self.f0,
            "version": self.version,
            "vocoder": self.vocoder,
        }