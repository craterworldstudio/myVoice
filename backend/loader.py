import torch

class RVCLoader:

    def __init__(self, path_pth):
        self.path_pth = path_pth
        self.model = None

    def load(self):

        

        self.model = torch.load(self.path_pth, map_location='cpu', weights_only=False)

        self.config  = self.model["config"]
        self.weights = self.model["weight"]
        self.sr      = self.model["sr"]
        self.f0      = self.model["f0"]
        self.version = self.model["version"]
        self.vocoder = self.model["vocoder"]

        """
        modules = sorted({
            name.split(".")[0]
            for name in self.weights.keys()
        })
        print(modules)
        print(self.weights["enc_p.encoder.attn_layers.0.emb_rel_k"].shape)
        print(self.weights["enc_p.encoder.attn_layers.0.emb_rel_v"].shape)
        """"""
        for key in self.weights:
            if "enc_p.encoder.norm_layers_1.0" in key:
                print(key, self.weights[key].shape)


            if key.startswith("flow."):
                print(key)
        """
        return self.model