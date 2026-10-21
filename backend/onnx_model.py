import os
import sys
import ctypes
import numpy as np
import onnxruntime as ort
import torch
from torch.nn import functional as F
from utils.logger import logger

# Suppress harmless MIOpen HIP kernel fallback warnings
os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")

_ROCM_LIBS_PRELOADED = False

def preload_rocm_libs():
    global _ROCM_LIBS_PRELOADED
    if _ROCM_LIBS_PRELOADED:
        return
    try:
        torch_lib_dir = os.path.join(os.path.dirname(torch.__file__), "lib")
        for lib in ["libhipfft.so", "libMIOpen.so"]:
            lib_path = os.path.join(torch_lib_dir, lib)
            if os.path.exists(lib_path):
                ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
        _ROCM_LIBS_PRELOADED = True
    except Exception as e:
        logger.warning(f"Could not preload ROCm libraries: {e}")

class ONNX_Model:
    def __init__(self, path):
        preload_rocm_libs()

        available = ort.get_available_providers()
        providers = []

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