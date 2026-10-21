import math

import numpy as np

from core.stage import Stage

class Normalizer(Stage):
    def __init__(self, target_level=0.8, mode="peak", pos="" ):
        super().__init__(f"normalizer{pos}")
        self.target_level = target_level
        self.modes = ["peak", "rms", "lufs"]
        self.mode = mode if mode in self.modes else "peak"


    def process(self, audio):
        # Normalize the audio signal to the target level
        if self.mode == "peak":
            max_amplitude = np.max(np.abs(audio))
            if max_amplitude > 0:
                normalization_factor = self.target_level / max_amplitude
                audio *= normalization_factor
            return audio
        
        elif self.mode == "rms":
            rms = math.sqrt(np.mean(audio[:, 0] ** 2))
            if rms > 0:
                normalization_factor = self.target_level / rms
                audio *= normalization_factor
            return audio
        
        elif self.mode == "lufs":
            # Implement LUFS normalization here if needed
            # For now, we'll just return the audio unchanged
            return audio

    def set_parameters(self, target_level):
        self.target_level = target_level