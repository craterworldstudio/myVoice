from core.stage import Stage
import numpy as np

class Limiter(Stage):
    def __init__(self, threshold=1.0):
        super().__init__("limiter")
        self.threshold = threshold

    def process(self, audio):
        # Apply limiting to the audio signal
        np.clip(audio, -self.threshold, self.threshold, out=audio)
        #audio = min(self.threshold, max(audio, -self.threshold))
        return audio

    def set_threshold(self, threshold):
        self.threshold = threshold