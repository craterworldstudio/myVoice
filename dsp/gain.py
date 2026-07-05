from core.stage import Stage

class Gain(Stage):
    def __init__(self, gain=1.0):
        super().__init__("gain")
        self.gain = gain

    def process(self, audio):
         audio *= self.gain
         return audio
    
    def set_gain(self, gain):
        self.gain = gain

    def step_gain(self, audio, gain):
        audio *= gain
        return audio