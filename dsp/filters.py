import numpy as np

from core.stage import Stage
import math

#Band Pass/Stop-Reject Filter

class HighPass(Stage): 
    def __init__(self, cutoff, sample_rate):
        super().__init__("high_pass")
        self.cutoff = cutoff
        self.sample_rate = sample_rate

        self.previous_input  = 0.0  #np.zeros((1024, 1), dtype=np.float32)
        self.previous_output = 0.0  #np.zeros((1024, 1), dtype=np.float32)

        self.alpha = self.calculate_alpha()

    def process(self, audio):
        filtered = np.zeros_like(audio)
        # y[n]=α(y[n−1]+x[n]−x[n−1])
        for idx in range(len(audio)):
            x = audio[idx, 0]
            delta = x - self.previous_input
            y = self.alpha * (delta + self.previous_output)

            filtered[idx, 0] = y
            
            self.previous_input = x
            self.previous_output = y

        return filtered
    
    def set_cutoff(self, cutoff):
        self.cutoff = cutoff

        self.alpha = self.calculate_alpha()

    def calculate_alpha(self):
        dt = 1.0 / self.sample_rate
        rc = 1.0 / (2*math.pi*self.cutoff)
        return (rc/(rc+dt))
    
class LowPass(Stage):
    def __init__(self, cutoff, sample_rate):
        super().__init__("low_pass")
        self.cutoff = cutoff
        self.sample_rate = sample_rate

        self.previous_output = 0.0

        self.alpha = self.calculate_alpha()

    def process(self, audio):
        filtered = np.zeros_like(audio)

        # y[i]=α⋅x[i]+(1−α)⋅y[i−1]

        for idx in range(len(audio)):
            x = audio[idx, 0]
            y = self.alpha * x + (1 - self.alpha) * self.previous_output
            
            filtered[idx, 0] = y

            self.previous_output = y
               
        return filtered

    def set_cutoff(self, cutoff):
        self.cutoff = cutoff
        self.alpha = self.calculate_alpha()

    def calculate_alpha(self):
        rc = 1.0 / (2*math.pi*self.cutoff)
        dt = 1.0 / self.sample_rate
        return (dt/(rc+dt))
        