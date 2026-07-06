from core.stage import Stage
import numpy as np
import math

class Limiter(Stage):
    def __init__(self, threshold=1.0, drive = 0.6, mode=None):
        super().__init__("limiter")
        self.threshold = threshold
        self.drive = drive
        self.modes = ["Hard", "Soft"]
        self.mode = self.modes[0] if mode == None else mode

    def set_parameters(self, mode = None, threshold=None, drive=None):
        self.threshold = threshold if threshold is not None else self.threshold
        self.drive = drive if drive is not None else self.drive
        self.mode = self.modes[0] if mode == None else mode

    def process(self, audio):
        # Apply limiting to the audio signal

        if self.mode == self.modes[0]:
            np.clip(audio, -self.threshold, self.threshold, out=audio)
        else:
            """
                I need to mke a Soft Limiter, so here is the function defination,
                for x := small values, x(threshold)-x(current) = large,
                f(x) = x,
                for x := large valus, x(threshold)-x(current) = small,
                f(x) = atan(x)



                f(x) = (tanh((x/threshold) * drive)) * threshold
                where ±threshold determines the vertical ceiling and
                drive determines the softness.

            """
            filtered = np.zeros_like(audio)
            for sample in range(len(audio)):
                x = audio[sample, 0]

                y = math.tanh((x/self.threshold) * self.drive) * self.threshold

                filtered[sample, 0] = y
            
        #audio = min(self.threshold, max(audio, -self.threshold))
        return filtered

    def set_threshold(self, threshold):
        self.threshold = threshold


class NoiseGate(Stage):
    def __init__(self, threshold=1.0, rls = 0.015, atk = 0.03, samplerate=48000, blocksize=1024):
        super().__init__("noisegate")
        self.threshold = threshold
        self.attack = atk
        self.release = rls
        self.sample_rate = samplerate

        self.attack_counter = 0
        self.release_counter = 0
        
        self.gain = 1.0
        self.block_duration = blocksize/samplerate

        self.attack_blocks = max(1, math.ceil(self.attack / self.block_duration))
        self.release_blocks = max(1, math.ceil(self.release / self.block_duration))

        self.attack_step = 1/self.attack_blocks
        self.release_step = 1/self.release_blocks


    def set_threshold(self, threshold):
        self.threshold = threshold 

    def set_parameters(self, atk, rls, samplerate):
        self.attack = atk
        self.release = rls
        self.sample_rate = samplerate

        self.attack_counter = 0
        self.release_counter = 0

        self.attack_blocks = math.ceil(self.attack / self.block_duration)
        self.release_blocks = math.ceil(self.release / self.block_duration)

        self.attack_step = 1/self.attack_blocks
        self.release_step = 1/self.release_blocks


    def process(self, audio):
        rms = math.sqrt(np.mean(audio[:, 0] ** 2))
        
        if rms > self.threshold:
            self.attack_counter += 1
            self.release_counter = 0

        if self.attack_counter >= self.attack_blocks:
            self.gain = min(1.0, self.gain + self.attack_step)
            
        if rms < self.threshold:
            self.release_counter += 1
            self.attack_counter = 0

            

        if self.release_counter >= self.release_blocks:
            self.gain = max(0.0, self.gain - self.release_step)
        
        return audio * self.gain
        #return np.zeros_like(audio)