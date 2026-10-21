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

    def get_parameters(self):

        return {
            "threshold": self.threshold,
            "mode": self.mode,
            "drive": self.drive
        }

    def process(self, audio):
        # Apply limiting to the audio signal

        if self.mode == self.modes[0]:
            np.clip(audio, -self.threshold, self.threshold, out=audio)
            #audio = min(self.threshold, max(audio, -self.threshold))
            return audio
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
            filtered = np.tanh(
                (audio/self.threshold) * self.drive
            ) * self.threshold
            
           
            return filtered
        
    def set_threshold(self, threshold):
        self.threshold = threshold


class NoiseGate(Stage):
    def __init__(self, threshold=0.035, rls = 0.125, atk = 0.015, samplerate=48000, blocksize=1024):
        super().__init__("noisegate")
        self.threshold = threshold
        self.attack = atk
        self.release = rls
        self.sample_rate = samplerate

        self.attack_counter = 0
        self.release_counter = 0
        
        self.gain = 1.0
        self.previous_gain = 1.0
        self.block_duration = blocksize / samplerate

        self.recompute_steps()

    def recompute_steps(self):
        self.attack_blocks = max(1, math.ceil(self.attack / self.block_duration))
        self.release_blocks = max(1, math.ceil(self.release / self.block_duration))
        self.attack_step = 1.0 / self.attack_blocks
        self.release_step = 1.0 / self.release_blocks

    def set_parameters(self, threshold=None, atk=None, rls=None, samplerate=None):
        if threshold is not None: self.threshold = threshold
        if atk is not None: self.attack = atk
        if rls is not None: self.release = rls
        if samplerate is not None:
            self.sample_rate = samplerate
            self.block_duration = getattr(config.audio, "block_size", 960) / self.sample_rate

        self.attack_counter = 0
        self.release_counter = 0
        self.recompute_steps()

    def process(self, audio):
        rms = math.sqrt(np.mean(audio[:, 0] ** 2))
        
        if rms >= self.threshold:
            self.release_counter = 0
            self.gain = 1.0
        else:
            self.release_counter += 1
            if self.release_counter >= self.release_blocks:
                self.gain = max(0.0, self.gain - self.release_step)
        
        gain_curve = np.linspace(self.previous_gain, self.gain, len(audio), dtype=audio.dtype)

        audio[:, 0] *= gain_curve
        self.previous_gain = self.gain

        return audio


class Compressor(Stage):
    def __init__(self, threshold, atk, rls, ratiou, samplerate=48000, blocksize=1024):
        super().__init__("Compressor")

        self.threshold = threshold
        self.attack = atk
        self.release = rls

        self.gain = 1.0
        self.previous_gain = 1.0
        self.ratiou = ratiou

        self.block_duration = blocksize/samplerate


        self.attack_blocks = math.ceil(self.attack / self.block_duration)
        self.release_blocks = math.ceil(self.release / self.block_duration)

        self.attack_step = 1/self.attack_blocks
        self.release_step = 1/self.release_blocks


    def set_parameters(self,threshold, ratiou, atk, rls, samplerate):
        self.threshold = threshold
        self.ratiou = ratiou
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

        if rms <= self.threshold:
            ngain = 1.0

        else:
            comped_lvl = self.threshold + ((rms-self.threshold)/self.ratiou)

            ngain = comped_lvl/rms 

        if ngain < self.gain:
            
            difference = self.gain - ngain
            self.gain -= difference * self.attack_step

            if self.gain < ngain:
                self.gain = ngain

        else:
            
            difference = ngain - self.gain 
            self.gain += difference * self.release_step


            if self.gain > ngain:
                self.gain = ngain
        
        gain_curve = np.linspace(self.previous_gain, self.gain, len(audio), dtype=audio.dtype)
        audio[:, 0] *= gain_curve

        self.previous_gain = self.gain

        return audio
    
