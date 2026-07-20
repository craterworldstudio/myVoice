import numpy as np
from utils.config import config
from dsp.filters import LowPass
import math
import scipy as sp
import scipy.signal as signal
from core.stage import Stage


class NResampler(Stage):

    def __init__(self, csr, tsr, stage):
        super().__init__(f"presampler{stage}")
        """
        - csr : Current Sample Rate \n
        - tsr : Target Sample Rate \n
        """
        self.csr = csr
        self.tsr = tsr

        self.AALPF = LowPass(                                  #dont apply, its in the Resampler.
            cutoff=       config.pipeline.AALP_threshold, 
            sample_rate=  config.audio.sample_rate,
            anti_aliasing=True
            )     
    
    def process(self, audio): 

        if self.csr > self.tsr:
            
            audio = self.AALPF.process(audio)
            newsampled = np.zeros((int(len(audio)/(self.csr/self.tsr))), 1)
            #Downsample logic...
            sampledidx = 0
            for idx in range(0, len(audio), int((self.csr/self.tsr))):
                x = audio[idx, 0]

                newsampled[sampledidx, 0] = x

                sampledidx += 1
        
        elif self.csr < self.tsr:
            newsampled = np.zeros((int(len(audio)*(self.tsr/self.csr))), 1)
            #Not Implemented

        else:
            #CSR = TSR
            newsampled = audio


        return newsampled
    

class PResampler(Stage):
    def __init__(self, csr, tsr, stage, block_size):
        super().__init__(f"presampler{stage}")
        """
        - csr : Current Sample Rate \n
        - tsr : Target Sample Rate \n
        """
        self.csr = csr
        self.tsr = tsr

        gcd = math.gcd(self.csr, self.tsr)  

        self.down = self.csr//gcd
        self.up = self.tsr//gcd

        self.input_buffer  = np.empty((0, 1), dtype=np.float32)

        self.block_size = math.ceil(block_size * csr / tsr)
        print("Resampler Block Size:", self.block_size)
        #self.output_buffer = np.empty()

    def process(self, audio):

            #48000 -> 16000
            #csr   -> tsr
            #3     -> 1
            #down  -> up
        if self.up == self.down: return audio
        self.input_buffer = np.concatenate((self.input_buffer, audio), axis=0)

        #if len(self.input_buffer) >= self.block_size:
        inputBlock = self.input_buffer[:self.block_size]
        self.input_buffer = self.input_buffer[self.block_size: ]
        audio = signal.resample_poly(inputBlock, self.up, self.down, axis=0)
        #self.output_buffer = np.concatenate((self.output_buffer, audio), axis=0)
        
        #outputBlock = self.output_buffer[:len(audio)]
        #self.output_buffer = self.output_buffer[len(audio):]
        return audio
        
