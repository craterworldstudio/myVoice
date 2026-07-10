import sounddevice as sd
import numpy as np
from utils.logger import logger
from utils.config import config

class AudioOutput:

    def __init__(self,
                 buffer,
                 sample=48000,
                 channels=2,
                 blockSize=1024
                 ):
        
        self.buffer = buffer
        self.stream = sd.OutputStream(
            #device=9,
            samplerate=sample,
            blocksize=blockSize,
            channels=channels,
            callback=self.callback
            )
        self.playback_Buffer = np.empty((0, 1))
        self.blockSize = blockSize
                      
    def callback(self, outdata, frames, time_info, status):
        """
    	indata:    NumPy array containing the recorded audio chunk. \n
    	frames:    Integer showing the exact number of frames in indata (equals blocksize). \n
    	time_info: CFFI structure containing timestamp data. \n
    	status:    CallbackFlags object indicating errors like overflows.
    	"""

        if status: logger.info(status)

        dataChunk = self.buffer.read()
        
        if dataChunk is None:
            outdata.fill(0)
            return
        
        self.playback_Buffer = np.concatenate((self.playback_Buffer, dataChunk ), axis=0)

        if len(self.playback_Buffer) >= self.blockSize:
            dataChunk = self.playback_Buffer[:self.blockSize]
            self.playback_Buffer = self.playback_Buffer[self.blockSize:]
        else:
            outdata.fill(0)
            return

        
        
        outdata[:, 0] = dataChunk[:, 0] #L
        outdata[:, 1] = dataChunk[:, 0] #R


    def start(self):
        self.stream.start()
    def stop(self):
        self.stream.stop()