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
        device = getattr(config.device, "output", None)
        if not device:
            try:
                devs = [d['name'] for d in sd.query_devices()]
                if "pulse" in devs:
                    device = "pulse"
                elif "default" in devs:
                    device = "default"
            except Exception:
                device = None

        self.stream = sd.OutputStream(
            device=device,
            samplerate=sample,
            blocksize=blockSize,
            channels=channels,
            callback=self.callback
        )
        self.playback_Buffer = np.empty((0, 1), dtype=np.float32)
        self.blockSize = blockSize
        self.prefill_target = max(blockSize * 4, int(0.24 * sample)) # ~240ms initial buffer
        self.prefilled = False
                      
    def callback(self, outdata, frames, time_info, status):
        if status:
            logger.info(status)

        # Drain all available chunks from the ring buffer into our playback buffer
        while True:
            chunk = self.buffer.read()
            if chunk is None:
                break
            chunk = np.asarray(chunk, dtype=np.float32)
            if chunk.ndim == 1:
                chunk = chunk.reshape(-1, 1)
            self.playback_Buffer = np.concatenate((self.playback_Buffer, chunk), axis=0)

        # Cap jitter buffer to avoid latency accumulation (max 0.45s)
        max_buffer = int(0.45 * 48000)
        if len(self.playback_Buffer) > max_buffer:
            self.playback_Buffer = self.playback_Buffer[-max_buffer:]

        # Wait for initial prefill on engine start to absorb GPU hop computation time smoothly
        if not self.prefilled:
            if len(self.playback_Buffer) >= self.prefill_target:
                self.prefilled = True
            else:
                outdata.fill(0)
                return

        # If we have enough samples, play them smoothly
        if len(self.playback_Buffer) >= frames:
            dataChunk = self.playback_Buffer[:frames]
            self.playback_Buffer = self.playback_Buffer[frames:]
            outdata[:, 0] = dataChunk[:, 0]
            outdata[:, 1] = dataChunk[:, 0]
        else:
            # Underflow: output what we have and fill the rest with smooth silence
            avail = len(self.playback_Buffer)
            if avail > 0:
                outdata[:avail, 0] = self.playback_Buffer[:, 0]
                outdata[:avail, 1] = self.playback_Buffer[:, 0]
                outdata[avail:].fill(0)
                self.playback_Buffer = np.empty((0, 1), dtype=np.float32)
            else:
                outdata.fill(0)
            # Do NOT reset self.prefilled = False so audio never enters a stutter freeze loop!




    def reset(self):
        self.playback_Buffer = np.empty((0, 1), dtype=np.float32)
        self.prefilled = False

    def start(self):
        self.stream.start()
    def stop(self):
        self.stream.stop()