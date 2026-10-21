import queue as qu
import numpy as np
from utils.logger import logger

class RingBuffer:

    def __init__(self, maxCSize: int = 32):
        self.maxCSize = maxCSize
        self.que = qu.Queue(maxsize=self.maxCSize)

    def read(self):
        try:
            return self.que.get_nowait()
        except qu.Empty:
            return None
        except Exception as e:
            logger.error(e)
            return None
        

    def write(self, audioChunk):
        try:
            self.que.put_nowait(audioChunk)
        except qu.Full:
            try:
                # Discard oldest chunk so buffer never lags behind live audio
                self.que.get_nowait()
            except qu.Empty:
                pass
            try:
                self.que.put_nowait(audioChunk)
            except Exception:
                pass
        except Exception as e:
            logger.error(e)

    def clear(self):
        while not self.que.empty():
            try:
                self.que.get_nowait()
            except qu.Empty:
                break
            except Exception:
                break