import queue as qu
import numpy as np

class RingBuffer:

    def __init__(self, maxCSize: int = 32):
        self.maxCSize = maxCSize
        self.que = qu.Queue(maxsize=self.maxCSize)

    def read(self):
        try:
            return self.que.get_nowait()
        except qu.Empty:
            return None
        except:
            return None
        

    def write(self, audioChunk):
        try:
            return self.que.put_nowait(audioChunk)
        except qu.Empty:
            return None
        
        except qu.Full:
            print("[!] Buffer is Full!")
            return None
        
        except:
            return None