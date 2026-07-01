import sounddevice as sd
from audio.ring_buffer import RingBuffer
from audio.input import AudioInput
from audio.output import AudioOutput
import threading, time

class Engine:

    def __init__(self):
        """Loads the TOML config with the help of utils/config.py"""

        self.Abuffer = RingBuffer(32)

        self.mic =     AudioInput(self.Abuffer)
        self.speaker = AudioOutput(self.Abuffer)

        self.shutdown_event = threading.Event()
        

    def start(self):

        self.mic.start()
        self.speaker.start()
        print("Mic Started....")

    def stop(self):
        if self.shutdown_event.is_set():
            return
        
        self.shutdown_event.set()
        self.mic.stop()
        self.speaker.stop()


    def wait(self):
        
        try:
            
            self.shutdown_event.wait()

            #while True: time.sleep(0.05)

        except KeyboardInterrupt:

            print("Stopping Threads.")
            self.stop()
            
    
