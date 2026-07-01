import sounddevice as sd
from audio.ring_buffer import RingBuffer
from audio.input import AudioInput
from audio.output import AudioOutput
import threading, time

from core.scheduler import Scheduler
from core.audio_pipeline import AudioPipeline

from utils.logger import logger

class Engine:

    def __init__(self):
        """Loads the TOML config with the help of utils/config.py"""

        
        #self.Abuffer = RingBuffer(32)
        
        self.Ibuffer = RingBuffer(32)
        self.Obuffer = RingBuffer(32)
        

        self.shutdown_event = threading.Event()

        self.audioPipeline = AudioPipeline()
        self.scheduler = Scheduler(
            input_buffer=self.Ibuffer,
            output_buffer=self.Obuffer,
            audio_pipeline=self.audioPipeline,
            shutdown_event=self.shutdown_event
        )

        self.mic =     AudioInput(self.Ibuffer)
        self.speaker = AudioOutput(self.Obuffer)

        
        

    def start(self):
        self.scheduler.start()
        
        self.speaker.start()
        self.mic.start()
        logger.info("Mic Started....")

    def stop(self):
        if self.shutdown_event.is_set():
            return
        
        self.shutdown_event.set()
        self.mic.stop()
        self.scheduler.stop()
        self.speaker.stop()
        logger.close()


    def wait(self):
        
        try:
            
            self.shutdown_event.wait()

            #while True: time.sleep(0.05)

        except KeyboardInterrupt:

            logger.fatal("Stopping Threads.")
            self.stop()
            
    
