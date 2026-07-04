import sounddevice as sd
from audio.ring_buffer import RingBuffer
from audio.input import AudioInput
from audio.output import AudioOutput
import threading, time

from core.scheduler import Scheduler
from core.audio_pipeline import AudioPipeline

from dsp.gain import Gain
from dsp.filters import HighPass, LowPass
from dsp.limiter import Limiter

from utils.logger import logger
from utils.config import config

class Engine:

    def __init__(self):
        """Loads the TOML config with the help of utils/config.py"""

        
        #self.Abuffer = RingBuffer(32)
        
        self.Ibuffer = RingBuffer(config.audio.Rbuffer_size)
        self.Obuffer = RingBuffer(config.audio.Rbuffer_size)
        

        self.shutdown_event = threading.Event()

        self.audioPipeline = AudioPipeline(config.pipeline.stages)
        self.scheduler = Scheduler(
            input_buffer=self.Ibuffer,
            output_buffer=self.Obuffer,
            audio_pipeline=self.audioPipeline,
            shutdown_event=self.shutdown_event
        )

        self.mic =     AudioInput(
            self.Ibuffer, 
            config.audio.sample_rate, 
            config.device.channelInput,
            config.audio.block_size
            )
        self.speaker = AudioOutput(
            self.Obuffer,
            config.audio.sample_rate, 
            config.device.channelOutput,
            config.audio.block_size,
            )

        
        

    def start(self):

        self.build_pipeline()
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
            
            while not self.shutdown_event.is_set():
                #time.sleep(0.1)
                pass



        except KeyboardInterrupt:

            logger.fatal("Stopping Threads.")
            self.stop()
    

    def build_pipeline(self):

        self.gain = Gain(
            gain=config.pipeline.gain
            )
        self.hpf = HighPass(
            cutoff=config.pipeline.hpf_cutoff, 
            sample_rate=config.audio.sample_rate
            )
        self.lpf = LowPass(
            cutoff=config.pipeline.lpf_cutoff, 
            sample_rate=config.audio.sample_rate
            )
        self.limiter = Limiter(
            threshold=config.pipeline.limiter
        )



        self.audioPipeline.add_stage(self.gain)
        self.audioPipeline.add_stage(self.hpf)
        self.audioPipeline.add_stage(self.lpf)
        self.audioPipeline.add_stage(self.limiter)



        self.audioPipeline.enable_stage(self.gain)
        self.audioPipeline.enable_stage(self.hpf)
        self.audioPipeline.enable_stage(self.lpf)
        self.audioPipeline.enable_stage(self.limiter)

    
