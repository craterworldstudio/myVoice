from audio.ring_buffer import RingBuffer
from audio.input import AudioInput
from audio.output import AudioOutput
from audio.resampler import *
import threading, time

from core.scheduler import *
from core.audio_pipeline import *

from dsp.gain import Gain
from dsp.filters import *
from dsp.dynamic import *
from dsp.normalize import *

from dsp.radio_effect import RadioEffect
from dsp.formant_shifter import FormantShifter
from backend.rvc import RVC

from utils.logger import logger
from utils.config import config

class Engine:

    def __init__(self):
        """Loads the TOML config with the help of utils/config.py"""

        
        #self.Abuffer = RingBuffer(32)
        
        self.Ibuffer = RingBuffer(config.audio.Rbuffer_size)
        self.Obuffer = RingBuffer(config.audio.Rbuffer_size)
        

        self.shutdown_event = threading.Event()
        self._running = False

        self.audioPipeline = AudioPipeline(config.pipeline.stages)
        self.scheduler = Scheduler(
            input_buffer=self.Ibuffer,
            output_buffer=self.Obuffer,
            audio_pipeline=self.audioPipeline,
            shutdown_event=self.shutdown_event
        )

        self.mic = AudioInput(
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
        if self._running:
            return

        self.shutdown_event.clear()
        self.build_pipeline()

        # Clear ring buffers
        self.Ibuffer.clear()
        self.Obuffer.clear()

        # Reset audio playback buffer and jitter buffer state
        if hasattr(self, 'speaker'):
            self.speaker.reset()

        # Reset RVC buffers if initialized
        if hasattr(self, 'rvcbackend'):
            self.rvcbackend.reset_buffers()

        self._running = True
        self.scheduler.start()
        self.speaker.start()

        time.sleep(0.3)
        self.mic.start()
        logger.info("Engine & Mic Started....")

    def stop(self):
        if not self._running and self.shutdown_event.is_set():
            return
        
        self._running = False
        self.shutdown_event.set()

        try:
            self.mic.stop()
        except Exception as e:
            logger.error(f"Error stopping mic: {e}")

        try:
            self.scheduler.stop()
        except Exception as e:
            logger.error(f"Error stopping scheduler: {e}")

        try:
            self.speaker.stop()
        except Exception as e:
            logger.error(f"Error stopping speaker: {e}")

        logger.info("Engine stopped successfully.")

    def set_rvc_model(self, model_path):
        self.build_pipeline()
        if hasattr(self, 'rvcbackend'):
            self.rvcbackend.load_model(model_path)
            logger.info(f"Loaded RVC model: {model_path} (SR: {self.rvcbackend.sr} Hz)")


    def wait(self):

        
        try:
            
            while not self.shutdown_event.is_set():
                #time.sleep(0.1)
                pass



        except KeyboardInterrupt:

            logger.fatal("Stopping Threads.")
            self.stop()

    def get_stage(self, name):
        return self.stage_map.get(name)

    def reload_config(self):
        config.reload()
        if hasattr(self, 'gain'):
            self.gain.set_parameters(config.dsp.gain.gain)
        if hasattr(self, 'hpf'):
            self.hpf.set_parameters(config.dsp.filters.hpf.cutoff)
        if hasattr(self, 'lpf'):
            self.lpf.set_parameters(config.dsp.filters.lpf.cutoff)
        if hasattr(self, 'ng'):
            self.ng.set_parameters(
                config.dsp.noisegate.threshold,
                config.dsp.noisegate.attack,
                config.dsp.noisegate.release,
                config.audio.sample_rate
            )
        if hasattr(self, 'compressor'):
            self.compressor.set_parameters(
                config.dsp.compressor.threshold,
                config.dsp.compressor.ratio,
                config.dsp.compressor.attack,
                config.dsp.compressor.release,
                config.audio.sample_rate
            )
        if hasattr(self, 'limiter'):
            self.limiter.set_parameters(
                config.dsp.limiter.mode,
                config.dsp.limiter.threshold,
                config.dsp.limiter.drive
            )
        if hasattr(self, 'rvcbackend'):
            self.rvcbackend.set_parameters(pitch=getattr(config.rvc, "pitch", 0))

    def build_pipeline(self):
        if getattr(self, "_pipeline_built", False):
            return
        self._pipeline_built = True

        self.normalizerF = Normalizer(
            target_level= config.dsp.normalizer.threshold_f,
            mode=         config.dsp.normalizer.mode_f,
            pos=          "F"
            )
        self.normalizerB = Normalizer(
            target_level= config.dsp.normalizer.threshold_b,
            mode=         config.dsp.normalizer.mode_b,
            pos=          "B"
            )
        self.compressor = Compressor(
            threshold=  config.dsp.compressor.threshold,
            atk=        config.dsp.compressor.attack,
            rls=        config.dsp.compressor.release,
            ratiou=     config.dsp.compressor.ratio,
            samplerate= config.audio.sample_rate,
            blocksize=  config.audio.block_size,

            )
        self.gain = Gain(
            gain=         config.dsp.gain.gain
            )
        self.hpf = HighPass(
            cutoff=       config.dsp.filters.hpf.cutoff, 
            sample_rate=  config.audio.sample_rate
            )
        self.lpf = LowPass(
            cutoff=       config.dsp.filters.lpf.cutoff, 
            sample_rate=  config.audio.sample_rate,
            anti_aliasing=False
            )
        self.ng = NoiseGate(
            threshold=    config.dsp.noisegate.threshold,
            atk=          config.dsp.noisegate.attack,
            rls=          config.dsp.noisegate.release,
            samplerate=   config.audio.sample_rate,
            blocksize=    config.audio.block_size
        )
        self.limiter = Limiter(
            threshold=    config.dsp.limiter.threshold,
            drive=        config.dsp.limiter.drive,
            mode=         config.dsp.limiter.mode
        )
        self.radio = RadioEffect(sample_rate=config.audio.sample_rate)
        self.formant = FormantShifter(sample_rate=config.audio.sample_rate, pitch_semitones=4.2, formant_semitones=3.0)
        self.rvcbackend = RVC()

        # Add active stages to the pipeline (sorted by config.pipeline.stages order)
        self.audioPipeline.add_stage(self.gain)
        self.audioPipeline.add_stage(self.hpf)
        self.audioPipeline.add_stage(self.lpf)
        self.audioPipeline.add_stage(self.ng)
        self.audioPipeline.add_stage(self.compressor)
        self.audioPipeline.add_stage(self.radio)
        self.audioPipeline.add_stage(self.formant)
        self.audioPipeline.add_stage(self.rvcbackend)
        self.audioPipeline.add_stage(self.limiter)

        self.audioPipeline.enable_stage(self.gain)
        self.audioPipeline.enable_stage(self.hpf)
        self.audioPipeline.enable_stage(self.lpf)
        self.audioPipeline.enable_stage(self.ng)
        self.audioPipeline.enable_stage(self.compressor)
        self.audioPipeline.disable_stage(self.radio)
        self.audioPipeline.disable_stage(self.formant)
        self.audioPipeline.disable_stage(self.rvcbackend)
        self.audioPipeline.enable_stage(self.limiter)

        self.stage_map = {
            stage.name: stage
            for stage in self.audioPipeline.stages
        }

    def set_profile(self, profile_name):
        """
        Switch active voice profile:
        - 'alastor': RadioEffect ON, others OFF
        - 'female' / 'formant': FormantShifter ON (Pitch +4.2, Formant +3.0), others OFF
        - 'rvc': RVC ON (pitch=+7.0, xiaomu model), others OFF
        - 'bypass' / 'clean': Clean mic passthrough
        """
        self.build_pipeline()
        name = profile_name.lower().strip()
        if name == "alastor":
            self.radio.reset()
            self.audioPipeline.enable_stage(self.radio)
            self.audioPipeline.disable_stage(self.formant)
            self.audioPipeline.disable_stage(self.rvcbackend)
            logger.info("Engine Profile set to: ALASTOR (Demon Radio DSP)")
        elif name in ("female", "formant"):
            self.formant.reset()
            self.audioPipeline.disable_stage(self.radio)
            self.audioPipeline.enable_stage(self.formant)
            self.audioPipeline.disable_stage(self.rvcbackend)
            logger.info("Engine Profile set to: FEMALE (Real-Time Formant Shifter DSP)")
        elif name == "rvc":
            self.audioPipeline.disable_stage(self.radio)
            self.audioPipeline.disable_stage(self.formant)
            if "xiaomu" not in getattr(self.rvcbackend, "path_pth", ""):
                self.rvcbackend.load_model("./models/xiaomu/xiaomu.pth")
            self.rvcbackend.set_parameters(pitch=7.0)
            self.rvcbackend.reset_buffers()
            self.audioPipeline.enable_stage(self.rvcbackend)
            logger.info("Engine Profile set to: FEMALE (Neural RVC GPU)")
        else:
            self.audioPipeline.disable_stage(self.radio)
            self.audioPipeline.disable_stage(self.formant)
            self.audioPipeline.disable_stage(self.rvcbackend)
            logger.info("Engine Profile set to: BYPASS / CLEAN")



