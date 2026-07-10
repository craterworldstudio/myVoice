import numpy as np

from utils.logger import logger

class Stage:
    def __init__(self, name):
        self.name = name
        self.enabled = False

    def __call__(self, audio):
        if self.enabled:
            return self.process(audio)
        else:
            #logger.warning(f"Stage {self.name} is disabled.")
            return audio
        
    def initialize(self):
        pass

    def shutdown(self):
        pass

    def process(self, audio):
        logger.error("Subclasses must implement the process method.")


