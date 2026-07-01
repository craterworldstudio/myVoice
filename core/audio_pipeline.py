from utils.logger import logger

class AudioPipeline:

    def __init__(self):
        self.stages = []

    def add_stage(self, stage):
        self.stages.append(stage)

    

    def process(self, audio):

        for stage in self.stages:
            audio = stage(audio)

        return audio