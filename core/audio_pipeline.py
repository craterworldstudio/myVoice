import numpy as np

from utils.logger import logger

class AudioPipeline:

    def __init__(self, order: list):
        self.stages = []

        self.order = {name: index for index, name in enumerate(order)}

    def add_stage(self, stage):
        if stage not in self.stages:
            self.stages.append(stage)
            self.stages.sort(key=lambda obj: self.order[obj.name])
            stage.initialize()
        else:
            logger.error(f"Stage {stage.name} already in pipeline.")


    def remove_stage(self, stage):
        if stage in self.stages:
            self.stages.remove(stage)
            self.stages.sort(key=lambda obj: self.order[obj.name])
            stage.shutdown()
        else:
            logger.error(f"Stage {stage.name} not found in pipeline.")

    def enable_stage(self, stage):
        if stage in self.stages:
            stage.enabled = True
            logger.info(f"Stage {stage.name} is Enabled.")
            
        else:
            logger.error(f"Stage {stage.name} not found in pipeline.")

    def disable_stage(self, stage):
        if stage in self.stages:
            stage.enabled = False
            logger.info(f"Stage {stage.name} is disabled.")

        else:
            logger.error(f"Stage {stage.name} not found in pipeline.")

    def process(self, audio):        
        #self.stages.sort(key=lambda obj: self.order[obj.name])
        
        for stage in self.stages:
            audio = stage(audio)
            #print(stage.name,
            #    audio.dtype,
            #    audio.shape,
            #    audio.min(),
            #    audio.max(), np.mean(audio))

            if audio is None:
                return None
            #print(stage.name, audio.shape)
        return audio
    
    def print_stages(self):
        stages = []

        for stage in self.stages:
            status = '✓' if stage.enabled else '✗'
            stages.append(f"[{status}] {stage.name}")

        logger.info(f"<=========== Current Stage Track Order ===========>")
        for stage in stages:
            logger.info(stage)
        #return stages
        #pass # ✗ ✓