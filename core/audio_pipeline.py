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
    
        return audio
    
    def print_stages(self):
        stages = []

        for stage in self.stages:
            status = '✓' if stage.enabled else '✗'
            stages.append(f"[{status}] {stage.name}")

        return stages
        pass # ✗ ✓