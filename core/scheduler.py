import threading, time
from utils.logger import logger

class Scheduler:
    def __init__(self,
                input_buffer,
                output_buffer,
                audio_pipeline,
                shutdown_event
                ):
        self.input_buffer = input_buffer
        self.output_buffer = output_buffer
        self.audio_pipeline = audio_pipeline
        self.shutdown_event = shutdown_event
        self.thread = None

    def start(self):
        self.thread = threading.Thread(
            target=self.run,
            name="AudioScheduler",
            daemon=True
        )
        logger.info("Launching Scheduler Thread...")
        self.thread.start()

    def stop(self):
        self.thread.join()

    def run(self):
        logger.info("Scheduler Running.")
        while not self.shutdown_event.is_set():
            
            audio = self.input_buffer.read()

            if audio is None:
                time.sleep(0.001)
                continue
            

            paudio = self.audio_pipeline.process(audio)

            self.output_buffer.write(paudio)
            

        logger.info("Scheduler stopped.")    