
"""import sounddevice as sd


from audio.ring_buffer import RingBuffer
from audio.input import AudioInput
from audio.output import AudioOutput


def main():
    buffer = RingBuffer(32)

    mic =     AudioInput(buffer)
    speaker = AudioOutput(buffer)

    mic.start()
    speaker.start()

    print("Mic Started....")

    try:
        while True:
            #time.sleep(0.05)
            pass

    except KeyboardInterrupt:

        print("Stopping Threads.")

    finally:
        mic.stop()
        speaker.stop()

"""

import time
from core.engine import Engine
from utils.logger import logger
from utils.config import config

def main():
    logger.console_logging = config.system.console_logging
    logger.new()
    engine = Engine()

    engine.start()

    engine.wait()

if __name__ == "__main__":
    print("Starting myVoice Engine...")
    main()