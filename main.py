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