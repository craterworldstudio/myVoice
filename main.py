import os
os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")

import time
from core.engine import Engine
from utils.logger import logger
from utils.config import config


import sys

def run_cli():
    logger.logging = config.system.logging
    logger.console_logging = config.system.console_logging
    logger.new()
    try:
        engine = Engine()
        engine.start()
        engine.wait()
    finally:
        logger.close()

def run_gui():
    logger.logging = config.system.logging
    logger.console_logging = config.system.console_logging
    logger.new()
    try:
        engine = Engine()
        from ui.main_window import MyVoiceGUI
        app = MyVoiceGUI(engine)
        app.mainloop()
    finally:
        logger.close()


def main():
    if "--gui" in sys.argv or "-g" in sys.argv:
        run_gui()
    else:
        run_cli()

if __name__ == "__main__":
    print("Starting myVoice Engine...")
    main()