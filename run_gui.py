import os
os.environ.setdefault("MIOPEN_LOG_LEVEL", "4")

from core.engine import Engine
from utils.logger import logger
from utils.config import config
from ui.main_window import MyVoiceGUI

def main():
    logger.logging = config.system.logging
    logger.console_logging = config.system.console_logging
    logger.new()
    engine = Engine()
    app = MyVoiceGUI(engine)
    app.mainloop()

if __name__ == "__main__":
    main()
