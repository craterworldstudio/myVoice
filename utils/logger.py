import os
import time, datetime

class Logger:

    PURPLE = "\033[95m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD_RED = "\033[1;91m"
    RESET = "\033[0m"

    # Map log levels to Dracula colors
    COLORS = {
        "INFO": GREEN,
        "WARNING": YELLOW,
        "ERROR": RED,
        "FATAL": BOLD_RED,
        "TERMINATE": PURPLE
    }



    def __init__(self, logdir=None, logging: bool = True, console_logging: bool = True):
        self.log_Dir = "./runtime/logs" if logdir in ("", None) else logdir
        self.opened = False
        self.c_file = None
        self.logging = logging
        self.console_logging = console_logging

        

        #if not os.path.isdir(self.log_Dir):
        #    os.mkdir(self.log_Dir)
        
        os.makedirs(self.log_Dir, exist_ok=True)

        

    def new(self):
        if self.opened:
            self.warning("Log file already in use.")
            return
        name = f"{datetime.date.today()}_{time.time()}"
        if self.logging:
            self.c_file = open(f"{self.log_Dir}/{name}.log", "a+")
            self.opened = True
        self.info("------------------------------Log File Starting------------------------------")

    def info(self, Iptinfo):
        self._write("INFO", Iptinfo)

    def warning(self, Iptwarning):
        self._write("WARNING", Iptwarning)

    def error(self, Ipterr):
        self._write("ERROR", Ipterr)

    def fatal(self, Iptftl):
        self._write("FATAL", Iptftl)
        self._write("TERMINATE", "Shutting down the application... ")
        

    def _write(self, level, msg):
        if not self.opened:
            return
            
        timestamp = datetime.datetime.now().strftime('%H:%M:%S')
        
        if self.logging:
            # 1. Plain text format for the actual .log file
            file_msg = f"[{timestamp}][{level}] {msg} \n"
            self.c_file.write(file_msg)
            self.c_file.flush()

        # 2. Colorized text format purely for the live terminal display
        if self.console_logging:
            color = self.COLORS.get(level, self.RESET)
            # Only wraps the level tag in color so the log message stays highly readable
            console_msg = f"[{timestamp}][{color}{level}{self.RESET}] {msg} \n"
            print(console_msg, end="")


    def close(self):
        self.info("------------------------------Log File Ending------------------------------")
        if self.logging:
            self.c_file.close()
            self.c_file = None
        self.opened = False

logger = Logger()