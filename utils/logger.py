import os
import time, datetime

class Logger:
    def __init__(self, logdir=None):
        self.log_Dir = "./runtime/logs" if logdir in ("", None) else logdir
        self.opened = False
        self.c_file = None

        

        #if not os.path.isdir(self.log_Dir):
        #    os.mkdir(self.log_Dir)
        
        os.makedirs(self.log_Dir, exist_ok=True)

        

    def new(self):
        if self.opened:
            self.warn("Log file already in use.")
            return
        name = f"{datetime.date.today()}_{time.time()}"
        self.c_file = open(f"{self.log_Dir}/{name}.log", "a+")
        self.opened = True
        self.info("------------------------------Log File Starting------------------------------")

    def info(self, Iptinfo):
        self._write("INFO", Iptinfo)

    def warn(self, Iptwarn):
        self._write("WARN", Iptwarn)

    def error(self, Ipterr):
        self._write("ERROR", Ipterr)

    def fatal(self, Iptftl):
        self._write("FATAL", Iptftl)
        self._write("TERMINATE", "Shutting down the application... ")
        

    def _write(self, level, msg):
        if not self.opened:
            return
        Rmsg = f"[{datetime.datetime.now().strftime('%H:%M:%S')}][{level}] {msg} \n"
        print(Rmsg, end="")
        self.c_file.write(Rmsg)
        self.c_file.flush()

    def close(self):
        self.info("------------------------------Log File Ending------------------------------")
        self.c_file.close()
        self.c_file = None
        self.opened = False

logger = Logger()