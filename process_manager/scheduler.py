import time
import logging
import signal


class Scheduler():
    def __init__(self, runtime):
        self.runtime = runtime

    def __call__(self):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        logging.info(f'Sleeping for {self.runtime}s')
        time.sleep(self.runtime)
    
