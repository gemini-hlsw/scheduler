from __init__ import ProcessManager
from datetime import datetime, timedelta
import logging

from bin import SchedulerTask, BinConfig, BinType
from scheduler import Scheduler

from random import randint

def set_logging():
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    handler = logging.StreamHandler()
    handler.setLevel(logging.DEBUG)
    fmt = logging.Formatter("%(asctime)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)


if __name__ == "__main__":
    
    # TODO: This is going to be moved to a config file

    '''
    
    
    config = [BinConfig(BinType.STANDARD,
                        datetime(2020, 1, 1, 0, 0, 0),
                        timedelta(seconds=10),
                        timedelta(seconds=30)),
              BinConfig(BinType.STANDARD,
                        datetime(2020, 1, 1, 0, 0, 0),
                        timedelta(seconds=10),
                        timedelta(seconds=30)),
              BinConfig(BinType.STANDARD,
                        datetime(2020, 1, 1, 0, 0, 0),
                        timedelta(seconds=10),
                        timedelta(seconds=30)),
              BinConfig(BinType.REALTIME,
                        datetime(2020, 1, 1, 0, 0, 0),
                        timedelta(seconds=10),
                        timedelta(seconds=30))]
    '''

    config = [BinConfig(BinType.STANDARD,
                        datetime(2020, 1, 1, 0, 0, 0),
                        timedelta(seconds=10),
                        timedelta(seconds=30))]
    set_logging()
    manager = ProcessManager(config)
    scheduler = Scheduler(randint(3, 15))
    task = SchedulerTask(datetime(2020, 1, 1, 0, 0, 0), datetime(2020, 1, 1, 1, 0, 0), 2, False, scheduler)
    manager.add_task(task)
    manager.run()
