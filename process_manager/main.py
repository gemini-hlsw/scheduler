
import argparse
import logging
import asyncio
import signal
from datetime import datetime, timedelta
from random import random, randint
from task import DEFAULT_TIMEOUT, SchedulerTask, TaskType
from scheduler import Scheduler
from manager import ProcessManager

# To mock dates
START_DATE = datetime(2020, 1, 1, 0, 0)
END_DATE = datetime(2021, 1, 1, 0, 0)

DEFAULT_PERIOD = 5
DEFAULT_POOL_SIZE = 10

def set_logging():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s: %(message)s")

def parse_cmdline():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mode", dest='mode', type=str, default="standard",
                        help="Mode of operation: standard or realtime")
    parser.add_argument('-p', dest='period', type=float, default=DEFAULT_PERIOD,
                        help=f"period between scheduling new jobs. Default: {DEFAULT_PERIOD}s")
    parser.add_argument('-s', dest='size', type=int, default=DEFAULT_POOL_SIZE,
                        help=f"maximum number of concurrent tasks. Default: {DEFAULT_POOL_SIZE}")
    parser.add_argument('-t', dest='timeout', type=float, default=DEFAULT_TIMEOUT,
                        help=f"timeout for the jobs. Default: {DEFAULT_TIMEOUT}s")

    return parser.parse_args()

# TODO: Period value in the args should be a random value between min and max if we want
# to mimic a subscription scenerio
def get_new_task(timeout: int):
    random_date = random() * (END_DATE - START_DATE) + START_DATE
    return SchedulerTask(random_date,
                         random_date + timedelta(days=1),
                         timeout=timeout,
                         target=Scheduler(randint(3, 15)))

async def main(args):
    done = asyncio.Event()
    manager = ProcessManager(args.size, args.timeout)

    def shutdown():
        done.set()
        manager.shutdown()
        asyncio.get_event_loop().stop()

    asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown)

    while not done.is_set():
        task = get_new_task(args.timeout)
        logging.info(f"Scheduling a job for {task}")
        manager.schedule_with_runner(task, TaskType(args.mode))
        if args.period == 0:
            # random case #
            await asyncio.sleep(randint(1, 10))
        else:
            await asyncio.sleep(args.period)

if __name__ == '__main__':
    set_logging()
    args = parse_cmdline()
    try:
        asyncio.run(main(args))
    except RuntimeError:
        # Likely you pressed Ctrl-C...
        ...
