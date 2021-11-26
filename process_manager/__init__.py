from datetime import datetime, timedelta

from bin import SchedulingBin, RealTimeSchedulingBin, SchedulerTask, BinType
import asyncio
import signal
from typing import  List, Tuple, Optional
from provider import TaskProvider
import logging


class ProcessManager:
    def __init__(self, config: List = None, period: int = 5) -> None:
        
        self.bins = {BinType.REALTIME: [],
                     BinType.STANDARD: []}
        if config is not None:

            for cfg in config:
                if cfg.bin_type is BinType.REALTIME:
                    self.bins[cfg.bin_type].append(RealTimeSchedulingBin(cfg.start,
                                                                         cfg.float_after,
                                                                         cfg.length))
                elif cfg.bin_type is BinType.STANDARD:
                    self.bins[cfg.bin_type].append(SchedulingBin(cfg.start,
                                                                 cfg.float_after,
                                                                 cfg.length,
                                                                 cfg.number_threads,
                                                                 cfg.bin_size))
                else:
                    raise ValueError('BinType unknown')
        self.period = period
        self.queue = asyncio.Queue()
        self.task_provider = TaskProvider()


    def new_bin(self,
                bin_type: str,
                start: datetime,
                float_after: timedelta,
                length: timedelta,
                number_threads: int,
                bin_size: int) -> None:
        """
        Adds a new bin to the process manager
        """
        if bin_type == 'realtime':
            if len(self.bins['realtime']) > 1:
                raise ValueError('Only one bin for realtime mode')
            self.bins['realtime'].append(RealTimeSchedulingBin(start, float_after, length, number_threads, bin_size))
       
        else:
            if bin_type not in self.bins:
                raise ValueError('Bin type not supported')
            self.bins[bin_type].append(SchedulingBin(start, float_after, length, number_threads, bin_size))
    
    async def run(self) -> None:
        """
        Main driver for the process manager.

        This runs in a loop asyncrhonously adding tasks to each bin accordinly.
        """

        done = asyncio.Event()

        def shutdown(ptask, ctask):
            done.set()
            # Kill all task in the bins 
            for bin in self.bins[BinType.REALTIME] + self.bins[BinType.STANDARD]:
                bin.shutdown()
            # Cancel provider tasks #
            ctask.cancel()
            ptask.cancel()

            # Stop the loop
            asyncio.get_event_loop().stop()

        ptask = asyncio.create_task(self.task_provider.producer(self.queue, 1))
        ctask = asyncio.create_task(self.task_provider.consumer(self.queue))

        loop = asyncio.get_event_loop()
        loop.add_signal_handler(signal.SIGINT, shutdown, ptask, ctask)

        while not done.is_set():
            
            task = await ctask
            print(task)
            if task is not None:
                if task.is_realtime:

                    logging.info(f"Real Time Scheduling a job for {task}")
                    self.bins[BinType.REALTIME].schedule_with_runner(task)
                    await asyncio.sleep(self.period)
                else:
                    logging.info(f"Standard Scheduling a job for {task}")
                    for bin in self.bins[BinType.STANDARD]:
                        if bin.start <= task.start_time and bin.start + bin.length < task.end_time:
                            bin.schedule_with_runner(task)
                            break
                    await asyncio.sleep(self.period)
