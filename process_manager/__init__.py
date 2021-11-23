from datetime import datetime, timedelta
import multiprocessing as mp
from bin import SchedulingBin, RealTimeSchedulingBin, SchedulerTask, BinType
from heapq import heappush, nsmallest, heappop
import asyncio
import signal
from typing import  List, Tuple, Optional

class ProcessManager:
    def __init__(self, config: dict = None) -> None:
        
        # TODO: if these are fix should we use Enum?
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

    def new_bin(self,
                bin_type: str,
                start: datetime,
                float_after: timedelta,
                length: timedelta,
                number_threads: int,
                bin_size: int) -> None:
        if bin_type == 'realtime':
            if len(self.bins['realtime']) > 1:
                raise ValueError('Only one bin for realtime mode')
            self.bins['realtime'].append(RealTimeSchedulingBin(start, float_after, length, number_threads, bin_size))
       
        else:
            if bin_type not in self.bins:
                raise ValueError('Bin type not supported')
            self.bins[bin_type].append(SchedulingBin(start, float_after, length, number_threads, bin_size))
    
    def add_task(self, task: SchedulerTask) -> None:
        if task.is_realtime:
            # self.bins['realtime'].priority_queue.append(task)
            heappush(self.bins[BinType.REALTIME][0].priority_queue, task)
        else:
            for bin in self.bins[BinType.STANDARD]:
                if bin.start <= task.start_time and bin.start + bin.length < task.end_time:
                    # TODO: This assume that one task run in one thread, which
                    if (any(task.priority > running_task.priority for running_task in bin.running_tasks) and
                       len(bin.running_task) == bin.bin_size):

                        # remove lower priority task from running #
                        lower_priority_task = nsmallest(1, bin.running_tasks, key=lambda x: x.priority)
                        lower_priority_process = lower_priority_task.process
                        lower_priority_process.join()
                        if lower_priority_process.is_alive():
                            lower_priority_process.terminate()
                            bin.running_task.remove(lower_priority_process)

                    heappush(bin.priority_queue, task)
                    return
            
            raise RuntimeError('No bin to accommodate this task')

    def run(self) -> None:
        
        #done = asyncio.Event()

        #def shutdown():
        #    done.set()
            #the_bin.shutdown()
        #    asyncio.get_event_loop().stop()

        #asyncio.get_event_loop().add_signal_handler(signal.SIGINT, shutdown)

        #while not done.is_set():
            

        while True:
            # TODO: This process need to be asyncronous and done by bi
            for bin in self.bins[BinType.REALTIME] + self.bins[BinType.STANDARD]:
                # TODO: Right now the collection of the output process is done after all task
                # are set to running. This surely be a problem when Task Management is implemented
                while len(bin.priority_queue) > 0:
                    # remove highest priority task from queue #
                    task = heappop(bin.priority_queue)
                    # run process to initialize scheduling task
                    bin.run_task(task)
                
                # Add floating condition: This is done one time but should be done periodically?
                if datetime.now() > bin.start + bin.float_after:
                    bin.float_bin()
                     
                plans = bin.wait()
                print(plans)
