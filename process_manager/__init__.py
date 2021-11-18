import datetime
import multiprocessing as mp
from bin import SchedulingBin, RealTimeSchedulingBin, SchedulerTask
from heapq import heappush, heapreplace, heappop


class ProcessManager:
    def __init__(self, config: dict = None) -> None:
        
        # TODO: if these are fix should we use Enum?
        self.bins = {'realtime': [],
                     'standard': []}
        if config is not None:
            if config['realtime_bins']['amount'] > 2:
                raise ValueError('Only one bin for realtime mode')
            
            for _ in range(config['realtime_bins']['amount']):
                self.bins['realtime'].append(RealTimeSchedulingBin(config['realtime_bins']['start'],
                                                                   config['realtime_bins']['float_after'],
                                                                   config['weekly_bins']['length']))
            for _ in range(config['biannual_bins']['amount']):
                self.bins['standard'].append(SchedulingBin(config['biannual_bins']['start'],
                                                           config['biannual_bins']['float_after'],
                                                           config['biannual_bins']['length'],
                                                           config['biannual_bins']['n_threads'],
                                                           config['biannual_bins']['bin_size']))
            for _ in range(config['weekly_bins']['amount']):
                self.bins['standard'].append(SchedulingBin(config['weekly_bins']['start'],
                                                           config['weekly_bins']['float_after'],
                                                           config['weekly_bins']['length'],
                                                           config['weekly_bins']['n_threads'],
                                                           config['weekly_bins']['bin_size']))
            for _ in range(config['custom_bins']['amount']):
                self.bins['standard'].append(SchedulingBin(config['custom_bins']['start'],
                                                           config['custom_bins']['float_after'],
                                                           config['custom_bins']['length'],
                                                           config['custom_bins']['n_threads'],
                                                           config['custom_bins']['bin_size']))

    def new_bin(self,
                bin_type: str,
                start: datetime.datetime,
                float_after: datetime.timedelta,
                length: datetime.timedelta,
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
            heappush(self.bins['realtime'][0].priority_queue, task)
        else:
            for bin in self.bins['standard']:
                if bin.start <= task.start_time and bin.start + bin.length < task.end_time:
                    # TODO: This assume that one task run in one thread, which
                    if (any(task.priority > running_task.priority for running_task in bin.running_tasks) and
                       len(bin.running_task) == bin.bin_size):

                        # remove lower priority task from running #
                        lower_priority_task = sorted(bin.priority_queue)[0]
                        lower_priority_process = lower_priority_task.process
                        lower_priority_process.join()
                        if lower_priority_process.is_alive():
                            lower_priority_process.terminate()
                            bin.running_task.remove(lower_priority_process)

                    heappush(bin.priority_queue, task)
                    return
            
            raise RuntimeError('No bin to accommodate this task')

    def run(self) -> None:
        while True:
            for bin in self.bins['realtime'] + self.bins['standard']:
                # TODO: Right now the collection of the output process is done after all task 
                # are set to running. This surely be a problem when Task Management is implemented
                while len(bin.priority_queue) > 0:
                    # remove highest priority task from queue #
                    task = heappop(bin.priority_queue)
                    # run process to initialize scheduling task
                    bin.run_task(task)
                plans = bin.wait()
                print(plans)
