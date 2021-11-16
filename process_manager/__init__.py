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
                    print(type(bin))
                    heappush(bin.priority_queue, task)
                    break

    def run(self) -> None:
        while True:
            for bin in self.bins['realtime'] + self.bins['standard']:
                while len(bin.priority_queue) > 0:
                    
                    task = heappop(bin.priority_queue)
                    if task.start_time + bin.float_after < datetime.datetime.now():
                        task.start_time += bin.float_after
                        # check if task still belong in the bin #
                        if task.start_time + bin.float_after > bin.start + bin.length:
                            # check if task is runnning #
                            # this needs to be done as Process()
                            continue
                    p = mp.Process(target=task.scheduler.new_schedule, args=(task.job_id,))
                    p.start()
                    p.join(task.timeout.total_seconds())
                    if p.is_alive():
                        p.terminate()
                        #heapreplace(bin.priority_queue, task)
                        print(f'Task {task.job_id} timed out')
                    else:
                        #heapreplace(bin.priority_queue, task)
                        print(f'Task {task.job_id} finished')
