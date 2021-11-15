import time
import multiprocessing as mp
from bin import SchedulingBin, RealTimeSchedulingBin, SchedulerTask


class ProcessManager:
    def __init__(self, config: dict = None) -> None:
        
        self.bins = []
        if config is not None:
            if config['realtime_bins'] > 2:
                raise ValueError('Only one bin for realtime mode')
            
            for i in range(config['realtime_bins']['amount']):
                self.bins.append(RealTimeSchedulingBin(0, 0, 0,
                                 config['realtime_bins']['n_threads'],
                                 config['realtime_bins']['bin_size']))
            for i in range(config['semestral_bins']['amount']):
                self.bins.append(SchedulingBin(0, 0, 0,
                                 config['semestral_bins']['n_threads'],
                                 config['semestral_bins']['bin_size']))
            for i in range(config['week_bins']['amount']):
                self.bins.append(SchedulingBin(0, 0, 0,
                                 config['week_bins']['n_threads'],
                                 config['week_bins']['bin_size']))
            for i in range(config['custom_bins']['amount']):
                self.bins.append(SchedulingBin(0, 0, 0,
                                 config['custom_bins']['n_threads'],
                                 config['custom_bins']['n_threads']))

    def new_bin(self, start, float_after, length, number_threads, bin_size):
        self.bins.append(SchedulingBin(start, float_after, length, number_threads, bin_size))
    
    def add_task(self, task: SchedulerTask) -> None:
        if task.mode == 'realtime':
            self.bins[0].priority_queue.append(task)
        
        else:
            for bin in self.bins[1:]:
                if bin.start <= task.start_time and bin.start + bin.length < task.finish_time:
                    bin.priority_queue.append(task)
                    break

    def run(self):
        while True:
            for bin in self.bins:
                if len(bin.priority_queue) > 0:
                    for task in bin.priority_queue:
                        # check if the task need floating ""
                        if task.start_time + task.float_after < time.time():
                            task.start_time += task.float_after
                            # check if task still belong in the bin #
                            if task.start_time + task.float_after > bin.start + bin.length:
                                bin.priority_queue.remove(task)
                                # check if task is runnning #
                                # this needs to be done as Process()
                                continue
                        p = mp.Process(target=task.scheduler.new_schedule, args=(task.job_id,))
                        p.start()
                        p.join(task.timeout)
                        if p.is_alive():
                            p.terminate()
                            bin.priority_queue.remove(task)
                            print(f'Task {task.job_id} timed out')
                        else:
                            bin.priority_queue.remove(task)
                            print(f'Task {task.job_id} finished')
