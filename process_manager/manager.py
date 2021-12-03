from runner import StandardRunner
from multiprocessing import Process


class ProcessManager:
    """
    Main handler for each runner, which is responsible for scheduling the task. 
    """
    def __init__(self, timeout, size):
        self.realtime_runner = StandardRunner(1, timeout)
        self.standard_runner = StandardRunner(size, timeout)

    def schedule_with_runner(self, task, mode):
        """
        Schedule a task with the corresponding runner for the given mode.
        """
        if mode == 'realtime':
            return self.realtime_runner.schedule(Process(target=task.target), task.timeout)
        elif mode == 'standard':
            return self.standard_runner.schedule(Process(target=task.target), task.timeout)
        else:
            raise Exception(f'Invalid mode {mode}')
    
    def shutdown(self):
        self.realtime_runner.terminate_all()
        self.standard_runner.terminate_all()
