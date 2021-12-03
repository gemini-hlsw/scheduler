from runner import StandardRunner
from multiprocessing import Process
from task import SchedulerTask


class ProcessManager:
    """
    Main handler for each runner, which is responsible for scheduling the task. 
    """
    def __init__(self, timeout: int, size: int):
        self.realtime_runner = StandardRunner(1, timeout)
        self.standard_runner = StandardRunner(size, timeout)

    def schedule_with_runner(self, task: SchedulerTask, mode: str):
        """
        Schedule a task with the corresponding runner for the given mode.
        """
        # TODO: Probably good for enums but right now the original input is a string
        # so it seems unnecessary to use enums?
        if mode == 'realtime':
            return self.realtime_runner.schedule(Process(target=task.target), task.timeout)
        elif mode == 'standard':
            return self.standard_runner.schedule(Process(target=task.target), task.timeout)
        else:
            raise Exception(f'Invalid mode {mode}')
    
    def shutdown(self):
        """
        Callback for shutting down the process manager.
        """
        self.realtime_runner.terminate_all()
        self.standard_runner.terminate_all()
