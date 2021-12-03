
import dataclasses
import functools
import logging
from itertools import count

from process import ProcessTask, Result

job_counter = count()

@dataclasses.dataclass(order=True)
class Job:
    process: ProcessTask = dataclasses.field(compare=False)
    sequence: int = dataclasses.field(compare=False,
                                      default_factory=functools.partial(next, job_counter))

    def __repr__(self):
        return f"Job-{self.sequence}"


class StandardRunner:
    def __init__(self, size, timeout=None):
        self.max_jobs = size
        self.jobs = []
        self.callbacks = []
        self.timeout = timeout
    
    def add_done_callback(self, callback):
        """
        Adds a callback that will be invoked when a job is finished. Useful
        to control the scheduling of new jobs.
        """
        self.callbacks.append(callback)

    async def terminated_job(self, job, ptask):
        """
        Called when a job has finished.

        Runs any added callbacks in order to notify that a new slot is free
        for scheduling.
        """
        res = ptask.result
        if res != Result.TERMINATED:
            # Terminated jobs had been evicted earlier (see maybe_evict) and we
            # don't need to do anything else about them.
            # The others need a bit more of work
            if res == Result.TIMEOUT:
                logging.warning(f"  - Task {job} timed out!")
            else:
                logging.info(f"  - Task {job} is done")

            try:
                del self.jobs[self.jobs.index(job)]
            except ValueError:
                logging.warning(f"  - Job {job} was not in the heap any longer!")

            # Notify that we're ready to queue something new
            for callback in self.callbacks:
                callback()
            
    def _run_job(self, proc, timeout):
        """
        Prepares a job and starts its associated process.

        Only internal use.
        """
        ptask = ProcessTask(proc)
        job = Job(ptask)
        proc.name = f'Job-{job.sequence}'
        ptask.add_done_callback(functools.partial(self.terminated_job, job))
        ptask.start(timeout=timeout)

        return job
    
    def maybe_evict(self, priority):
        """
        Evict and kill the lowest priority job if the specified
        priority is higher.
        """
        try:
            # Assume that lower priority number means higher priority
            lowest = max(self.jobs)
            if lowest.priority > priority:
                logging.debug(f"  - Evicting job {lowest}")
                lowest.process.terminate()
                del self.jobs[self.jobs.index(lowest)]
        except ValueError:
            # No jobs...
            ...
    
    def evict(self):
        """
        Kill the latest job
        """
        try:
            job = self.jobs[-1]
            job.process.terminate()
            del self.jobs[-1]
        except ValueError:
            logging.warning(f"  - No jobs to evict!")

    def terminate(self, task):
        """
        Terminates a task from the queue.
        """
        try:
            job = self.jobs[self.jobs.index(task)]
            job.process.terminate()
            del self.jobs[self.jobs.index(task)]
        except ValueError:
            logging.warning(f"  - Task {task} was not in the heap any longer!")
    
    def schedule(self, process, timeout):
        """
        Attempts scheduling a new job.

        Returns True if the task was successfully scheduled,
        False otherwise.
        """
        if len(self.jobs) == self.max_jobs:
            self.evict()
        if len(self.jobs) < self.max_jobs:
            self.jobs.append(self._run_job(process, timeout))
            return True
        else:
            return False
    
    def terminate_all(self):
        """
        Ends all running processes.
        """
        for job in self.jobs:
            job.process.terminate()

        self.jobs = []
