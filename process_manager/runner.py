import dataclasses
import functools
import logging
from itertools import count
from multiprocessing import Process

from process import ProcessTask, Result

job_counter = count()

@dataclasses.dataclass(order=True)
class Job:
    priority: int
    process: ProcessTask = dataclasses.field(compare=False)
    sequence: int = dataclasses.field(compare=False,
                                      default_factory=functools.partial(next, job_counter))

    def __repr__(self):
        return f"Job-{self.sequence} {{prio={self.priority}}}"


class PriorityRunner:
    """
    A class that controls running processes according a certain priority.

    The class is instantiated with a certain `size`, indicating the maximum
    number of concurrent jobs at a certain time.

    Jobs are accepted unconditionally as long as there's still room for new
    jobs. If scheduling is attempted with a full set, then:

      * If the new task's priority is lower than any of the running ones,
        the scheduling will be rejected.
      * Otherwise, the lowest priority job currently running is evicted
        (and its associated process terminated) to make room for the new
        task.
    """
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

    def _run_job(self, proc, priority, timeout):
        """
        Prepares a job and starts its associated process.

        Only internal use.
        """
        ptask = ProcessTask(proc)
        job = Job(priority, ptask)
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

    def schedule(self, process, priority, timeout):
        """
        Attempts scheduling a new job.

        Returns True if the task was successfully scheduled,
        False otherwise.
        """
        if len(self.jobs) == self.max_jobs:
            self.maybe_evict(priority)
        if len(self.jobs) < self.max_jobs:
            self.jobs.append(self._run_job(process, priority, timeout))
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
