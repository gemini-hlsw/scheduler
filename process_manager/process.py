import asyncio
import threading
import logging
from enum import Enum
from functools import partial

class NotRunningError(Exception):
    ...

class CantStartError(Exception):
    ...

class Result(Enum):
    SUCCESS = 0
    TIMEOUT = 1
    TERMINATED = 2
    ERROR = 3

async def delayed_action(wait_time, action):
    "Executes an arbitrary `action` after `wait_time` seconds have passed"
    await asyncio.sleep(wait_time)
    res = action()
    if asyncio.iscoroutine(res):
        await res

class ProcessTask:
    '''
    A class that allows an async process fine control over a
    multiprocessing.Process

    Initialize by passing an instance of Process, that must not
    be running.
    '''
    def __init__(self, process):
        self.process = process
        self.done = asyncio.Event()
        self.running_task = None
        self.done_callbacks = []
        self.result = None

    def add_done_callback(self, callback):
        """
        Add a callback to be invoked when the process finishes by
        any reason. The callback must accept one argument (the ProcessTask
        instance itself)
        """
        self.done_callbacks.append(callback)
        return self

    def success(self):
        "True if the process ended on its own without issues"
        return self.result == Result.SUCCESS

    def error(self):
        "True if the process ended on its own with an exit status != 0"
        return self.result == Result.ERROR

    def timed_out(self):
        "True if the process timed out"
        return self.result == Result.TIMEOUT

    def killed(self):
        "True if the process was actively terminated"
        return self.result == Result.TERMINATED

    async def wait(self):
        "Will block until the process finishes (or is terminated)"
        if not self.process.is_alive():
            raise NotRunningError("Can't wait on a process that is not running")

        await self.done.wait()

    def terminate(self, result=Result.TERMINATED):
        """
        Cancel the task waiting for the process to finish. As a side
        effect, this will terminate the process itself.

        Sets the result to the specified value.
        """
        if self.process.is_alive():
            if self.result is None:
                self.result = result
            if self.running_task:
                self.running_task.cancel()

    async def _set_done(self):
        """
        Notifies any task blocked waiting for this processes, and invokes any
        callback that was added for asynchronous notification.

        Only for internal use.
        """
        self.done.set()
        for callback in self.done_callbacks:
            res = callback(self)
            if asyncio.iscoroutine(res):
                await res

    async def await_process(self):
        """
        A coroutine meant to be run as a background task.

        Sits waiting for the process to finish. It sets the result to SUCCESS or ERROR
        in case that the process finishes on its own.

        If cancelled, it will terminate the (presumably still running) process.

        In any case, it will signal that the process is done to anyone waiting.
        """
        try:
            # This needs to be run on a thread, because Process.join() is a synchronous
            # method and it would lock the whole event loop otherwise.
            await asyncio.get_event_loop().run_in_executor(None, self.process.join)
        except asyncio.CancelledError:
            self.process.terminate()
        finally:
            self.running_task = None
            if self.result is None:
                self.result = Result.SUCCESS if self.process.exitcode == 0 else Result.ERROR
            await self._set_done()

    def start(self, timeout=None):
        """
        Starts the process and a background task will wait for it to finish. Optionally, if
        `timeout` is not None, a second task will be launched to ensure that the process is
        done within the deadline.

        Can only be invoked once. It will raise CantStartError if the process was already
        alive, or if the task has finished.
        """
        if self.process.is_alive():
            raise CantStartError("The process is running already")
        elif self.done.is_set():
            raise CantStartError("This process ran already")

        self.process.start()
        self.running_task = asyncio.create_task(self.await_process(), name=f"Running {self.process.name}")

        if timeout is not None:
            asyncio.create_task(delayed_action(timeout, partial(self.terminate, Result.TIMEOUT)),
                                name=f"Timeout for {self.process.name}")
