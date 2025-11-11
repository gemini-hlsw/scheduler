from scheduler.core.meta import Singleton


class SchedulerQueueClient(metaclass=Singleton):

    def new_schedule(self):
        pass



schedule_queue = SchedulerQueueClient()