from scheduler.core.meta import Singleton


class SchedulerQueueClient(metaclass=Singleton):

    def add_schedule_event(self):
        pass



schedule_queue = SchedulerQueueClient()