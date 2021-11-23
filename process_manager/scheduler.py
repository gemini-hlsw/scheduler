import time
from random import randrange


class Scheduler():

    def new_schedule(self, name):
        time.sleep(randrange(10))
        return(f'schedule {name}')
