from app.process_manager import TaskType
from datetime import datetime

task_queue = []


class Observe:
    def __init__(self, *args, **kwargs):
        pass
    # TODO: Setup the easiest way to mock a subscription change
    # probably would need a graphql server

    @staticmethod
    def start():
        while(True):
            i = input("Enter a command: ")
            if i in 'standard':
                task_queue.append((datetime.now(), TaskType.STANDARD))
            elif i in 'realtime':
                task_queue.append((datetime.now(), TaskType.REALTIME))
            else:
                print("Invalid command")
                continue
