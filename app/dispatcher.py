import functools

from .config import config

from app.core.scheduler.modes import SchedulerModes

from app import app
from app.core.scheduler.modes import SchedulerModes
from app.process_manager import ProcessManager
from .config import config

def dispatch_with(mode: SchedulerModes):
    # Setup scheduler mode
    try:
        mode = SchedulerModes[config.mode.upper()]
    except:
        raise ValueError('Mode is Invalid!')
    def decorator_dispatcher(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            scheduler = func(*args, **kwargs) 
            scheduler.set_mode(mode.value)
            return scheduler 
        return wrapper
    return decorator_dispatcher
