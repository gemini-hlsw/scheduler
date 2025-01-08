from scheduler.services import logger_factory
from scheduler.services.celestial_monitor.celestial_monitor import CelestialMonitor

logger = logger_factory.create_logger(__name__)

class OCSCelestialMonitor(CelestialMonitor):

    def __init__(self, subdir):
        super.__init__()
        self._subdir = subdir


    def _load_too_activations(self, site, name):
        path = self._subdir / name

        try:
            with open(path) as f:
                for line in f:
                    obs_id, date, target = line.split(',')

        except FileNotFoundError:
            logger.error(f'Time loss file not available: {path}')

    def get_toos_for_the_night(self):
