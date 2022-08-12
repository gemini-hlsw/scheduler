from datetime import timedelta

from common.minimodel.observation import Observation

from .geminiproperties import GeminiProperties


def with_igrins_cal(func):
    def add_calibration(self):
        if GeminiProperties.Instruments.IGRINS in self.required_resources() and self.partner_used() > 0:
            return func(self) + timedelta(seconds=(1 / 6))
        return func(self)
    return add_calibration


class GeminiObservation(Observation):
    """
    A Gemini-specific extension of the Observation class.
    """
    @with_igrins_cal
    def total_used(self) -> timedelta:
        return super().total_used()
