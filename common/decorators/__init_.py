
import numpy as np
from datetime import timedelta 

def scalar_input(func):
    """
    Decorator to convert a function that returns a tuple to a function that returns a scalar.
    TODO: TO IMPLEMENT THIS IS NECESSARY TO REMOVE ASTROPY outputs or made them numpy compatible.
    """

    def wrapper(*args, **kwargs):

        # transform the input to numpy
        np_args = [np.asarray(arg) for arg in args]
        if any(arg.ndim == 0 for arg in np_args):
            args = [arg[np.newaxis] for arg in np_args]
           
        return np.squeeze(func(*args, **kwargs))
    return wrapper


def with_igrins_cal(func):
    def add_calibration(self):
        if ('IGRINS' in [res.id for res in self.required_resources()] and self.partner_used() > 0):
            return func + timedelta(seconds=(1 / 6))
        return func
    return add_calibration
