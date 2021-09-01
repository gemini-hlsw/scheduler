import numpy as np
import astropy.units as u

class SkyConditions:
    """
    Sky constraints for an observation 
    """
    def __init__(self, 
                 image_quality: float,
                 brightness: float,
                 cloud_conditions: float,
                 water_vapour: float) -> None:
        self.image_quality = image_quality
        self.cloud_conditions = cloud_conditions
        self.brightness = brightness
        self.water_vapour = water_vapour
      
class WindConditions:
    """
    Wind constraints for the night
    """
    def __init__(self, 
                 wind_separation: float,
                 wind_speed: float,
                 wind_direction: float,
                 time_blocks: float,):
        self.wind_separation = wind_separation
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.time_blocks = time_blocks

    def get_wind_conditions(self, azimuth):
        
        if np.asarray(self.wind_speed).ndim == 0:
            speed = np.full(len(azimuth), self.wind_speed.to(u.m / u.s).value) * u.m / u.s
        
        wwind = np.ones(len(azimuth))
        ii = np.where(np.logical_and(speed > 10 * u.m / u.s,
                                    np.logical_or(abs(azimuth - self.wind_direction) <= self.wind_separation,
                                                360. * u.deg - abs(azimuth - self.wind_direction) <= self.wind_separation)))[0][:]
        if len(ii) != 0:
            wwind[ii] = 0

        return wwind