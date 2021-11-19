from enum import Enum
from typing import Callable, List, Union

from astropy.units.quantity import Quantity
import numpy as np

import astropy.units as u
from astropy.time import Time

import numbers


class ComparableEnum(Enum):
    def __gt__(self, other):
        try:
            return self.value > other.value
        except:
            pass
        try:
            if isinstance(other, numbers.Real):
                return self.value > other
        except:
            pass
        return NotImplemented

    def __lt__(self, other):
        try:
            return self.value < other.value
        except:
            pass
        try:
            if isinstance(other, numbers.Real):
                return self.value < other
        except:
            pass
        return NotImplemented

    def __ge__(self, other):
        try:
            return self.value >= other.value
        except:
            pass
        try:
            if isinstance(other, numbers.Real):
                return self.value >= other
            if isinstance(other, str):
                return self.name == other
        except:
            pass
        return NotImplemented

    def __le__(self, other):
        try:
            return self.value <= other.value
        except:
            pass
        try:
            if isinstance(other, numbers.Real):
                return self.value <= other
            if isinstance(other, str):
                return self.name == other
        except:
            pass
        return NotImplemented

    def __eq__(self, other):
        if self.__class__ is other.__class__:
            return self == other
        try:
            return self.value == other.value
        except:
            pass
        try:
            if isinstance(other, numbers.Real):
                return self.value == other
            if isinstance(other, str):
                return self.name == other
        except:
            pass
        return NotImplemented


class SB(ComparableEnum):
    SB20 = 0.2
    SB50 = 0.5
    SB80 = 0.8
    SBANY = 1.0


class CC(ComparableEnum):
    CC50 = 0.5
    CC70 = 0.7
    CC80 = 0.8
    CCANY = 1.0


class IQ(ComparableEnum):
    IQ20 = 0.2
    IQ70 = 0.7
    IQ85 = 0.85
    IQANY = 1.0


class WV(ComparableEnum):
    WV20 = 0.2
    WV50 = 0.5
    WV80 = 0.8
    WVANY = 1.0


def conditions_parser(conditions: str) -> tuple:
    """
    Parser for the conditions string retrieve from XML files.
    Return a tuple for each condition (iq, cc, bg, wv)
    """
    def parser_by_instance(condition: str, 
                           parser: Callable[[Union[str, float]], Enum]) -> Union[np.ndarray, Enum]:
        if isinstance(condition, np.ndarray):
            return np.array(list(map(parser, condition)))
        elif isinstance(condition, str) or isinstance(condition, float):
            return parser(condition)
        else:
            raise ValueError('Must be type str, float, or np.ndarray')
    
    def iq_parser(iq: str) -> IQ:
        return IQ.IQANY if 'ANY' in iq or iq == 'NULL' else IQ(float(iq)/100)
    
    def cc_parser(cc: str) -> CC:
        return CC.CCANY if 'ANY' in cc or cc == 'NULL' else CC(float(cc)/100)

    def sb_parser(sb: str) -> SB:
        return SB.SBANY if 'ANY' in sb or sb == 'NULL' else SB(float(sb)/100)
    
    def wv_parser(wv: str) -> WV:
        return WV.WVANY if 'ANY' in wv or wv == 'NULL' else WV(float(wv)/100)
    
    str_iq, str_cc, str_sb, str_wv = conditions.split(',')
    return (parser_by_instance(str_sb, sb_parser),
            parser_by_instance(str_cc, cc_parser),
            parser_by_instance(str_iq, iq_parser),
            parser_by_instance(str_wv, wv_parser))


class SkyConditions:
    """
    Sky constraints for an observation
    """
    def __init__(self, 
                 sb: SB = SB.SBANY,
                 cc: CC = CC.CCANY,
                 iq: IQ = IQ.IQANY,
                 wv: WV = WV.WVANY):
        self.sb = sb
        self.cc = cc
        self.iq = iq
        self.wv = wv

    def __str__(self):
        return f'{str(self.sb.name)},{str(self.cc.name)},{str(self.iq.name)},{str(self.wv.name)}'

    def __repr__(self):
        return f'Conditions({str(self.sb)},{str(self.cc)},{str(self.iq)},{str(self.wv)})'


class WindConditions:
    """
    Wind constraints for the night
    """
    def __init__(self, 
                 wind_separation: Quantity,
                 wind_speed: Quantity,
                 wind_direction: Quantity,
                 time_blocks: List[Time]):
        self.wind_separation = wind_separation
        self.wind_speed = wind_speed
        self.wind_direction = wind_direction
        self.time_blocks = time_blocks

    def get_wind_conditions(self, azimuth) -> np.ndarray:
     
        wwind = np.ones(len(azimuth))

        ii = np.where(np.logical_and(self.wind_speed > 10 * u.m / u.s,
                                     np.logical_or(abs(azimuth - self.wind_direction) <= self.wind_separation,
                                                   360. * u.deg - abs(azimuth - self.wind_direction)
                                                   <= self.wind_separation)))[0]
        if len(ii) != 0:
            wwind[ii] = 0

        return wwind


class Conditions:
    def __init__(self,
                 sky: SkyConditions,
                 wind: WindConditions):
        self.sky = sky
        self.wind = wind


from random import randint
def random_wv():
    value = randint(0, 3)
    if value == 0:
        return WV.WV20
    if value == 1:
        return WV.WV50
    if value == 2:
        return WV.WV80
    if value == 3:
        return WV.WVANY


