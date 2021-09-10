import numpy as np
import astropy.units as u
from enum import Enum
import numbers
from typing import Callable, Union

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
                           parser: Callable[[str],Enum])-> Union[np.ndarray,Enum]:
        if isinstance(condition, np.ndarray):
            return np.array(list(map(parser,condition)))
        elif isinstance(condition,str) or isinstance(condition,float):
            return parser(condition)
        else:
            raise ValueError('Must be type str, float, or np.ndarray')
    
    def find_values(values :str) -> float:
        return float(''.join(x for x in values if x.isdigit()))/100
    
    def iq_parser(iq: str) -> IQ:
        
        if 'ANY' in iq or 'NULL' in iq:
            return IQ.IQANY
        else:
            _iq = find_values(iq)
            if 0.0 < _iq <= 0.2:
                return IQ.IQ20
            elif 0.2 < _iq <= 0.7:
                return IQ.IQ70
            elif 0.7 < _iq <= 0.85:
                return IQ.IQ85
            else:
                return IQ.IQANY
    
    def cc_parser(cc: str) -> CC:
        
        if 'ANY' in cc or 'NULL' in cc:
            return CC.CCANY
        else:
            _cc= find_values(cc)
            if 0.0 < _cc <= 0.5:
                return CC.CC50
            elif 0.5 < _cc <= 0.7:
                return CC.CC70
            elif 0.7 < _cc <= 0.80:
                return CC.CC80
            else:
                return CC.CCANY

    def sb_parser(sb: str) -> SB:
        
        if 'ANY' in sb or 'NULL' in sb:
            return SB.SBANY
        else:
            _sb= find_values(sb)
            if 0.0 < _sb <= 0.2:
                return SB.SB20
            elif 0.2 < _sb <= 0.5:
                return SB.SB50
            elif 0.5 < _sb <= 0.80:
                return SB.SB80
            else:
                return SB.SB80
    
    def wv_parser(wv: str)-> WV:

        if 'ANY' in wv or 'NULL' in wv:
            return WV.WVANY
        else:
            _wv= find_values(wv)
            if 0.0 < _wv <= 0.2:
                return WV.WV20
            elif 0.2 < _wv <= 0.5:
                return WV.WV50
            elif 0.5 < _wv <= 0.80:
                return WV.WV80
            else:
                return WV.WV80
    
    str_iq, str_cc, str_sb, str_wv = conditions.split(',')
    return (parser_by_instance(str_sb, sb_parser),
            parser_by_instance(str_cc, cc_parser),
            parser_by_instance(str_iq,iq_parser),
            parser_by_instance(str_wv, wv_parser) 
    )


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