from dataclasses import dataclass, field
from typing import final, Callable, Dict

import astropy.units as u
import numpy as np
import numpy.typing as npt
from astropy.coordinates import Angle
from lucupy.minimodel import ALL_SITES, Site
from lucupy.types import MinMax


