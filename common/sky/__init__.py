"""
All code in this package is a refactored, numpy-vectorized version of thorskyutil.py:

https://github.com/jrthorstensen/thorsky/blob/master/thorskyutil.py

utility and miscellaneous time and the sky routines built mostly on astropy.

Copyright John Thorstensen, 2018; offered under the GNU Public License 3.0.

Vectorized by Bryan Miller, Gemini Observatory.
Refactored and clarified by Sergio Troncoso, Gemini Observatory.
Modified by Sebastian Raaphorst, Gemini Observatory, to remove all deviations from Python 3.x PEP8 style guide.
"""

from .altitude import *
from .brightness import *
from .constants import *
from .events import *
from .moon import *
from .sun import *
from .utils import *
