from enum import Enum


class ToOType(Enum):
    """The different possible target of opportunity flags"""
    NONE = "none"
    STANDARD = "standard"
    RAPID = "rapid"
