from enum import auto, IntEnum


class QAState(IntEnum):
    """
    These correspond to the QA States in the OCS for Observations.
    Entries in the obs log should be made uppercase for lookups into
    this enum.
    """
    NONE = auto()
    UNDEFINED = auto()
    FAIL = auto()
    USABLE = auto()
    PASS = auto()
    # TODO: Not in original mini-model description, but returned by OCS.
    CHECK = auto()
