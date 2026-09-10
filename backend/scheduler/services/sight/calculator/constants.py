from typing import Optional

from gpp_client.generated.enums import Instrument

SITE_ID_TO_KEY = {
    1: "GN",
    2: "GS",
}

SITE_KEY_TO_ID = {
    "GN": 1,
    "GS": 2,
}

# GPP Instrument -> Sight site key, for instruments whose enum name does not end
# in NORTH/SOUTH (those are handled by the suffix check below). Mirrors and
# extends ``GppProgramProvider._site_for_inst``.
#
# This lives here rather than beside its original caller in night_monitor:
# importing anything from that package runs its __init__, which pulls in
# event_consumer -> graphql_mid.types -> engine and cycles back through
# scheduler_queue_client. This module imports nothing from scheduler, so any
# consumer can use it without dragging that graph in.
_INSTRUMENT_TO_SITE_KEY = {
    Instrument.FLAMINGOS2: "GS",
    Instrument.GHOST: "GS",
    Instrument.GPI: "GS",
    Instrument.GSAOI: "GS",
    Instrument.ZORRO: "GS",
    Instrument.SCORPIO: "GS",
    Instrument.GNIRS: "GN",
    Instrument.NIRI: "GN",
    Instrument.IGRINS2: "GN",
    Instrument.ALOPEKE: "GN",
    Instrument.MAROON_X: "GN",
}


def site_key_from_instrument(instrument) -> Optional[str]:
    """Derive the Sight site key (``"GN"`` / ``"GS"``) from an
    observation's instrument.
    """
    if instrument is None:
        return None
    name = getattr(instrument, "name", str(instrument)).upper()
    if name.endswith("NORTH"):
        return "GN"
    if name.endswith("SOUTH"):
        return "GS"
    return _INSTRUMENT_TO_SITE_KEY.get(instrument)
