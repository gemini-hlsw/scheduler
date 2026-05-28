"""Temporary adapter layer between lucupy types and sight's pure compute.

The realtime collector path needs to call sight's ``calculate_night_events_for_night``,
``calculate_stage1``, and ``calculate_visibility`` without any DB session. Those
functions take ORM-shaped objects (sight's SQLAlchemy ``Site`` / ``Target`` /
``NightEvent``); this package provides duck-typed ``SimpleNamespace`` shims built
from lucupy ``Site`` enum values and lucupy ``SiderealTarget`` /
``NonsiderealTarget``.

This is intentionally a TEMPORARY shim. The longer-term fix is one of:
  - make sight's compute functions accept lucupy types directly (preferred);
  - hide RT compute behind a proper service interface that owns this glue.

Until then, anything that depends on this folder should treat it as scaffolding
and not grow new responsibilities here.
"""
